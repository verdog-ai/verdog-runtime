from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

import pytest
from report_helpers import call_reports, table_rows

from verdog_runtime.declarations import (
    CallContext,
    CallVisitDefinition,
    EdgeDefinition,
    GraphDefinition,
    NodeContext,
    NodeDefinition,
    PortDefinition,
    Python,
    SubroutineCall,
    SubroutineDefinition,
    Success,
    VisitDefinition,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId
from verdog_runtime.interpreter import Dispatcher

_TRANSITION_LIMIT = 100_000
# Nested output directories intentionally trade unbounded filesystem depth for
# navigability. This bounded test stays below PATH_MAX while calls still exceed
# the reduced Python recursion limit.
_DEEP_CALL_DEPTH = 90
_TEST_RECURSION_LIMIT = 80


@dataclass(frozen=True, slots=True)
class _LoopState:
    pass


@dataclass(frozen=True, slots=True)
class _CallState:
    calls: int = 0


def _deep_local_call_workflow(
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
    /,
    *,
    write_artifacts: bool = False,
    call_id: str = "call",
) -> tuple[
    WorkflowDefinition[int, int, None, object],
    tuple[GraphId, ...],
]:
    """Build a fixed-width sibling chain without recursing in the fixture."""

    graph_ids = (GraphId("deep.main"),) + tuple(
        GraphId(f"deep.main__level_{level:04d}")
        for level in range(1, depth + 1)
    )
    module_names = tuple(
        f"runtime_test_deep_local_call_{level:04d}"
        for level in range(depth + 1)
    )
    params_types = {(".", graph_id): type(None) for graph_id in graph_ids}

    def adapt(
        input: int,
        state: _CallState,
        context: CallContext[None, None, int, int],
        /,
    ) -> Success[int, _CallState]:
        child_output = context.invoke(input + 1)
        if write_artifacts:
            (context.output_dir / "adapter.txt").write_text(
                f"{context.graph_id}: {input}->{child_output}",
                encoding="utf-8",
            )
        return Success(
            output=child_output,
            state=_CallState(calls=state.calls + 1),
        )

    graphs: list[GraphDefinition[int, int, None, object]] = []
    for position, graph_id in enumerate(graph_ids):
        enter = PortDefinition(id=NodeId("enter"))
        exit_ = PortDefinition(id=NodeId("exit"))
        failure = PortDefinition(id=NodeId("failure"))
        if position == depth:
            graph = GraphDefinition[int, int, None, object](
                id=graph_id,
                params_type=type(None),
                enter=enter,
                exit=exit_,
                failure=failure,
                nodes=(),
                edges=(
                    EdgeDefinition(
                        id=EdgeId("done"),
                        source=enter.id,
                        target=exit_.id,
                    ),
                ),
            )
        else:
            call: NodeDefinition[_CallState, object] = NodeDefinition(
                id=NodeId(call_id),
                name="Call next subroutine",
                operation=SubroutineCall(
                    definition_id=graph_ids[position + 1],
                    definition_module=module_names[position + 1],
                    params_types=params_types,
                    profile_arguments={},
                    session_arguments={},
                ),
                state_type=_CallState,
            )
            graph = GraphDefinition[int, int, None, object](
                id=graph_id,
                params_type=type(None),
                enter=enter,
                exit=exit_,
                failure=failure,
                nodes=(call,),
                edges=(
                    EdgeDefinition(
                        id=EdgeId("invoke"),
                        source=enter.id,
                        target=call.id,
                        visit=CallVisitDefinition(implementation=adapt),
                    ),
                    EdgeDefinition(
                        id=EdgeId("done"),
                        source=call.id,
                        target=exit_.id,
                    ),
                ),
            )
        graphs.append(graph)

    for module_name, graph in zip(module_names, graphs, strict=True):
        module = ModuleType(module_name)
        definition = SubroutineDefinition(graph=graph)
        setattr(module, "definition", lambda definition=definition: definition)  # noqa: B010
        monkeypatch.setitem(sys.modules, module_name, module)

    workflow: WorkflowDefinition[int, int, None, object] = WorkflowDefinition(
        id=GraphId("deep.workflow"),
        input_type=int,
        entry=SubroutineCall(
            definition_id=graph_ids[0],
            definition_module=module_names[0],
            params_types=params_types,
            profile_arguments={},
            session_arguments={},
        ),
        configuration=WorkflowConfiguration(),
    )
    return workflow, graph_ids


def _report_chain(report: Path) -> list[Path]:
    chain: list[Path] = []
    children = call_reports(report)
    while children:
        assert len(children) == 1
        report = children[0]
        chain.append(report)
        children = call_reports(report)
    return chain


class _GraphOutputLike(Protocol):
    root: Path
    visits: dict[NodeId, int]


@contextmanager
def _cheap_node_output(
    graph_output: _GraphOutputLike,
    node_id: NodeId,
    node_type: str,
    *,
    status: str = "succeeded",
) -> Generator[Path]:
    """Retain visit ordinals without 100,000 directories and trace writes."""

    del node_type, status
    graph_output.visits[node_id] = graph_output.visits.get(node_id, 0) + 1
    yield graph_output.root


def test_iterative_driver_reaches_100_000_transition_limit_without_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visits = 0

    def loop(
        input: int,
        state: _LoopState,
        _context: NodeContext[None],
        /,
    ) -> Success[int, _LoopState]:
        nonlocal visits
        visits += 1
        return Success(output=input, state=state)

    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    loop_node: NodeDefinition[_LoopState, object] = NodeDefinition(
        id=NodeId("loop"),
        name="Loop",
        operation=Python(),
        state_type=_LoopState,
    )
    graph: GraphDefinition[int, int, None, object] = GraphDefinition(
        id=GraphId("iterative_limit"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(loop_node,),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter_loop"),
                source=enter.id,
                target=loop_node.id,
                visit=VisitDefinition(implementation=loop),
            ),
            EdgeDefinition(
                id=EdgeId("loop_again"),
                source=loop_node.id,
                target=loop_node.id,
                visit=VisitDefinition(implementation=loop),
            ),
        ),
    )
    module_name = "runtime_test_iterative_limit"
    module = ModuleType(module_name)
    setattr(module, "definition", lambda: SubroutineDefinition(graph=graph))  # noqa: B010
    monkeypatch.setitem(sys.modules, module_name, module)
    definition: WorkflowDefinition[int, int, None, object] = WorkflowDefinition(
        id=GraphId("iterative_limit_workflow"),
        input_type=int,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module_name,
            params_types={(".", graph.id): type(None)},
            profile_arguments={},
            session_arguments={},
        ),
        configuration=WorkflowConfiguration(),
    )

    # The acceptance property is the iterative control driver, not filesystem
    # throughput. All transition, routing, state, budget, and event code remains
    # real.
    monkeypatch.setattr(
        "verdog_runtime.interpreter.execution._GraphOutput.node",
        _cheap_node_output,
    )

    with pytest.raises(
        RuntimeError,
        match="workflow transition limit exceeded",
    ) as caught:
        Dispatcher(transition_limit=_TRANSITION_LIMIT).run(
            definition,
            1,
            output_dir=tmp_path / "run",
        )

    assert type(caught.value) is RuntimeError
    assert visits == _TRANSITION_LIMIT
    assert "Verdog edge: loop_again" in getattr(caught.value, "__notes__", ())


def test_nested_local_calls_exceed_python_recursion_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, graph_ids = _deep_local_call_workflow(
        monkeypatch,
        _DEEP_CALL_DEPTH,
    )
    output = tmp_path / "d"

    previous_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(_TEST_RECURSION_LIMIT)
        assert sys.getrecursionlimit() < _DEEP_CALL_DEPTH
        result = Dispatcher(
            project_root=tmp_path,
            transition_limit=2 * _DEEP_CALL_DEPTH + 1,
        ).run(
            workflow,
            0,
            output_dir=output,
        )
    finally:
        sys.setrecursionlimit(previous_limit)

    assert result.output == _DEEP_CALL_DEPTH
    assert (output / "enter/000001").is_dir()

    graph_directory = output
    for _graph_id in graph_ids[1:]:
        graph_directory /= "call/000001"
        assert graph_directory.is_dir()
    assert (
        len(graph_directory.relative_to(output).parts) == 2 * _DEEP_CALL_DEPTH
    )
    assert not (output / "activations").exists()


def test_nested_node_visits_preserve_artifacts_and_report_call_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = 2
    workflow, graph_ids = _deep_local_call_workflow(
        monkeypatch,
        depth,
        write_artifacts=True,
    )
    output = tmp_path / "nested-calls"

    result = Dispatcher(
        project_root=tmp_path,
        transition_limit=2 * depth + 1,
    ).run(
        workflow,
        0,
        output_dir=output,
    )

    assert result.output == depth
    config_chain = _report_chain(output / "config.md")
    stats_chain = _report_chain(output / "stats.md")
    assert len(config_chain) == depth
    assert [path.parent for path in stats_chain] == [
        path.parent for path in config_chain
    ]

    child_dirs = [path.parent for path in config_chain]
    assert len(set(child_dirs)) == depth
    assert call_reports(config_chain[-1]) == []
    assert call_reports(stats_chain[-1]) == []

    report_dirs = [output, *child_dirs]
    for index, child in enumerate(child_dirs):
        assert child == report_dirs[index] / "call/000001"
    for graph_id, directory in zip(graph_ids, report_dirs, strict=True):
        assert (directory / "enter/000001").is_dir()
        assert {
            row[1] for row in table_rows(directory / "stats.md", "Nodes")
        } == {str(graph_id)}

    expected_artifacts = tuple(
        report_dirs[index] / "call" / "000001" / "adapter.txt"
        for index in range(depth)
    )
    assert set(output.rglob("adapter.txt")) == set(expected_artifacts)
    assert [
        artifact.read_text(encoding="utf-8") for artifact in expected_artifacts
    ] == [
        f"{graph_ids[0]}: 0->{depth}",
        f"{graph_ids[1]}: 1->{depth}",
    ]
    assert [
        len(artifact.relative_to(output).parts)
        for artifact in expected_artifacts
    ] == [3, 5]


def test_root_node_named_trace_does_not_collide_with_the_run_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, _ = _deep_local_call_workflow(monkeypatch, 1, call_id="trace")
    output = tmp_path / "trace-node"

    result = Dispatcher(project_root=tmp_path).run(
        workflow, 0, output_dir=output
    )

    assert result.output == 1
    assert {path.name for path in output.iterdir() if path.is_dir()} == {
        "enter",
        "trace",
        "exit",
    }
    assert (output / "trace/000001/enter/000001").is_dir()
    assert "START trace/000001" in (output / "trace.log").read_text("utf-8")
    assert call_reports(output / "config.md") == [
        output / "trace/000001/config.md"
    ]
