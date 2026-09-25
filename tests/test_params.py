from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import verdog_runtime.interpreter.execution as execution_module
from report_helpers import call_reports, table_rows
from verdog_runtime.declarations import (
    CallContext,
    CallVisitDefinition,
    EdgeDefinition,
    Feature,
    FeatureDefinition,
    FeatureKind,
    FeatureNodeDefinition,
    FeatureState,
    FeatureSuccess,
    GraphDefinition,
    NodeContext,
    NodeDefinition,
    NumericalEffectObservation,
    NumericalFeatureEffect,
    PortDefinition,
    Python,
    SubroutineCall,
    SubroutineDefinition,
    Success,
    VisitDefinition,
    WorkflowCall,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import EdgeId, FeatureId, GraphId, NodeId
from verdog_runtime.interpreter import Dispatcher
from verdog_runtime.interpreter.execution import (
    _ParameterRegistry,  # pyright: ignore[reportPrivateUsage]
)


def _workflow(
    graph: GraphDefinition[int, int, Any, object],
    /,
) -> WorkflowDefinition[int, int, Any, object]:
    subroutine = SubroutineDefinition(graph=graph)
    module_name = "runtime_test_params_entry"
    declaration = ModuleType(module_name)
    setattr(declaration, "definition", lambda: subroutine)  # noqa: B010
    sys.modules[module_name] = declaration
    params_types = {(".", graph.id): graph.params_type}
    for node in graph.nodes:
        if isinstance(node.operation, SubroutineCall):
            params_types.update(node.operation.params_types)
    return WorkflowDefinition(
        id=GraphId(f"{graph.id}.workflow"),
        input_type=int,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module_name,
            params_types=params_types,
            profile_arguments={
                parameter.id: parameter.id for parameter in graph.profile_parameters
            },
            session_arguments={
                parameter.id: parameter.id for parameter in graph.session_parameters
            },
        ),
        configuration=WorkflowConfiguration(),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ParentParams:
    seed: int = 7
    input: int = 11
    runtime: int = 13


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildParams:
    offset: int = 3


@dataclass(frozen=True, slots=True)
class EmptyState:
    pass


class CustomPython(Python):
    pass


class CustomSubroutineCall(SubroutineCall):
    pass


class CustomWorkflowCall(WorkflowCall):
    pass


@pytest.mark.parametrize(
    ("python_type", "call_type"),
    [(Python, SubroutineCall), (CustomPython, CustomSubroutineCall)],
)
def test_subroutine_calls_share_defaults_and_override_one_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_type: type[Python],
    call_type: type[SubroutineCall],
) -> None:
    @dataclass(frozen=True, slots=True)
    class RuntimeOptions:
        backend: str = "test"

    child_contexts: list[ChildParams | None] = []

    def child_run(
        input: int,
        state: EmptyState,
        context: NodeContext[ChildParams | None],
        /,
    ) -> Success[int, EmptyState]:
        child_contexts.append(context.params)
        offset = 0 if context.params is None else context.params.offset
        return Success(output=input + offset, state=state)

    child_enter = PortDefinition(id=NodeId("child_enter"))
    child_exit = PortDefinition(id=NodeId("child_exit"))
    child_failure = PortDefinition(id=NodeId("child_failure"))
    child_node = NodeDefinition[EmptyState, object](
        id=NodeId("child_work"),
        name="child work",
        state_type=EmptyState,
        operation=python_type(),
    )
    child_graph: GraphDefinition[int, int, ChildParams, object] = GraphDefinition(
        id=GraphId("sample.parent__child"),
        params_type=ChildParams,
        enter=child_enter,
        exit=child_exit,
        failure=child_failure,
        nodes=(child_node,),
        edges=(
            EdgeDefinition(
                id=EdgeId("child_in"),
                source=child_enter.id,
                target=child_node.id,
                visit=VisitDefinition(implementation=child_run),
            ),
            EdgeDefinition(
                id=EdgeId("child_out"),
                source=child_node.id,
                target=child_exit.id,
            ),
        ),
    )
    child_definition = SubroutineDefinition(graph=child_graph)

    child_module = "sample.subroutines.parent.subroutines.child"
    declaration = ModuleType(child_module)
    setattr(declaration, "definition", lambda: child_definition)  # noqa: B010
    monkeypatch.setitem(sys.modules, child_module, declaration)

    parent_contexts: list[ParentParams] = []
    child_params: list[ChildParams] = []
    override = ChildParams(offset=10)

    def call_default(
        input: int,
        state: EmptyState,
        context: CallContext[ParentParams, ChildParams, int, int],
        /,
    ) -> Success[int, EmptyState]:
        parent_contexts.append(context.params)
        child_params.append(context.child_params)
        return Success(
            output=context.invoke(input),
            state=state,
        )

    def call_override(
        input: int,
        state: EmptyState,
        context: CallContext[ParentParams, ChildParams, int, int],
        /,
    ) -> Success[int, EmptyState]:
        parent_contexts.append(context.params)
        return Success(
            output=context.invoke(input, params=override),
            state=state,
        )

    def call_none(
        input: int,
        state: EmptyState,
        context: CallContext[ParentParams, ChildParams | None, int, int],
        /,
    ) -> Success[int, EmptyState]:
        parent_contexts.append(context.params)
        return Success(
            output=context.invoke(input, params=None),
            state=state,
        )

    initialized: list[ParentParams] = []

    def initialize(
        input: int, state: FeatureState[object], context: NodeContext[ParentParams], /
    ) -> FeatureSuccess[object]:
        initialized.append(context.params)
        return FeatureSuccess(state=state.replace(seed, context.params.seed))

    seed = FeatureDefinition[int, object](
        id=FeatureId("seed"),
        label="seed",
        description="root parameter probe",
        kind=FeatureKind.INTEGER,
    )
    parent_enter = PortDefinition(id=NodeId("enter"))
    parent_exit = PortDefinition(id=NodeId("exit"))
    parent_failure = PortDefinition(id=NodeId("failure"))
    initialize_node = FeatureNodeDefinition(
        id=NodeId("initialize"), name="Initialize seed", operation=Feature()
    )
    calls = tuple(
        NodeDefinition[EmptyState, object](
            id=NodeId(f"call_{index}"),
            name=f"call {index}",
            state_type=EmptyState,
            operation=call_type(
                definition_id=child_graph.id,
                definition_module=child_module,
                params_types={(".", child_graph.id): ChildParams},
                profile_arguments={},
                session_arguments={},
            ),
        )
        for index in range(4)
    )
    parent_graph: GraphDefinition[int, int, ParentParams, object] = GraphDefinition(
        id=GraphId("sample.parent"),
        params_type=ParentParams,
        enter=parent_enter,
        exit=parent_exit,
        failure=parent_failure,
        nodes=(initialize_node, *calls),
        edges=(
            EdgeDefinition(
                id=EdgeId("initialize"),
                source=parent_enter.id,
                target=initialize_node.id,
                visit=VisitDefinition(implementation=initialize),
            ),
            EdgeDefinition(
                id=EdgeId("in"),
                source=initialize_node.id,
                target=calls[0].id,
                effects=(
                    NumericalFeatureEffect(
                        feature_id=seed.id,
                        observation=NumericalEffectObservation.UNCONSTRAINED,
                    ),
                ),
                visit=CallVisitDefinition(implementation=call_default),
            ),
            EdgeDefinition(
                id=EdgeId("override"),
                source=calls[0].id,
                target=calls[1].id,
                visit=CallVisitDefinition(implementation=call_override),
            ),
            EdgeDefinition(
                id=EdgeId("explicit_none"),
                source=calls[1].id,
                target=calls[2].id,
                visit=CallVisitDefinition(implementation=call_none),
            ),
            EdgeDefinition(
                id=EdgeId("default_again"),
                source=calls[2].id,
                target=calls[3].id,
                visit=CallVisitDefinition(implementation=call_default),
            ),
            EdgeDefinition(id=EdgeId("out"), source=calls[3].id, target=parent_exit.id),
        ),
        features=(seed,),
    )

    result = Dispatcher(project_root=tmp_path).run(
        _workflow(parent_graph),
        1,
        output_dir=tmp_path / "output",
        runtime_options=RuntimeOptions(),
    )

    assert result.output == 17
    assert result.state.get(seed) == ParentParams().seed
    assert len({id(params) for params in parent_contexts + initialized}) == 1
    assert child_params == [ChildParams()] * 4
    assert all(params is child_params[0] for params in child_params)
    assert child_contexts == [ChildParams(), override, None, ChildParams()]
    assert child_contexts[0] is child_contexts[3]
    output = tmp_path / "output"
    assert table_rows(output / "config.md") == [
        ["input", "1"],
        ["runtime.backend", "test"],
        ["params.seed", "7"],
        ["params.input", "11"],
        ["params.runtime", "13"],
    ]
    configuration_reports = call_reports(output / "config.md")
    assert len(configuration_reports) == len(calls)
    for call, child_report, expected in zip(
        calls,
        configuration_reports,
        (
            ["params.offset", "3"],
            ["params.offset", "10"],
            ["params", "None"],
            ["params.offset", "3"],
        ),
        strict=True,
    ):
        child_output = child_report.parent
        assert child_output == (
            output / str(call.id) / "000001"
        ).resolve()
        assert table_rows(child_report) == [expected]
        child_rows = table_rows(child_output / "stats.md", "Nodes")
        assert {row[1] for row in child_rows} == {str(child_graph.id)}
        assert any(row[2:5] == ["child_work", "python", "1"] for row in child_rows)
        assert call_reports(child_output / "config.md") == []
        assert call_reports(child_output / "stats.md") == []
    for name in ("config.md", "stats.md"):
        children = [report.with_name(name) for report in configuration_reports]
        assert call_reports(output / name) == children
        assert set(output.rglob(name)) == {output / name, *children}
    assert {path.name for path in output.iterdir() if path.is_file()} == {
        "config.md",
        "stats.md",
        "trace.log",
    }
    assert not list(output.rglob("configuration.md"))
    rows = table_rows(output / "stats.md", "Nodes")
    assert {row[1] for row in rows} == {str(parent_graph.id)}
    assert not any(
        row[0] == "python" for row in table_rows(output / "stats.md", "Summary")
    )
    assert all(
        any(
            row[1:5] == [str(parent_graph.id), str(call.id), "subroutine_call", "1"]
            for row in rows
        )
        for call in calls
    )


@pytest.mark.parametrize("call_type", [WorkflowCall, CustomWorkflowCall])
def test_workflow_call_delegates_parameter_defaults_to_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call_type: type[WorkflowCall]
) -> None:
    child_workflow_id = GraphId("child.workflow")
    child_path = "external/child"
    captured: dict[str, object] = {}

    def fake_child(**kwargs: object) -> tuple[object, int]:
        captured.update(kwargs)
        return cast(int, kwargs["input"]) + ChildParams().offset, cast(
            int, kwargs["transitions_remaining"]
        )

    monkeypatch.setattr(execution_module, "invoke_child_process", fake_child)

    def adapt(
        input: int,
        state: EmptyState,
        context: CallContext[None, ChildParams, int, int],
        /,
    ) -> Success[int, EmptyState]:
        return Success(
            output=context.invoke(input),
            state=state,
        )

    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    call = NodeDefinition[EmptyState, object](
        id=NodeId("call"),
        name="call",
        state_type=EmptyState,
        operation=call_type(
            definition_id=child_workflow_id,
            definition_module="child.workflows.workflow",
            project_path=child_path,
        ),
    )
    graph: GraphDefinition[int, int, None, object] = GraphDefinition(
        id=GraphId("parent.body"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(call,),
        edges=(
            EdgeDefinition(
                id=EdgeId("in"),
                source=enter.id,
                target=call.id,
                visit=CallVisitDefinition(implementation=adapt),
            ),
            EdgeDefinition(id=EdgeId("out"), source=call.id, target=exit_.id),
        ),
    )

    result = Dispatcher(project_root=tmp_path).run(
        _workflow(graph), 5, output_dir=tmp_path / "workflow-output"
    )

    assert result.output == 8
    assert captured["definition_id"] == child_workflow_id
    assert "params_override" not in captured
    rows = table_rows(tmp_path / "workflow-output/stats.md", "Nodes")
    assert any(
        row[1:5] == [str(graph.id), "call", "workflow_call", "1"] for row in rows
    )
    assert (
        not {
            "input_type",
            "output_type",
            "params",
            "params_types",
            "params_root",
            "params_type",
        }
        & captured.keys()
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RequiredParams:
    value: int


def test_missing_required_parameter_value_names_its_address() -> None:
    address = (".", GraphId("required"))
    with pytest.raises(TypeError) as caught:
        _ParameterRegistry.create(
            {address: RequiredParams},
            {},
        )
    assert any("parameter value is missing" in note for note in caught.value.__notes__)
