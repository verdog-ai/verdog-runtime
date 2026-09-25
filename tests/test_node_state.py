from __future__ import annotations

import sys
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, assert_type, cast

import pytest

from verdog_runtime.declarations import (
    EdgeDefinition,
    GraphDefinition,
    NodeDefinition,
    PortDefinition,
    Python,
    StateKey,
    SubroutineCall,
    SubroutineDefinition,
    Success,
    VisitDefinition,
    WorkflowConfiguration,
    WorkflowDefinition,
    WorkflowState,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId
from verdog_runtime.interpreter import Dispatcher, validate_graph
from verdog_runtime.interpreter.validation import require_immutable_state


def _graph(
    state_type: object,
    implementation: object,
    /,
) -> tuple[
    GraphDefinition[int, int, None, object], NodeDefinition[Any, object]
]:
    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    node: NodeDefinition[Any, object] = NodeDefinition(
        id=NodeId("work"),
        name="work",
        operation=Python(),
        state_type=cast(Any, state_type),
    )
    graph: GraphDefinition[int, int, None, object] = GraphDefinition(
        id=GraphId("state"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(node,),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter_work"),
                source=enter.id,
                target=node.id,
                visit=VisitDefinition(
                    implementation=cast(Any, implementation),
                ),
            ),
            EdgeDefinition(
                id=EdgeId("work_exit"),
                source=node.id,
                target=exit_.id,
            ),
        ),
    )
    return graph, node


def _definition(
    graph: GraphDefinition[int, int, None, object], /
) -> WorkflowDefinition[int, int, None, object]:
    subroutine = SubroutineDefinition(graph=graph)
    module_name = "runtime_test_node_state_entry"
    module = ModuleType(module_name)
    module.__dict__["definition"] = lambda: subroutine
    sys.modules[module_name] = module
    return WorkflowDefinition(
        id=GraphId("state_workflow"),
        input_type=int,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module_name,
            params_types={(".", graph.id): graph.params_type},
            profile_arguments={},
            session_arguments={},
        ),
        configuration=WorkflowConfiguration(),
    )


def _preserve(
    input: int,
    state: object,
    _context: object,
    /,
) -> Success[int, object]:
    return Success(output=input, state=state)


def test_each_run_constructs_one_fresh_node_state(tmp_path: Path) -> None:
    constructed: list[FreshState] = []

    @dataclass(frozen=True, slots=True)
    class FreshState:
        count: int = 0

        def __post_init__(self) -> None:
            constructed.append(self)

    graph, _ = _graph(FreshState, _preserve)
    definition = _definition(graph)
    dispatcher = Dispatcher()

    (dispatcher.run(definition, 1, output_dir=tmp_path / "first"))
    assert len(constructed) == 1
    (dispatcher.run(definition, 1, output_dir=tmp_path / "second"))

    assert len(constructed) == 2
    assert constructed[0] is not constructed[1]


def test_state_constructor_failure_uses_the_graph_failure_boundary(
    tmp_path: Path,
) -> None:
    calls: list[None] = []

    @dataclass(frozen=True, slots=True)
    class BrokenState:
        count: int = 0

        def __post_init__(self) -> None:
            calls.append(None)
            raise RuntimeError("state constructor exploded")

    graph, _ = _graph(BrokenState, _preserve)
    output = tmp_path / "output"

    with pytest.raises(
        RuntimeError, match="state constructor exploded"
    ) as caught:
        (Dispatcher().run(_definition(graph), 1, output_dir=output))

    assert calls == [None]
    assert any(
        "state initialization: node=work" in note
        for note in caught.value.__notes__
    )
    stacktrace = (output / "failure" / "000001" / "stacktrace.txt").read_text(
        "utf-8"
    )
    assert "in __post_init__" in stacktrace
    assert "RuntimeError: state constructor exploded" in stacktrace
    assert "Verdog failure boundary:" in stacktrace


@dataclass(frozen=True, slots=True)
class ValidState:
    value: int = 0


class FirstScope:
    pass


class SecondScope:
    pass


def scopes_are_statically_distinct(
    state: WorkflowState[FirstScope],
    foreign_key: StateKey[ValidState, SecondScope],
) -> None:
    state.get(foreign_key)  # pyright: ignore[reportArgumentType]


@dataclass(slots=True)
class MutableState:
    value: int = 0


@dataclass(frozen=True)
class DictBackedState:
    value: int = 0


@dataclass(frozen=True, slots=True)
class RequiredState:
    value: int


@dataclass(frozen=True, slots=True)
class FactoryState:
    value: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BaseState:
    value: int = 0


@dataclass(frozen=True, slots=True)
class InheritedState(BaseState):
    other: int = 0


@dataclass(frozen=True, slots=True)
class InitVariableState:
    value: InitVar[int] = 0


@dataclass(frozen=True, slots=True)
class ClassVariableState:
    value: ClassVar[int] = 0


@dataclass(frozen=True, slots=True)
class CustomizedFieldState:
    value: int = field(default=0, repr=False)


@pytest.mark.parametrize(
    ("state_type", "message"),
    (
        (int, "dataclass record"),
        (MutableState, "must be frozen"),
        (DictBackedState, "must use slots"),
        (RequiredState, "must have a default"),
        (FactoryState, "must use a direct default"),
        (InheritedState, "must not inherit fields"),
        (InitVariableState, "InitVar"),
        (ClassVariableState, "ClassVar"),
        (CustomizedFieldState, "must not customize field"),
    ),
)
def test_node_state_type_is_one_uniform_record(
    state_type: object,
    message: str,
) -> None:
    graph, _ = _graph(state_type, _preserve)
    with pytest.raises(TypeError, match=message):
        validate_graph(graph)


def test_uniform_record_is_accepted() -> None:
    graph, _ = _graph(ValidState, _preserve)
    validate_graph(graph)


def test_paths_are_immutable_state_values() -> None:
    require_immutable_state(Path("workspace"))


def test_state_changes_reject_another_invocation_of_the_same_graph() -> None:
    node = NodeDefinition[ValidState, object](
        id=NodeId("typed"),
        name="typed",
        operation=Python(),
        state_type=ValidState,
    )
    first = WorkflowState[object]._initial(  # pyright: ignore[reportPrivateUsage]
        ((node, ValidState()),)
    )
    second = WorkflowState[object]._initial(  # pyright: ignore[reportPrivateUsage]
        ((node, ValidState()),)
    )

    with pytest.raises(ValueError, match="workflow state universe changed"):
        first._changes(second)  # pyright: ignore[reportPrivateUsage]

    assert_type(first.get(node), ValidState)
