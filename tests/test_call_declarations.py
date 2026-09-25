from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from verdog_runtime.declarations import (
    Agent,
    AgentProfileId,
    AgentSessionId,
    CallContext,
    CallVisitDefinition,
    EdgeDefinition,
    Feature,
    FeatureNodeDefinition,
    GraphDefinition,
    NodeDefinition,
    PortDefinition,
    Python,
    SubroutineCall,
    Success,
    VisitDefinition,
    WorkflowCall,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId, RunId
from verdog_runtime.interpreter import validate_graph


@dataclass(frozen=True, slots=True)
class State:
    calls: int = 0


def _call(
    input: int,
    state: State,
    context: CallContext[None, str, int, int],
    /,
) -> Success[str, State]:
    child_output = context.invoke(input + 1)
    return Success(
        output=f"{context.child_params}:{child_output}",
        state=State(calls=state.calls + 1),
    )


def test_call_context_invokes_child_synchronously(tmp_path: Path) -> None:
    def invoke(input: int, params: str) -> int:
        return input + len(params)

    context: CallContext[None, str, int, int] = CallContext(
        run_id=RunId("run"),
        graph_id=GraphId("graph"),
        node_id=NodeId("call"),
        edge_id=EdgeId("enter_call"),
        output_dir=tmp_path,
        params=None,
        child_params="configured",
        _invoke=invoke,
    )

    assert context.child_params == "configured"
    assert context.invoke(2) == 12
    assert context.invoke(2, params="x") == 3
    with pytest.raises(FrozenInstanceError):
        context.child_params = "changed"  # type: ignore[misc]


def test_call_visit_definition_is_distinct_from_legacy_visit() -> None:
    durable: CallVisitDefinition[
        int,
        str,
        State,
        None,
        int,
        int,
        str,
    ] = CallVisitDefinition(implementation=_call)
    edge = EdgeDefinition(
        id=EdgeId("enter_call"),
        source=NodeId("enter"),
        target=NodeId("call"),
        visit=durable,
    )

    assert edge.visit is durable
    assert durable.implementation is _call
    assert VisitDefinition(implementation=None).implementation is None


def _graph(
    node: NodeDefinition[State, object] | FeatureNodeDefinition,
    visit: object,
    /,
) -> GraphDefinition[Any, Any, None, object]:
    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    return GraphDefinition(
        id=GraphId("durable_call_validation"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(node,),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter_node"),
                source=enter.id,
                target=node.id,
                visit=cast(Any, visit),
            ),
            EdgeDefinition(
                id=EdgeId("node_exit"),
                source=node.id,
                target=exit_.id,
            ),
        ),
    )


def _node(operation: object) -> NodeDefinition[State, object]:
    return NodeDefinition(
        id=NodeId("node"),
        name="Node",
        operation=cast(Any, operation),
        state_type=State,
    )


DURABLE_VISIT: CallVisitDefinition[
    int,
    str,
    State,
    None,
    int,
    int,
    str,
] = CallVisitDefinition(implementation=_call)


@pytest.mark.parametrize(
    "operation",
    (
        SubroutineCall(
            definition_id=GraphId("child"),
            definition_module="sample.child",
            params_types={(".", GraphId("child")): type(None)},
            profile_arguments={},
            session_arguments={},
        ),
        WorkflowCall(
            definition_id=GraphId("child_workflow"),
            definition_module="sample.child_workflow",
        ),
    ),
)
def test_durable_visit_is_valid_only_for_call_operations(
    operation: object,
) -> None:
    validate_graph(_graph(_node(operation), DURABLE_VISIT))


@pytest.mark.parametrize(
    "node",
    (
        _node(Python()),
        _node(
            Agent(
                profile=AgentProfileId("profile"),
                session=AgentSessionId("session"),
            )
        ),
        FeatureNodeDefinition(
            id=NodeId("node"),
            name="Feature",
            operation=Feature(),
        ),
    ),
)
def test_durable_visit_is_rejected_for_non_call_operations(
    node: NodeDefinition[State, object] | FeatureNodeDefinition,
) -> None:
    with pytest.raises(
        ValueError,
        match="may only target a subroutine or workflow call node",
    ):
        validate_graph(_graph(node, DURABLE_VISIT))


def test_durable_visit_requires_callable_implementation() -> None:
    invalid = replace(DURABLE_VISIT, implementation=cast(Any, None))

    with pytest.raises(ValueError, match="implementation must be callable"):
        validate_graph(
            _graph(
                _node(
                    WorkflowCall(
                        definition_id=GraphId("child"),
                        definition_module="sample.child",
                    )
                ),
                invalid,
            )
        )


def _legacy(
    input: int,
    state: State,
    _context: object,
    /,
) -> Success[int, State]:
    return Success(output=input, state=state)


def test_plain_visit_is_rejected_for_call_operations() -> None:
    with pytest.raises(
        ValueError, match="call node must declare CallVisitDefinition"
    ):
        validate_graph(
            _graph(
                _node(
                    WorkflowCall(
                        definition_id=GraphId("child"),
                        definition_module="sample.child",
                    )
                ),
                VisitDefinition(implementation=_legacy),
            )
        )
