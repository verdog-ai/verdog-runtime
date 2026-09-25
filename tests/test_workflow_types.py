from types import UnionType
from typing import Final, Literal, TypeAlias

from typing_extensions import TypeForm

from verdog_runtime.declarations import (
    GraphDefinition,
    PortDefinition,
    SubroutineCall,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import GraphId, NodeId


Input: TypeAlias = int | str
Mode: TypeAlias = Literal["fast", "careful"]


def _workflow(
    input_type: TypeForm[object] | UnionType,
) -> WorkflowDefinition[object, object, None, object]:
    graph = GraphDefinition[object, object, None, object](
        id=GraphId("body"),
        params_type=type(None),
        enter=PortDefinition(id=NodeId("enter")),
        exit=PortDefinition(id=NodeId("exit")),
        failure=PortDefinition(id=NodeId("failure")),
        nodes=(),
        edges=(),
    )
    return WorkflowDefinition(
        id=GraphId("main"),
        input_type=input_type,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module="sample.subroutines.body",
            params_types={(".", graph.id): graph.params_type},
            profile_arguments={},
            session_arguments={},
        ),
        configuration=WorkflowConfiguration(),
    )


def test_workflows_retain_their_cli_input_type() -> None:
    workflow: Final[WorkflowDefinition[object, object, None, object]] = _workflow(Input)

    assert workflow.input_type == Input


def test_literal_inputs_remain_static_metadata() -> None:
    workflow = _workflow(Mode)

    assert workflow.input_type == Mode
