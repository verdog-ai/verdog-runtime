from dataclasses import dataclass
from pathlib import Path
from typing import assert_type

from verdog_runtime.declarations import (
    FeatureState,
    FeatureSuccess,
    NodeContext,
    Success,
    WorkflowState,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId, RunId
from verdog_runtime.interpreter.nodes import feature, python


@dataclass(frozen=True)
class Params:
    label: str


class Scope:
    pass


def test_handlers_preserve_typed_values_and_context(tmp_path: Path) -> None:
    params = Params(label="typed")
    context = NodeContext(
        run_id=RunId("run"),
        graph_id=GraphId("graph"),
        node_id=NodeId("node"),
        edge_id=EdgeId("edge"),
        output_dir=tmp_path,
        params=params,
    )
    expected = Success(output="done", state=3)

    def visit_python(
        value: int, state: int, received: NodeContext[Params], /
    ) -> Success[str, int]:
        assert value == 7
        assert state == 2
        assert received is context
        assert received.params is params
        return expected

    python_result = (python.execute(visit_python, 7, 2, context))
    assert_type(python_result, Success[str, int])
    assert python_result is expected

    workflow_state = WorkflowState[Scope]()
    feature_results: list[FeatureSuccess[Scope]] = []

    def visit_feature(
        value: str, state: FeatureState[Scope], received: NodeContext[Params], /
    ) -> FeatureSuccess[Scope]:
        assert value == "input"
        assert type(state) is FeatureState
        assert state._as_workflow_state() is workflow_state  # pyright: ignore[reportPrivateUsage]
        assert received is context
        assert received.params is params
        result = FeatureSuccess(state=state)
        feature_results.append(result)
        return result

    feature_result = (
        feature.execute(visit_feature, "input", workflow_state, context)
    )
    assert_type(feature_result, FeatureSuccess[Scope])
    assert feature_result is feature_results[0]
