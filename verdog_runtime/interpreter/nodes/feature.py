from __future__ import annotations

from typing import TypeVar

from ...declarations import FeatureState, NodeContext, WorkflowState
from . import Visit

InputT = TypeVar("InputT")
ScopeT = TypeVar("ScopeT")
ParamsT = TypeVar("ParamsT")
ResultT = TypeVar("ResultT")


def execute(
    implementation: Visit[InputT, FeatureState[ScopeT], NodeContext[ParamsT], ResultT],
    value: InputT,
    state: WorkflowState[ScopeT],
    context: NodeContext[ParamsT],
    /,
) -> ResultT:
    feature_state = FeatureState[ScopeT]._from_workflow_state(  # pyright: ignore[reportPrivateUsage]
        state
    )
    return implementation(value, feature_state, context)
