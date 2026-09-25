"""Execute visits that may update declared feature state."""

from __future__ import annotations

from typing import TypeVar

from verdog_runtime import declarations
from verdog_runtime.interpreter import nodes

InputT = TypeVar("InputT")
ScopeT = TypeVar("ScopeT")
ParamsT = TypeVar("ParamsT")
ResultT = TypeVar("ResultT")


def execute(
    implementation: nodes.Visit[
        InputT,
        declarations.FeatureState[ScopeT],
        declarations.NodeContext[ParamsT],
        ResultT,
    ],
    value: InputT,
    state: declarations.WorkflowState[ScopeT],
    context: declarations.NodeContext[ParamsT],
    /,
) -> ResultT:
    """Invoke a feature visit with its input and feature-state view."""
    feature_state = declarations.FeatureState[ScopeT]._from_workflow_state(  # pyright: ignore[reportPrivateUsage]
        state
    )
    return implementation(value, feature_state, context)
