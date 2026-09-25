"""Execute ordinary Python visits with their typed context."""

from __future__ import annotations

from typing import TypeVar

from verdog_runtime import declarations
from verdog_runtime.interpreter import nodes

InputT = TypeVar("InputT")
StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")
ResultT = TypeVar("ResultT")


def execute(
    implementation: nodes.Visit[
        InputT, StateT, declarations.NodeContext[ParamsT], ResultT
    ],
    value: InputT,
    state: StateT,
    context: declarations.NodeContext[ParamsT],
    /,
) -> ResultT:
    """Invoke a Python visit with input, immutable state, and node context."""
    return implementation(value, state, context)
