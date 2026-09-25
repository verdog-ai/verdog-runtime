from __future__ import annotations

from typing import TypeVar

from ...declarations import NodeContext
from . import Visit

InputT = TypeVar("InputT")
StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")
ResultT = TypeVar("ResultT")


def execute(
    implementation: Visit[InputT, StateT, NodeContext[ParamsT], ResultT],
    value: InputT,
    state: StateT,
    context: NodeContext[ParamsT],
    /,
) -> ResultT:
    return implementation(value, state, context)
