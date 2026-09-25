"""Typed visit contracts for invoking child subroutines and workflows."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from verdog_runtime.declarations import context as contexts

# `graph` imports this module while `results -> state` imports `graph` again.
# Postponed annotations keep the public protocol typed without completing that
# cycle.
if TYPE_CHECKING:
    from verdog_runtime.declarations import results


InputContraT = TypeVar("InputContraT", contravariant=True)
InputT = TypeVar("InputT")
OutputCoT = TypeVar("OutputCoT", covariant=True)
OutputT = TypeVar("OutputT")
StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")
ChildInputT = TypeVar("ChildInputT")
ChildOutputT = TypeVar("ChildOutputT")
ChildParamsT = TypeVar("ChildParamsT")


class CallImplementation(
    Protocol[
        InputContraT,
        OutputCoT,
        StateT,
        ParamsT,
        ChildInputT,
        ChildOutputT,
        ChildParamsT,
    ]
):
    """A visit that invokes a child and returns its own output and state."""

    def __call__(
        self,
        input: InputContraT,
        state: StateT,
        context: contexts.CallContext[
            ParamsT, ChildParamsT, ChildInputT, ChildOutputT
        ],
        /,
    ) -> results.Success[OutputCoT, StateT]:
        """Run the visit using the child invocation supplied by its context."""
        ...


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CallVisitDefinition(
    Generic[
        InputT,
        OutputT,
        StateT,
        ParamsT,
        ChildInputT,
        ChildOutputT,
        ChildParamsT,
    ]
):
    """The implementation and type contract of a child-call visit."""

    implementation: CallImplementation[
        InputT,
        OutputT,
        StateT,
        ParamsT,
        ChildInputT,
        ChildOutputT,
        ChildParamsT,
    ]
