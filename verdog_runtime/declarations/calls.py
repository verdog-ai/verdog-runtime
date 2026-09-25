from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from .context import CallContext

# `graph` imports this module while `results -> state` imports `graph` again.
# Postponed annotations keep the public protocol typed without completing that cycle.
if TYPE_CHECKING:
    from .results import Success


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
    def __call__(
        self,
        input: InputContraT,
        state: StateT,
        context: CallContext[ParamsT, ChildParamsT, ChildInputT, ChildOutputT],
        /,
    ) -> Success[OutputCoT, StateT]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
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
    implementation: CallImplementation[
        InputT,
        OutputT,
        StateT,
        ParamsT,
        ChildInputT,
        ChildOutputT,
        ChildParamsT,
    ]
