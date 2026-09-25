from __future__ import annotations

from typing import Protocol, TypeVar

from .agents import AgentInvoker

InputT = TypeVar("InputT", contravariant=True)
ParamsT = TypeVar("ParamsT", contravariant=True)


class AgentProfileInitializer(Protocol[InputT, ParamsT]):
    def __call__(self, input: InputT, params: ParamsT, /) -> AgentInvoker: ...
