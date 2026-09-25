"""Protocols implemented by workflow resource initializers."""

from __future__ import annotations

from typing import Protocol, TypeVar

from verdog_runtime.declarations import agents as agent_declarations

InputT = TypeVar("InputT", contravariant=True)
ParamsT = TypeVar("ParamsT", contravariant=True)


class AgentProfileInitializer(Protocol[InputT, ParamsT]):
    """A function binding graph input and parameters to an agent provider."""

    def __call__(
        self, input: InputT, params: ParamsT, /
    ) -> agent_declarations.AgentInvoker:
        """Bind input and parameters to an agent invocation implementation."""
        ...
