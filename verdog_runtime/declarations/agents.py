"""Agent requests, results, and provider session capabilities."""

from __future__ import annotations

import dataclasses
import enum
import pathlib
from typing import Protocol

from verdog_runtime import cancellation as cancellation_module
from verdog_runtime.declarations import context as contexts
from verdog_runtime.declarations import ids


class AgentSessionAction(enum.StrEnum):
    """Whether a provider resumes a session or branches from latest reply."""

    CONTINUE = "continue"
    FORK = "fork"


@dataclasses.dataclass(frozen=True, slots=True)
class AgentSessionCapabilities:
    """Session operations supported by an agent provider."""

    fork_latest: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentRequest:
    """One invocation with workspace, access mode, session, and cancellation."""

    prompt: str
    profile_id: ids.AgentProfileId
    session_id: ids.AgentSessionId
    persistent: bool
    provider_session_id: ids.ProviderSessionId | None
    node_context: contexts.NodeContext[object]
    workspace: pathlib.Path
    access: contexts.AgentAccess
    artifact_dir: pathlib.Path
    provider_session_action: AgentSessionAction = AgentSessionAction.CONTINUE
    cancellation: cancellation_module.CancellationToken = dataclasses.field(
        default_factory=cancellation_module.CancellationToken
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentReply:
    """Provider response text and the session identifier available for reuse."""

    text: str
    provider_session_id: ids.ProviderSessionId | None = None


class AgentInvocationError(RuntimeError):
    """An agent provider could not be started or returned no usable reply."""


class AgentInvoker(Protocol):
    """A synchronous provider returning a reply or an invocation error."""

    @property
    def session_provider(self) -> str:
        """The stable provider name used to identify stored sessions."""
        ...

    def __call__(self, request: AgentRequest, /) -> AgentReply:
        """Run the request and return response text with an session ID."""
        ...


class ForkingAgentInvoker(AgentInvoker, Protocol):
    """An agent provider that declares its session branching capabilities."""

    @property
    def session_capabilities(self) -> AgentSessionCapabilities:
        """The supported session operations for this provider."""
        ...
