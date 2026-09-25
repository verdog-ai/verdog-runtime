from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from ..cancellation import CancellationToken
from .context import AgentAccess, NodeContext
from .ids import (
    AgentProfileId,
    AgentSessionId,
    ProviderSessionId,
)


class AgentSessionAction(StrEnum):
    CONTINUE = "continue"
    FORK = "fork"


@dataclass(frozen=True, slots=True)
class AgentSessionCapabilities:
    fork_latest: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRequest:
    prompt: str
    profile_id: AgentProfileId
    session_id: AgentSessionId
    persistent: bool
    provider_session_id: ProviderSessionId | None
    node_context: NodeContext[object]
    workspace: Path
    access: AgentAccess
    artifact_dir: Path
    provider_session_action: AgentSessionAction = AgentSessionAction.CONTINUE
    cancellation: CancellationToken = field(default_factory=CancellationToken)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentReply:
    text: str
    provider_session_id: ProviderSessionId | None = None


class AgentInvocationError(RuntimeError):
    """An agent provider could not be started or returned no usable reply."""


class AgentInvoker(Protocol):
    @property
    def session_provider(self) -> str: ...

    def __call__(self, request: AgentRequest, /) -> AgentReply: ...


class ForkingAgentInvoker(AgentInvoker, Protocol):
    @property
    def session_capabilities(self) -> AgentSessionCapabilities: ...
