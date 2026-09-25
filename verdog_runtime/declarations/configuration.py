from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .agents import AgentInvoker
from .ids import AgentProfileId


def _profile_arguments() -> Mapping[AgentProfileId, AgentInvoker]:
    return {}


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowConfiguration:
    profile_arguments: Mapping[AgentProfileId, AgentInvoker] = field(
        default_factory=_profile_arguments
    )
