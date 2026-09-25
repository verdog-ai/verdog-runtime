"""Runtime-selected agent implementations for declared workflow profiles."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from verdog_runtime.declarations import agents as agent_declarations
from verdog_runtime.declarations import ids


def _profile_arguments() -> Mapping[
    ids.AgentProfileId, agent_declarations.AgentInvoker
]:
    return {}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowConfiguration:
    """Concrete providers bound to workflow profile parameters."""

    profile_arguments: Mapping[
        ids.AgentProfileId, agent_declarations.AgentInvoker
    ] = dataclasses.field(default_factory=_profile_arguments)
