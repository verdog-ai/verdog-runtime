"""Bind declared profiles and session resources to graph activations."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import Any, cast

from verdog_runtime import agents, declarations
from verdog_runtime.declarations import ids
from verdog_runtime.interpreter import _invocations


@dataclasses.dataclass(slots=True)
class SessionResource:
    persistent: bool
    provider: str | None = None
    provider_session_id: ids.ProviderSessionId | None = None
    access: declarations.AgentAccess | None = None
    copy_on_write: bool = False
    require_copy_on_write: bool = False
    branch_supported: bool | None = None
    tainted: bool = False

    def resume(
        self,
        session_id: ids.AgentSessionId,
        provider: str,
        access: declarations.AgentAccess,
        /,
    ) -> ids.ProviderSessionId | None:
        if self.provider is not None and self.provider != provider:
            raise agents.AgentInvocationError(
                f"agent session {session_id} belongs to provider "
                f"{self.provider}, not {provider}"
            )
        if self.provider_session_id is not None and self.access is not access:
            raise agents.AgentInvocationError(
                "a persistent agent session must keep one access mode"
            )
        return self.provider_session_id

    def request(
        self,
        session_id: ids.AgentSessionId,
        profile: agents.AgentInvoker,
        access: declarations.AgentAccess,
        /,
    ) -> tuple[ids.ProviderSessionId | None, agents.AgentSessionAction]:
        provider = profile.session_provider
        source = self.resume(session_id, provider, access)
        if not self.persistent:
            return source, agents.AgentSessionAction.CONTINUE
        capabilities = getattr(
            profile, "session_capabilities", agents.AgentSessionCapabilities()
        )
        if not isinstance(capabilities, agents.AgentSessionCapabilities):
            raise agents.AgentInvocationError(
                f"agent provider {provider} returned invalid "
                f"session capabilities"
            )
        self.branch_supported = capabilities.fork_latest
        if source is None or not self.copy_on_write:
            return source, agents.AgentSessionAction.CONTINUE
        if capabilities.fork_latest:
            return source, agents.AgentSessionAction.FORK
        if self.require_copy_on_write:
            raise agents.AgentInvocationError(
                f"agent provider {provider} cannot fork persistent session "
                f"{session_id}; checkpointing is required"
            )
        self.copy_on_write = False
        return source, agents.AgentSessionAction.CONTINUE

    def advance(
        self,
        provider: str,
        access: declarations.AgentAccess,
        provider_session_id: ids.ProviderSessionId | None,
        action: agents.AgentSessionAction = agents.AgentSessionAction.CONTINUE,
        source_provider_session_id: ids.ProviderSessionId | None = None,
        /,
    ) -> None:
        if not self.persistent:
            return
        provider_session_id = provider_session_id or self.provider_session_id
        if provider_session_id is None:
            raise agents.AgentInvocationError(
                "a persistent agent invocation returned no session id"
            )
        if (
            action is agents.AgentSessionAction.FORK
            and provider_session_id == source_provider_session_id
        ):
            self.tainted = True
            raise agents.AgentInvocationError(
                "a forked persistent agent invocation returned "
                "its source session id"
            )
        self.provider = provider
        self.provider_session_id = provider_session_id
        self.access = access
        self.copy_on_write = False


@dataclasses.dataclass(frozen=True, slots=True)
class Resources:
    profiles: Mapping[ids.AgentProfileId, agents.AgentInvoker]
    sessions: Mapping[ids.AgentSessionId, SessionResource]
    invocation_journal: _invocations.InvocationJournal | None = None
    invocation_epoch: int | None = None


NO_RESOURCES = Resources(types.MappingProxyType({}), types.MappingProxyType({}))


def invoker(value: object, label: str, /) -> agents.AgentInvoker:
    provider = getattr(value, "session_provider", None)
    if not callable(value) or not isinstance(provider, str) or not provider:
        raise TypeError(
            f"{label} must be a callable AgentInvoker with a session_provider"
        )
    return cast(agents.AgentInvoker, value)


def invocation_resources(
    graph: declarations.GraphDefinition[Any, Any, Any, Any],
    input: object,
    params: object,
    arguments: Resources,
    /,
) -> Resources:
    profiles = dict(arguments.profiles)
    for definition in graph.profiles:
        try:
            profiles[definition.id] = invoker(
                definition.implementation(input, params),
                f"agent profile {definition.id}",
            )
        except Exception as error:
            error.add_note(
                f"Verdog agent profile initialization: graph={graph.id} "
                f"profile={definition.id}"
            )
            raise

    sessions = dict(arguments.sessions)
    sessions.update(
        {
            definition.id: SessionResource(persistent=definition.persistent)
            for definition in graph.sessions
        }
    )
    return Resources(
        types.MappingProxyType(profiles),
        types.MappingProxyType(sessions),
        arguments.invocation_journal,
        arguments.invocation_epoch,
    )


def child_resource_arguments(
    operation: declarations.SubroutineCall,
    graph: declarations.GraphDefinition[Any, Any, Any, Any],
    caller: Resources,
    /,
) -> Resources:
    expected_profiles = {parameter.id for parameter in graph.profile_parameters}
    expected_sessions = {parameter.id for parameter in graph.session_parameters}
    if set(operation.profile_arguments) != expected_profiles:
        raise ValueError(f"profile resource arguments do not match {graph.id}")
    if set(operation.session_arguments) != expected_sessions:
        raise ValueError(f"session resource arguments do not match {graph.id}")

    profiles: dict[ids.AgentProfileId, agents.AgentInvoker] = {}
    for parameter in graph.profile_parameters:
        source_id = operation.profile_arguments[parameter.id]
        if source_id not in caller.profiles:
            raise ValueError(f"caller profile resource is missing: {source_id}")
        profiles[parameter.id] = caller.profiles[source_id]

    sessions: dict[ids.AgentSessionId, SessionResource] = {}
    for parameter in graph.session_parameters:
        source_id = operation.session_arguments[parameter.id]
        if source_id not in caller.sessions:
            raise ValueError(f"caller session resource is missing: {source_id}")
        sessions[parameter.id] = caller.sessions[source_id]
    return Resources(
        types.MappingProxyType(profiles),
        types.MappingProxyType(sessions),
        caller.invocation_journal,
        caller.invocation_epoch,
    )
