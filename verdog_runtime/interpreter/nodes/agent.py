from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast

from ...cancellation import CancellationToken
from ...agents import AgentInvocationError, AgentInvoker, AgentReply, AgentRequest
from ...declarations import Agent, AgentAccess, AgentNodeContext, NodeContext
from .._agents import Resources
from . import Visit

InputT = TypeVar("InputT")
StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")
ResultT = TypeVar("ResultT")


def execute(
    operation: Agent,
    implementation: Visit[InputT, StateT, AgentNodeContext[ParamsT], ResultT],
    value: InputT,
    state: StateT,
    context: NodeContext[ParamsT],
    resources: Resources,
    cancellation: CancellationToken,
    /,
) -> ResultT:
    return implementation(
        value, state, _agent_context(operation, context, resources, cancellation)
    )


def _agent_reply(value: object, /) -> AgentReply:
    if not isinstance(value, AgentReply):
        raise TypeError("agent invoker did not return AgentReply")
    text = cast(object, value.text)
    if not isinstance(text, str):
        raise TypeError("agent reply text must be a string")
    provider_session_id = cast(object, value.provider_session_id)
    if provider_session_id is not None and (
        not isinstance(provider_session_id, str) or not provider_session_id
    ):
        raise TypeError("agent reply provider session id must be a non-empty string")
    return value


def _invoke_with_recovery(
    profile: AgentInvoker,
    request: AgentRequest,
    provider: str,
    resources: Resources,
    slot: int,
    /,
) -> AgentReply:
    journal = resources.invocation_journal
    if journal is None:
        return _agent_reply(profile(request))
    epoch = resources.invocation_epoch
    if epoch is None:
        raise RuntimeError("agent invocation journal has no boundary epoch")
    address = journal.address(
        request,
        transition_epoch=epoch,
        slot=slot,
    )
    recorded = journal.prepare(address, request, provider)
    if recorded is None:
        reply = _agent_reply(profile(request))
        recorded = journal.complete(address, request, provider, reply)
    return _agent_reply(recorded.reply)


def _agent_context(
    operation: Agent,
    context: NodeContext[ParamsT],
    resources: Resources,
    cancellation: CancellationToken,
    /,
) -> AgentNodeContext[ParamsT]:
    invocation_count = 0

    def invoke_agent(prompt: object, workspace: object, access: object) -> str:
        nonlocal invocation_count
        if not isinstance(prompt, str):
            raise TypeError("agent prompt must be a string")
        if not isinstance(workspace, Path):
            raise TypeError("agent workspace must be a Path")
        if not isinstance(access, AgentAccess):
            raise TypeError("agent access must be AgentAccess")
        resolved_workspace = workspace.resolve()
        if not resolved_workspace.is_dir():
            raise ValueError("agent workspace must be an existing directory")
        profile = resources.profiles.get(operation.profile)
        if profile is None:
            raise AgentInvocationError(
                f"agent profile resource is unavailable: {operation.profile}"
            )
        session = resources.sessions.get(operation.session)
        if session is None:
            raise AgentInvocationError(
                f"agent session resource is unavailable: {operation.session}"
            )
        provider = profile.session_provider
        provider_session_id, session_action = session.request(
            operation.session, profile, access
        )
        invocation_count += 1
        artifact_dir = context.output_dir / "invocations" / f"{invocation_count:06d}"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        (artifact_dir / "prompt.txt").write_text(prompt, "utf-8")
        request = AgentRequest(
            prompt=prompt,
            profile_id=operation.profile,
            session_id=operation.session,
            persistent=session.persistent,
            provider_session_id=provider_session_id,
            node_context=cast(NodeContext[object], context),
            workspace=resolved_workspace,
            access=access,
            artifact_dir=artifact_dir,
            provider_session_action=session_action,
            cancellation=cancellation,
        )
        reply = _invoke_with_recovery(
            profile,
            request,
            provider,
            resources,
            invocation_count,
        )
        session.advance(
            provider,
            access,
            reply.provider_session_id,
            session_action,
            provider_session_id,
        )
        (artifact_dir / "response.txt").write_text(reply.text, "utf-8")
        return reply.text

    return AgentNodeContext(
        run_id=context.run_id,
        graph_id=context.graph_id,
        node_id=context.node_id,
        edge_id=context.edge_id,
        output_dir=context.output_dir,
        params=context.params,
        _invoke=invoke_agent,
    )
