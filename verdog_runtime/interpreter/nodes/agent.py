"""Execute agent visits with provider and session resources."""

from __future__ import annotations

import pathlib
from typing import TypeVar, cast

from verdog_runtime import agents, declarations
from verdog_runtime import cancellation as cancellation_module
from verdog_runtime.interpreter import _agents, nodes

InputT = TypeVar("InputT")
StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")
ResultT = TypeVar("ResultT")


def execute(
    operation: declarations.Agent,
    implementation: nodes.Visit[
        InputT, StateT, declarations.AgentNodeContext[ParamsT], ResultT
    ],
    value: InputT,
    state: StateT,
    context: declarations.NodeContext[ParamsT],
    resources: _agents.Resources,
    cancellation: cancellation_module.CancellationToken,
    /,
) -> ResultT:
    """Invoke an agent visit with the resolved profile and session context."""
    return implementation(
        value,
        state,
        _agent_context(operation, context, resources, cancellation),
    )


def _agent_reply(value: object, /) -> agents.AgentReply:
    if not isinstance(value, agents.AgentReply):
        raise TypeError("agent invoker did not return AgentReply")
    text = cast(object, value.text)
    if not isinstance(text, str):
        raise TypeError("agent reply text must be a string")
    provider_session_id = cast(object, value.provider_session_id)
    if provider_session_id is not None and (
        not isinstance(provider_session_id, str) or not provider_session_id
    ):
        raise TypeError(
            "agent reply provider session id must be a non-empty string"
        )
    return value


def _invoke_with_recovery(
    profile: agents.AgentInvoker,
    request: agents.AgentRequest,
    provider: str,
    resources: _agents.Resources,
    slot: int,
    /,
) -> agents.AgentReply:
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
    operation: declarations.Agent,
    context: declarations.NodeContext[ParamsT],
    resources: _agents.Resources,
    cancellation: cancellation_module.CancellationToken,
    /,
) -> declarations.AgentNodeContext[ParamsT]:
    invocation_count = 0

    def invoke_agent(prompt: object, workspace: object, access: object) -> str:
        nonlocal invocation_count
        if not isinstance(prompt, str):
            raise TypeError("agent prompt must be a string")
        if not isinstance(workspace, pathlib.Path):
            raise TypeError("agent workspace must be a Path")
        if not isinstance(access, declarations.AgentAccess):
            raise TypeError("agent access must be AgentAccess")
        resolved_workspace = workspace.resolve()
        if not resolved_workspace.is_dir():
            raise ValueError("agent workspace must be an existing directory")
        profile = resources.profiles.get(operation.profile)
        if profile is None:
            raise agents.AgentInvocationError(
                f"agent profile resource is unavailable: {operation.profile}"
            )
        session = resources.sessions.get(operation.session)
        if session is None:
            raise agents.AgentInvocationError(
                f"agent session resource is unavailable: {operation.session}"
            )
        provider = profile.session_provider
        provider_session_id, session_action = session.request(
            operation.session, profile, access
        )
        invocation_count += 1
        artifact_dir = (
            context.output_dir / "invocations" / f"{invocation_count:06d}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=False)
        (artifact_dir / "prompt.txt").write_text(prompt, "utf-8")
        request = agents.AgentRequest(
            prompt=prompt,
            profile_id=operation.profile,
            session_id=operation.session,
            persistent=session.persistent,
            provider_session_id=provider_session_id,
            node_context=cast(declarations.NodeContext[object], context),
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

    # Pylint misses the dataclass fields inherited from NodeContext.
    # pylint: disable-next=unexpected-keyword-arg
    return declarations.AgentNodeContext(
        run_id=context.run_id,
        graph_id=context.graph_id,
        node_id=context.node_id,
        edge_id=context.edge_id,
        output_dir=context.output_dir,
        params=context.params,
        _invoke=invoke_agent,
    )
