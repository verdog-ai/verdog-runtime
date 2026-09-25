from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from ..declarations import AgentAccess
from ..declarations.agents import (
    AgentInvocationError,
    AgentReply,
    AgentRequest,
    AgentSessionAction,
    AgentSessionCapabilities,
)
from ..declarations.ids import ProviderSessionId
from ._command import (
    invoke_provider,
    json_objects,
    validate_extra_args,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaudeInvoker:
    session_provider = "claude"
    session_capabilities = AgentSessionCapabilities(fork_latest=True)

    executable: str = "claude"
    model: str | None = None
    reasoning_effort: str | None = None
    extra_args: tuple[str, ...] = ()
    # Read-only agents get WebSearch and WebFetch only when the profile opts in.
    web_search: bool = False

    def __post_init__(self) -> None:
        validate_extra_args(
            self.extra_args,
            {
                "--add-dir",
                "--allow-dangerously-skip-permissions",
                "--allowed-tools",
                "--allowedTools",
                "--continue",
                "--dangerously-skip-permissions",
                "--effort",
                "--fork-session",
                "--model",
                "--no-session-persistence",
                "--output-format",
                "--permission-mode",
                "--print",
                "--resume",
                "--session-id",
                "--tools",
                "--verbose",
                "--worktree",
                "-c",
                "-p",
                "-r",
                "-w",
            },
            "claude",
        )

    def __call__(self, request: AgentRequest, /) -> AgentReply:
        return invoke_provider(
            request,
            "claude",
            self.executable,
            self.model,
            self._command(request),
            lambda result: _response(result.events),
        )

    def _command(self, request: AgentRequest, /) -> list[str]:
        command = [
            self.executable,
            "-p",
            *self.extra_args,
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if not request.persistent:
            command.append("--no-session-persistence")
        if request.provider_session_id is not None:
            command.extend(("--resume", str(request.provider_session_id)))
            if request.provider_session_action is AgentSessionAction.FORK:
                command.append("--fork-session")
        if self.model is not None:
            command.extend(("--model", self.model))
        if self.reasoning_effort is not None:
            command.extend(("--effort", self.reasoning_effort))
        if request.access is AgentAccess.READ_ONLY:
            tools = "Read,Glob,Grep,WebSearch,WebFetch" if self.web_search else "Read,Glob,Grep"
            command.extend(("--permission-mode", "plan", "--tools", tools))
        else:
            command.extend(("--permission-mode", "acceptEdits"))
        return command


def _claude_session(event: dict[str, object], /) -> ProviderSessionId | None:
    value = event.get("session_id")
    return ProviderSessionId(value) if isinstance(value, str) and value else None


def _claude_reasoning(event: dict[str, object], /) -> tuple[str, ...]:
    if event.get("type") != "assistant":
        return ()
    message = event.get("message")
    if not isinstance(message, dict):
        return ()
    content = cast(dict[str, object], message).get("content")
    if not isinstance(content, list):
        return ()
    found: list[str] = []
    for raw_block in cast(list[object], content):
        if not isinstance(raw_block, dict):
            continue
        block = cast(dict[str, object], raw_block)
        thinking = block.get("thinking")
        if (
            block.get("type") == "thinking"
            and isinstance(thinking, str)
            and (stripped := thinking.strip())
        ):
            found.append(stripped)
    return tuple(found)


def _response(events: str, /) -> tuple[str, ProviderSessionId | None, str]:
    terminal: dict[str, object] | None = None
    provider_session_id: ProviderSessionId | None = None
    reasoning: list[str] = []
    for event in json_objects(events):
        provider_session_id = provider_session_id or _claude_session(event)
        if event.get("type") == "result":
            terminal = event
        reasoning.extend(_claude_reasoning(event))
    if terminal is None:
        raise AgentInvocationError("claude returned no result event")
    if terminal.get("is_error") is True:
        raise AgentInvocationError(
            f"claude returned an error: {terminal.get('result')}"
        )
    response = terminal.get("result")
    if not isinstance(response, str):
        raise AgentInvocationError("claude result has no text")
    return response, provider_session_id, "\n\n".join(reasoning)
