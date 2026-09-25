from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ..declarations.agents import (
    AgentInvocationError,
    AgentReply,
    AgentRequest,
    AgentSessionAction,
    AgentSessionCapabilities,
)
from ..declarations.ids import ProviderSessionId
from ._command import (
    CommandResult,
    invoke_provider,
    json_objects,
    validate_extra_args,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexInvoker:
    session_provider = "codex"
    session_capabilities = AgentSessionCapabilities(fork_latest=True)

    executable: str = "codex"
    model: str | None = None
    reasoning_effort: str | None = None
    extra_args: tuple[str, ...] = ()
    # Enables Codex's native web_search tool when the profile opts in.
    web_search: bool = False

    def __post_init__(self) -> None:
        validate_extra_args(
            self.extra_args,
            {
                "--add-dir",
                "--approve-for-me",
                "--cd",
                "--dangerously-bypass-approvals-and-sandbox",
                "--ephemeral",
                "--json",
                "--model",
                "--output-last-message",
                "--sandbox",
                "--skip-git-repo-check",
                "-C",
                "-m",
                "-o",
                "-s",
                "fork",
                "resume",
            },
            "codex",
        )

    def __call__(self, request: AgentRequest, /) -> AgentReply:
        with TemporaryDirectory() as temporary:
            last_message = Path(temporary) / "last-message.txt"
            return invoke_provider(
                request,
                "codex",
                self.executable,
                self.model,
                self._command(request, last_message),
                lambda result: _provider_result(result, last_message),
            )

    def _command(self, request: AgentRequest, last_message: Path, /) -> list[str]:
        command = [self.executable, "exec", *self.extra_args]
        command.extend(
            (
                "--json",
                "-o",
                str(last_message),
                "--skip-git-repo-check",
                "-C",
                str(request.workspace),
            )
        )
        if self.model is not None:
            command.extend(("-m", self.model))
        if self.reasoning_effort is not None:
            command.extend(
                (
                    "-c",
                    "model_reasoning_effort=" + json.dumps(self.reasoning_effort),
                )
            )
        if not request.persistent:
            command.append("--ephemeral")
        if self.web_search:
            command.append("--search")
        command.extend(("--sandbox", str(request.access)))
        if request.provider_session_id is not None:
            subcommand = (
                "fork"
                if request.provider_session_action is AgentSessionAction.FORK
                else "resume"
            )
            command.extend((subcommand, str(request.provider_session_id)))
        command.append("-")
        return command


def _codex_session(event: dict[str, object], /) -> ProviderSessionId | None:
    session = (
        event.get("thread_id")
        or event.get("session_id")
        or event.get("conversation_id")
    )
    return ProviderSessionId(session) if isinstance(session, str) and session else None


def _codex_event_text(event: dict[str, object], /) -> tuple[str | None, str | None]:
    message: str | None = None
    top_level_reasoning = event.get("type") == "agent_reasoning"
    reasoning_value = (
        event.get("text") or event.get("reasoning") if top_level_reasoning else None
    )
    item = event.get("item")
    if isinstance(item, dict):
        value = cast(dict[str, object], item)
        kind = value.get("type")
        if kind == "agent_message":
            text = value.get("text")
            if isinstance(text, str):
                message = text
        elif kind == "reasoning" and not top_level_reasoning:
            reasoning_value = value.get("text") or value.get("summary")
    reasoning = (
        reasoning_value.strip()
        if isinstance(reasoning_value, str) and reasoning_value.strip()
        else None
    )
    return message, reasoning


def _provider_result(
    result: CommandResult, last_message: Path, /
) -> tuple[str, ProviderSessionId | None, str]:
    try:
        response = last_message.read_text("utf-8")
    except FileNotFoundError:
        response = ""
    provider_session_id: ProviderSessionId | None = None
    messages: list[str] = []
    reasoning: list[str] = []
    for event in json_objects(result.events):
        provider_session_id = provider_session_id or _codex_session(event)
        message, thought = _codex_event_text(event)
        if message is not None:
            messages.append(message)
        if thought is not None:
            reasoning.append(thought)
    if not response:
        if not messages:
            raise AgentInvocationError("codex returned no final response")
        response = messages[-1]
    return response, provider_session_id, "\n\n".join(reasoning)
