"""Invoke Codex CLI sessions and collect their responses and artifacts."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
from typing import cast

from verdog_runtime.agents import _command as provider_command
from verdog_runtime.declarations import agents as agent_declarations
from verdog_runtime.declarations import ids


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CodexInvoker:
    """Codex CLI configuration implementing the agent invocation protocol."""

    session_provider = "codex"
    session_capabilities = agent_declarations.AgentSessionCapabilities(
        fork_latest=True
    )

    executable: str = "codex"
    model: str | None = None
    reasoning_effort: str | None = None
    extra_args: tuple[str, ...] = ()
    # Enables Codex's native web_search tool when the profile opts in.
    web_search: bool = False

    def __post_init__(self) -> None:
        """Reject extra arguments that override runtime invocation flags."""
        provider_command.validate_extra_args(
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

    def __call__(
        self, request: agent_declarations.AgentRequest, /
    ) -> agent_declarations.AgentReply:
        """Run one request and retain its session and diagnostic artifacts."""
        with tempfile.TemporaryDirectory() as temporary:
            last_message = pathlib.Path(temporary) / "last-message.txt"
            return provider_command.invoke_provider(
                request,
                "codex",
                self.executable,
                self.model,
                self._command(request, last_message),
                lambda result: _provider_result(result, last_message),
            )

    def _command(
        self,
        request: agent_declarations.AgentRequest,
        last_message: pathlib.Path,
        /,
    ) -> list[str]:
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
                    "model_reasoning_effort="
                    + json.dumps(self.reasoning_effort),
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
                if request.provider_session_action
                is agent_declarations.AgentSessionAction.FORK
                else "resume"
            )
            command.extend((subcommand, str(request.provider_session_id)))
        command.append("-")
        return command


def _codex_session(event: dict[str, object], /) -> ids.ProviderSessionId | None:
    session = (
        event.get("thread_id")
        or event.get("session_id")
        or event.get("conversation_id")
    )
    return (
        ids.ProviderSessionId(session)
        if isinstance(session, str) and session
        else None
    )


def _codex_event_text(
    event: dict[str, object], /
) -> tuple[str | None, str | None]:
    message: str | None = None
    top_level_reasoning = event.get("type") == "agent_reasoning"
    reasoning_value = (
        event.get("text") or event.get("reasoning")
        if top_level_reasoning
        else None
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
    result: provider_command.CommandResult, last_message: pathlib.Path, /
) -> tuple[str, ids.ProviderSessionId | None, str]:
    try:
        response = last_message.read_text("utf-8")
    except FileNotFoundError:
        response = ""
    provider_session_id: ids.ProviderSessionId | None = None
    messages: list[str] = []
    reasoning: list[str] = []
    for event in provider_command.json_objects(result.events):
        provider_session_id = provider_session_id or _codex_session(event)
        message, thought = _codex_event_text(event)
        if message is not None:
            messages.append(message)
        if thought is not None:
            reasoning.append(thought)
    if not response:
        if not messages:
            raise agent_declarations.AgentInvocationError(
                "codex returned no final response"
            )
        response = messages[-1]
    return response, provider_session_id, "\n\n".join(reasoning)
