"""Invoke Claude CLI sessions and collect their responses and artifacts."""

from __future__ import annotations

import argparse
import dataclasses
from typing import cast

from verdog_runtime import declarations
from verdog_runtime.agents import _command as provider_command
from verdog_runtime.declarations import agents as agent_declarations
from verdog_runtime.declarations import ids


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ClaudeInvoker:
    """Claude CLI configuration implementing the agent invocation protocol."""

    session_provider = "claude"
    session_capabilities = agent_declarations.AgentSessionCapabilities(
        fork_latest=True
    )

    executable: str = "claude"
    model: str | None = None
    reasoning_effort: str | None = None
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject extra arguments that override runtime invocation flags."""
        provider_command.validate_extra_args(
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
                "--verbose",
                "--worktree",
                "-c",
                "-p",
                "-r",
                "-w",
            },
            "claude",
        )

    def __call__(
        self, request: agent_declarations.AgentRequest, /
    ) -> agent_declarations.AgentReply:
        """Run one request and retain its session and diagnostic artifacts."""
        return provider_command.invoke_provider(
            request,
            "claude",
            self.executable,
            self.model,
            self._command(request),
            lambda result: _response(result.events),
        )

    def _command(
        self, request: agent_declarations.AgentRequest, /
    ) -> list[str]:
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
            if (
                request.provider_session_action
                is agent_declarations.AgentSessionAction.FORK
            ):
                command.append("--fork-session")
        if self.model is not None:
            command.extend(("--model", self.model))
        if self.reasoning_effort is not None:
            command.extend(("--effort", self.reasoning_effort))
        if request.access is declarations.AgentAccess.READ_ONLY:
            # Tool selection is a provider argument; read-only access still
            # excludes editing and command execution, including repeated lists.
            parser = argparse.ArgumentParser(
                add_help=False, allow_abbrev=False, exit_on_error=False
            )
            parser.add_argument("--tools", nargs="+", action="extend")
            try:
                options, _ = parser.parse_known_args(self.extra_args)
            except argparse.ArgumentError as error:
                raise ValueError(f"claude extra_args: {error}") from error
            tool_values = cast(list[str] | None, options.tools)
            tools = (
                "Read,Glob,Grep"
                if tool_values is None
                else ",".join(tool_values)
            )
            selected = set(tools.replace(",", " ").split())
            unsupported = selected - {
                "Read",
                "Glob",
                "Grep",
                "WebSearch",
                "WebFetch",
            }
            if unsupported:
                raise ValueError(
                    "claude tools are not allowed for read-only requests: "
                    + ", ".join(sorted(unsupported))
                )
            # A provider option may consume a flag-looking value that argparse
            # recognizes. Always emit the validated restriction ourselves.
            command.extend(("--permission-mode", "plan", "--tools", tools))
        else:
            command.extend(("--permission-mode", "acceptEdits"))
        return command


def _claude_session(
    event: dict[str, object], /
) -> ids.ProviderSessionId | None:
    value = event.get("session_id")
    return (
        ids.ProviderSessionId(value)
        if isinstance(value, str) and value
        else None
    )


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


def _response(events: str, /) -> tuple[str, ids.ProviderSessionId | None, str]:
    terminal: dict[str, object] | None = None
    provider_session_id: ids.ProviderSessionId | None = None
    reasoning: list[str] = []
    for event in provider_command.json_objects(events):
        provider_session_id = provider_session_id or _claude_session(event)
        if event.get("type") == "result":
            terminal = event
        reasoning.extend(_claude_reasoning(event))
    if terminal is None:
        raise agent_declarations.AgentInvocationError(
            "claude returned no result event"
        )
    if terminal.get("is_error") is True:
        raise agent_declarations.AgentInvocationError(
            f"claude returned an error: {terminal.get('result')}"
        )
    response = terminal.get("result")
    if not isinstance(response, str):
        raise agent_declarations.AgentInvocationError(
            "claude result has no text"
        )
    return response, provider_session_id, "\n\n".join(reasoning)
