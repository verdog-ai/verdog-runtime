from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import cast

from .._process import (
    cleanup_after_interruption,
    process_options,
    wait_until_reaped,
)
from ..declarations.agents import (
    AgentInvocationError,
    AgentReply,
    AgentRequest,
    AgentSessionAction,
)
from ..declarations.ids import ProviderSessionId
from ._artifacts import begin, complete, failure_detail, metadata

_PROVIDER_WAIT_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    events: str
    stderr: str


def validate_extra_args(
    arguments: object,
    reserved: Collection[str],
    provider: str,
    /,
) -> None:
    message = f"{provider} extra_args must be a tuple of strings"
    if not isinstance(arguments, tuple):
        raise TypeError(message)
    short = tuple(option for option in reserved if len(option) == 2)
    for argument in cast(tuple[object, ...], arguments):
        if not isinstance(argument, str):
            raise TypeError(message)
        option = argument.partition("=")[0]
        if option in reserved or any(
            option.startswith(candidate) for candidate in short
        ):
            raise ValueError(
                f"{provider} extra_args may not override runtime option {option}"
            )


def json_objects(events: str, /) -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = []
    for line in events.splitlines():
        try:
            value = cast(object, json.loads(line))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(cast(dict[str, object], value))
    return tuple(values)


def validate_session_request(
    request: AgentRequest,
    provider: str,
    /,
) -> None:
    action = request.provider_session_action
    if action is not AgentSessionAction.FORK:
        return
    if not request.persistent:
        raise AgentInvocationError(
            f"{provider} cannot fork a nonpersistent agent session"
        )
    if request.provider_session_id is None:
        raise AgentInvocationError(f"{provider} cannot fork without a provider session")


def run_command(command: Sequence[str], request: AgentRequest, /) -> CommandResult:
    """Run one provider while its complete transport streams go straight to disk."""

    events_path = request.artifact_dir / "events.jsonl"
    stderr_path = request.artifact_dir / "stderr.txt"
    environment = os.environ.copy()
    group_options, owns_group = process_options(environment)
    request.cancellation.raise_if_cancelled()
    with (
        (request.artifact_dir / "prompt.txt").open("rb") as prompt,
        events_path.open("wb") as events,
        stderr_path.open("wb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            stdin=prompt,
            stdout=events,
            stderr=stderr,
            cwd=request.workspace,
            env=environment,
            **group_options,
        )
        try:
            while not wait_until_reaped(
                process,
                timeout=request.cancellation.remaining(_PROVIDER_WAIT_SECONDS),
            ):
                request.cancellation.raise_if_cancelled()
        except BaseException:
            cleanup_after_interruption(process, owns_group=owns_group)
            raise
    if process.returncode is None:  # pragma: no cover - the wait reaps the process
        raise RuntimeError("agent process was not reaped")
    return CommandResult(
        returncode=process.returncode,
        events=events_path.read_text("utf-8", errors="replace"),
        stderr=stderr_path.read_text("utf-8", errors="replace"),
    )


def invoke_provider(
    request: AgentRequest,
    provider: str,
    executable: str,
    model: str | None,
    command: Sequence[str],
    parse: Callable[[CommandResult], tuple[str, ProviderSessionId | None, str]],
    /,
) -> AgentReply:
    """Own the lifecycle shared by every command-backed provider."""

    validate_session_request(request, provider)
    begin(request, provider, model)
    started = monotonic()
    returncode: int | None = None
    try:
        result = run_command(command, request)
        returncode = result.returncode
        if result.returncode != 0:
            raise AgentInvocationError(
                f"{executable} exited with {result.returncode}: "
                f"{failure_detail(result.stderr, result.events)}"
            )
        response, provider_session_id, reasoning = parse(result)
        if request.provider_session_action is AgentSessionAction.FORK:
            if provider_session_id is None:
                raise AgentInvocationError(
                    f"{provider} fork returned no provider session"
                )
            if provider_session_id == request.provider_session_id:
                raise AgentInvocationError(
                    f"{provider} fork returned its source provider session"
                )
        else:
            provider_session_id = provider_session_id or request.provider_session_id
    except (AgentInvocationError, OSError, UnicodeError, ValueError) as error:
        metadata(
            request,
            provider,
            model,
            status="failed",
            duration_seconds=monotonic() - started,
            returncode=returncode,
            provider_session_id=request.provider_session_id,
        )
        if isinstance(error, AgentInvocationError):
            raise
        raise AgentInvocationError(
            f"{executable} invocation failed: {error}"
        ) from error
    except Exception:
        raise
    except BaseException:
        metadata(
            request,
            provider,
            model,
            status="cancelled",
            duration_seconds=monotonic() - started,
            returncode=returncode,
            provider_session_id=request.provider_session_id,
        )
        raise
    complete(
        request,
        provider,
        model,
        reasoning,
        duration_seconds=monotonic() - started,
        returncode=result.returncode,
        provider_session_id=provider_session_id,
    )
    return AgentReply(text=response, provider_session_id=provider_session_id)
