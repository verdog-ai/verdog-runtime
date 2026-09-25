"""Persist provider prompts, responses, and diagnostic artifacts."""

from __future__ import annotations

import json

from verdog_runtime.declarations import agents as agent_declarations
from verdog_runtime.declarations import ids


def begin(
    request: agent_declarations.AgentRequest,
    provider: str,
    model: str | None,
    /,
) -> None:
    metadata(
        request,
        provider,
        model,
        status="running",
        duration_seconds=None,
        returncode=None,
        provider_session_id=request.provider_session_id,
    )


def complete(
    request: agent_declarations.AgentRequest,
    provider: str,
    model: str | None,
    reasoning: str,
    /,
    *,
    duration_seconds: float,
    returncode: int,
    provider_session_id: ids.ProviderSessionId | None,
) -> None:
    if reasoning:
        (request.artifact_dir / "reasoning.txt").write_text(reasoning, "utf-8")
    metadata(
        request,
        provider,
        model,
        status="succeeded",
        duration_seconds=duration_seconds,
        returncode=returncode,
        provider_session_id=provider_session_id,
    )


def metadata(
    request: agent_declarations.AgentRequest,
    provider: str,
    model: str | None,
    /,
    *,
    status: str,
    duration_seconds: float | None,
    returncode: int | None,
    provider_session_id: ids.ProviderSessionId | None,
) -> None:
    value = {
        "status": status,
        "profile": str(request.profile_id),
        "provider": provider,
        "model": model,
        "workspace": str(request.workspace),
        "access": str(request.access),
        "duration": duration_seconds,
        "returncode": returncode,
        "session": str(request.session_id),
        "provider_session_action": str(request.provider_session_action),
        "provider_session_source": (
            None
            if request.provider_session_id is None
            else str(request.provider_session_id)
        ),
        "provider_session": (
            None if provider_session_id is None else str(provider_session_id)
        ),
    }
    (request.artifact_dir / "metadata.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "utf-8",
    )


def failure_detail(stderr: str, events: str, /) -> str:
    detail = (stderr.strip() or events.strip() or "no diagnostic output")[
        -2000:
    ]
    return detail
