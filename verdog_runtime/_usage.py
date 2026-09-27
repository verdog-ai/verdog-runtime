"""Durable per-attempt accounting, independent of graph timing rollback."""

from __future__ import annotations

import dataclasses
import decimal
import json
import pathlib
import uuid
from typing import cast

from verdog_runtime import _run_metadata
from verdog_runtime.declarations import agents

TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Scope:
    root: pathlib.Path
    run_id: str
    project_path: str
    report_path: str


def validate_snapshot(value: object) -> dict[str, object] | None:
    """Validate optional accounting carried by checkpoints and journals."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("agent usage snapshot must be an object")
    result = cast(dict[str, object], value)
    if result.get("scope") not in ("session", "invocation", "unknown"):
        raise ValueError("agent usage snapshot has an invalid scope")
    for field in TOKEN_FIELDS:
        counter = result.get(field)
        if counter is not None and (type(counter) is not int or counter < 0):
            raise ValueError("agent usage snapshot has an invalid counter")
    cost = result.get("cost_usd")
    if cost is not None and _money(cost) is None:
        raise ValueError("agent usage snapshot has an invalid cost")
    for field in ("provider", "model", "provider_session_id"):
        item = result.get(field)
        if item is not None and (not isinstance(item, str) or not item):
            raise ValueError("agent usage snapshot has an invalid identity")
    return result


def _money(value: object) -> decimal.Decimal | None:
    if not isinstance(value, str):
        return None
    try:
        number = decimal.Decimal(value)
    except decimal.InvalidOperation:
        return None
    return number if number.is_finite() and number >= 0 else None


def begin(
    request: agents.AgentRequest,
    scope: Scope | None,
    baseline: dict[str, object] | None,
) -> None:
    if scope is None:
        return
    context = request.node_context
    _run_metadata.atomic_json(
        request.artifact_dir / "usage.json",
        {
            "schema_version": 1,
            "attempt_id": str(uuid.uuid4()),
            "run_id": scope.run_id,
            "project_path": scope.project_path,
            "graph_id": str(context.graph_id),
            "node_id": str(context.node_id),
            "visit_path": context.output_dir.relative_to(scope.root).as_posix(),
            "report_path": scope.report_path,
            "source_session_id": request.provider_session_id,
            "baseline": baseline,
            "status": "running",
            "snapshot": None,
            "usage": None,
        },
    )


def _read(directory: pathlib.Path) -> dict[str, object] | None:
    path = directory / "usage.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("agent usage record must be a regular file")
    value: object = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("agent usage record must be an object")
    return cast(dict[str, object], value)


def _delta(
    snapshot: dict[str, object],
    baseline: dict[str, object] | None,
    source: object,
) -> dict[str, object]:
    fresh = source is None or snapshot.get("scope") == "invocation"
    matching = (
        baseline is not None
        and baseline.get("provider_session_id") == source
        and baseline.get("provider") == snapshot.get("provider")
        and baseline.get("scope") == snapshot.get("scope") == "session"
    )
    if matching and baseline is not None:
        for field in (*TOKEN_FIELDS, "cost_usd"):
            current, previous = snapshot.get(field), baseline.get(field)
            if field == "cost_usd":
                current, previous = _money(current), _money(previous)
            if (
                isinstance(current, (int, decimal.Decimal))
                and isinstance(previous, (int, decimal.Decimal))
                and current < previous
            ):
                # A reset breaks the session baseline for all counters.
                matching = False
                break
    result: dict[str, object] = {}
    for field in (*TOKEN_FIELDS, "cost_usd"):
        current = snapshot.get(field)
        previous = (
            0
            if fresh
            else baseline.get(field)
            if matching and baseline
            else None
        )
        if field == "cost_usd":
            current = _money(current)
            previous = decimal.Decimal(0) if fresh else _money(previous)
            result[field] = (
                str(current - previous)
                if current is not None
                and previous is not None
                and current >= previous
                else None
            )
        else:
            result[field] = (
                current - previous
                if type(current) is int
                and type(previous) is int
                and current >= previous
                else None
            )
    return result


def capture(
    directory: pathlib.Path, snapshot: dict[str, object] | None
) -> None:
    """Called by built-in providers after their subprocess has stopped."""
    record = _read(directory)
    if record is None:
        return
    snapshot = validate_snapshot(snapshot)
    record["snapshot"] = snapshot
    if snapshot is not None:
        from verdog_runtime.agents._usage import price

        usage = _delta(
            snapshot,
            validate_snapshot(record.get("baseline")),
            record.get("source_session_id"),
        )
        usage["cost_usd"], record["pricing"] = price(usage, snapshot)
        record["usage"] = usage
    _run_metadata.atomic_json(directory / "usage.json", record)


def finish(
    directory: pathlib.Path,
    status: str,
    provider_session_id: str | None,
) -> dict[str, object] | None:
    record = _read(directory)
    if record is None:
        return None
    snapshot = validate_snapshot(record.get("snapshot"))
    if snapshot is not None:
        snapshot = dict(snapshot)
        reported = snapshot.get("provider_session_id")
        if reported is not None and provider_session_id not in (None, reported):
            snapshot = None
            record["usage"] = None
        elif provider_session_id is not None:
            snapshot["provider_session_id"] = provider_session_id
    record.update(status=status, snapshot=snapshot)
    _run_metadata.atomic_json(directory / "usage.json", record)
    return snapshot


def replay(directory: pathlib.Path) -> None:
    _run_metadata.atomic_json(
        directory / "usage.json", {"schema_version": 1, "replayed": True}
    )
