"""Normalize provider summaries and estimate incremental token costs."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import cast

from verdog_runtime._usage import TOKEN_FIELDS

_PRICING_DATE = "2026-09-27"
_OPENAI_SOURCE = "https://developers.openai.com/api/docs/pricing"
_CLAUDE_SOURCE = "https://code.claude.com/docs/en/agent-sdk/cost-tracking"
# USD per million: uncached input, cache read, cache write, output.
# Earlier models charge cache writes at their ordinary input rate.
_OPENAI_RATES = {
    "gpt-6-astra": ("10", "1", "12.5", "50"),
    "gpt-6-sol": ("2", "0.2", "2.5", "10"),
    "gpt-6-luna": ("0.1", "0.01", "0.125", "0.5"),
    "gpt-5.6-sol": ("4", "0.4", "5", "20"),
    "gpt-5.6-terra": ("2", "0.2", "2.5", "12"),
    "gpt-5.6-luna": ("0.2", "0.02", "0.25", "1.2"),
    "gpt-5.5": ("5", "0.5", "5", "30"),
    "gpt-5.4": ("2.5", "0.25", "2.5", "15"),
    "gpt-5.3-codex": ("1.75", "0.175", "1.75", "14"),
}


def _object(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _money(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return format(amount, "f") if amount.is_finite() and amount >= 0 else None


def _sum(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(cast(list[int], values))


def snapshot(
    events: str,
    provider: str,
    model: str | None,
    *,
    provider_version: str | None = None,
    command: Sequence[str] = (),
) -> dict[str, object] | None:
    """Read the latest provider summary, preserving unknown measurements."""
    records: list[dict[str, object]] = []
    for line in events.splitlines():
        try:
            # Preserve the provider's decimal cost rather than rounding it.
            value = cast(object, json.loads(line, parse_float=str))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(cast(dict[str, object], value))
    if provider == "codex":
        ambiguous = _codex_override(command)
        value = _codex(records, None if ambiguous else model)
        if value is not None and ambiguous:
            value["configured_model"] = model
            value["pricing"] = {
                "basis": "unknown",
                "reason": "provider or model configuration override",
                "configured_model": model,
            }
        return value
    if provider == "claude":
        return _claude(records, model, provider_version)
    return None


def _codex_override(command: Sequence[str]) -> bool:
    for index, argument in enumerate(command):
        option = argument.partition("=")[0]
        if option in ("--oss", "--local-provider", "--profile") or (
            argument.startswith("-p") and not argument.startswith("--")
        ):
            return True
        value = ""
        if argument in ("-c", "--config") and index + 1 < len(command):
            value = command[index + 1]
        elif argument.startswith("--config="):
            value = argument.removeprefix("--config=")
        elif argument.startswith("-c") and not argument.startswith("--"):
            value = argument[2:].removeprefix("=")
        key = value.partition("=")[0].strip().split(".")[0].strip("\"'")
        if key in (
            "model",
            "model_provider",
            "model_providers",
            "profile",
        ):
            return True
    return False


def _codex(
    records: list[dict[str, object]], model: str | None
) -> dict[str, object] | None:
    terminal: dict[str, object] = {}
    session: str | None = None
    for event in records:
        session = session or _text(
            event.get("thread_id")
            or event.get("session_id")
            or event.get("conversation_id")
        )
        if event.get("type") in ("turn.completed", "turn.failed"):
            terminal = event
    usage = _object(terminal.get("usage"))
    if not usage:
        return None
    return {
        **{key: _count(usage.get(key)) for key in TOKEN_FIELDS},
        "cost_usd": None,
        "model": model,
        "provider": "codex",
        "provider_session_id": session,
        "provider_version": None,
        "scope": "session",
        "pricing": {},
        "raw": {"usage": usage},
    }


def _claude(
    records: list[dict[str, object]],
    model: str | None,
    provider_version: str | None,
) -> dict[str, object] | None:
    terminal: dict[str, object] = {}
    session: str | None = None
    for event in records:
        session = session or _text(event.get("session_id"))
        if event.get("type") == "system" and event.get("subtype") == "init":
            provider_version = (
                _text(event.get("claude_code_version"))
                or _text(event.get("version"))
                or provider_version
            )
        if event.get("type") == "result":
            terminal = event
    if not terminal:
        return None
    version = re.search(r"(\d+)\.(\d+)\.(\d+)", provider_version or "")
    scope = (
        "unknown"
        if version is None
        else "session"
        if tuple(map(int, version.groups())) >= (2, 1, 277)
        else "invocation"
    )
    models = _object(terminal.get("modelUsage"))
    entries = [_object(value) for value in models.values()]
    counts = {
        target: _sum([_count(entry.get(source)) for entry in entries])
        for target, source in (
            ("input_tokens", "inputTokens"),
            ("cached_input_tokens", "cacheReadInputTokens"),
            ("cache_write_input_tokens", "cacheCreationInputTokens"),
            ("output_tokens", "outputTokens"),
        )
    }
    counts["input_tokens"] = _sum([counts[key] for key in TOKEN_FIELDS[:3]])
    cost = _money(terminal.get("total_cost_usd"))
    # A crashed process can emit a synthetic, zeroed final result. It does
    # not prove zero spend, nor a trustworthy baseline for a later resume.
    if terminal.get("subtype") == "error_during_execution" and (
        cost is None or Decimal(cost) == 0
    ):
        counts = dict.fromkeys(counts)
        cost = None
    if any(entry.get("costBasis") == "unknown" for entry in entries):
        cost = None
    return {
        **counts,
        "reasoning_output_tokens": None,
        "cost_usd": cost,
        "model": next(iter(models))
        if len(models) == 1
        else (None if models else model),
        "provider": "claude",
        "provider_session_id": session,
        "provider_version": provider_version,
        "scope": scope,
        "pricing": {
            "basis": "provider-estimate",
            "source": _CLAUDE_SOURCE,
            "provider_version": provider_version,
            "cost_basis": {
                name: _object(value).get("costBasis")
                for name, value in models.items()
            },
        },
        "raw": {
            key: terminal.get(key)
            for key in ("usage", "modelUsage", "total_cost_usd", "subtype")
        },
    }


def price(
    delta: dict[str, object], current: dict[str, object]
) -> tuple[str | None, dict[str, object]]:
    """Price an invocation delta, never an undifferenced session total."""
    if current.get("provider") == "claude":
        return _money(delta.get("cost_usd")), _object(current.get("pricing"))
    if _object(current.get("pricing")).get("basis") == "unknown":
        return None, _object(current.get("pricing"))
    model = _text(current.get("model"))
    rates = _OPENAI_RATES.get(model or "")
    metadata: dict[str, object] = {
        "basis": "openai-standard-short-context-estimate",
        "as_of": _PRICING_DATE,
        "source": (
            f"https://developers.openai.com/api/docs/models/{model}"
            if model in ("gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4")
            else _OPENAI_SOURCE
        ),
        "model": model,
        "excludes": ["tools", "regional uplifts", "long-context uplifts"],
    }
    if current.get("provider") != "codex" or rates is None:
        return None, {**metadata, "reason": "unknown model or provider"}
    metadata["rates_usd_per_million"] = dict(
        zip(
            ("uncached_input", "cached_input", "cache_write_input", "output"),
            rates,
            strict=True,
        )
    )
    input_rate, cached_rate, write_rate, output_rate = map(Decimal, rates)
    inputs = _count(delta.get("input_tokens"))
    cached = _count(delta.get("cached_input_tokens"))
    writes = _count(delta.get("cache_write_input_tokens"))
    outputs = _count(delta.get("output_tokens"))
    if writes is None and input_rate == write_rate:
        writes = 0  # This category has the same price as uncached input.
    if inputs is None or cached is None or writes is None or outputs is None:
        return None, {**metadata, "reason": "missing token categories"}
    if cached + writes > inputs:
        return None, {**metadata, "reason": "inconsistent token categories"}
    amount = (
        (inputs - cached - writes) * input_rate
        + cached * cached_rate
        + writes * write_rate
        + outputs * output_rate
    ) / 1_000_000
    return format(amount, "f"), metadata
