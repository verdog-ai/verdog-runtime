from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from verdog_runtime.agents import (
    AgentInvocationError,
    AgentRequest,
    _command,
    _usage,
)
from verdog_runtime.declarations import AgentAccess, NodeContext
from verdog_runtime.declarations.ids import (
    AgentProfileId,
    AgentSessionId,
    EdgeId,
    GraphId,
    NodeId,
    RunId,
)


def _events(*values: dict[str, object]) -> str:
    return "\n".join(map(json.dumps, values))


def _codex_events() -> str:
    return _events(
        {"type": "thread.started", "thread_id": "thread"},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 400,
                "cache_write_input_tokens": 100,
                "output_tokens": 80,
                "reasoning_output_tokens": 60,
            },
        },
    )


def test_codex_preserves_cumulative_categories_and_never_guesses_model() -> (
    None
):
    current = _usage.snapshot("bad json\n[]\n" + _codex_events(), "codex", None)
    assert current is not None
    assert current["input_tokens"] == 1000
    assert current["output_tokens"] == 80
    assert current["reasoning_output_tokens"] == 60
    assert current["scope"] == "session"
    assert current["provider_session_id"] == "thread"
    assert current["model"] is None
    assert _usage.price(current, current)[0] is None
    missing = _usage.snapshot(
        _events({"type": "turn.completed", "usage": {"input_tokens": 0}}),
        "codex",
        "gpt-6-sol",
    )
    assert missing is not None
    assert missing["input_tokens"] == 0
    assert missing["output_tokens"] is None
    assert _usage.price(missing, missing)[0] is None


def test_codex_prices_only_delta_and_does_not_double_charge_subsets() -> None:
    current = _usage.snapshot(_codex_events(), "codex", "gpt-6-sol")
    assert current is not None
    delta: dict[str, object] = {
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "cache_write_input_tokens": 10,
        "output_tokens": 8,
        "reasoning_output_tokens": 6,
    }
    amount, pricing = _usage.price(delta, current)
    assert amount is not None
    assert Decimal(amount) == Decimal("0.000213")
    assert pricing["as_of"] == "2026-09-27"
    assert pricing["basis"] == "openai-standard-short-context-estimate"
    assert pricing["rates_usd_per_million"] == {
        "uncached_input": "2",
        "cached_input": "0.2",
        "cache_write_input": "2.5",
        "output": "10",
    }


@pytest.mark.parametrize(
    "options",
    (
        ("--oss",),
        ("--local-provider", "ollama"),
        ("-p", "custom"),
        ("--profile=custom",),
        ("-pcustom",),
        ("--config", 'model="other"'),
        ("-c", 'model_provider="proxy"'),
        ("--config=model_provider=proxy",),
        ("-cmodel=other",),
        ("-c=model_provider=proxy",),
        ("-c", 'model_providers.openai.base_url="https://example.test"'),
    ),
)
def test_codex_custom_provider_configuration_cannot_imply_openai_prices(
    options: tuple[str, ...],
) -> None:
    current = _usage.snapshot(
        _codex_events(),
        "codex",
        "gpt-6-sol",
        command=("codex", "exec", *options, "-m", "gpt-6-sol"),
    )
    assert current is not None
    assert current["model"] is None
    assert current["configured_model"] == "gpt-6-sol"
    assert current["input_tokens"] == 1000
    amount, pricing = _usage.price(current, current)
    assert amount is None
    assert pricing["basis"] == "unknown"


def test_codex_reasoning_configuration_keeps_known_model_pricing() -> None:
    current = _usage.snapshot(
        _codex_events(),
        "codex",
        "gpt-6-sol",
        command=("codex", "exec", "-c", 'model_reasoning_effort="high"'),
    )
    assert current is not None
    assert current["model"] == "gpt-6-sol"
    assert _usage.price(current, current)[0] is not None


@pytest.mark.parametrize("invalid", (True, -1, "12", 1.5, None))
def test_invalid_token_counts_remain_unknown(invalid: object) -> None:
    current = _usage.snapshot(
        _events({"type": "turn.completed", "usage": {"input_tokens": invalid}}),
        "codex",
        "gpt-6-sol",
    )
    assert current is not None
    assert current["input_tokens"] is None


def _claude_events(version: str | None = "2.1.283") -> str:
    return _events(
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session",
            "claude_code_version": version,
        },
        {
            "type": "result",
            "subtype": "error_max_budget_usd",
            "is_error": True,
            "total_cost_usd": "0.1234567890123456789",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "modelUsage": {
                "claude-main": {
                    "inputTokens": 100,
                    "cacheReadInputTokens": 300,
                    "cacheCreationInputTokens": 20,
                    "outputTokens": 50,
                    "costBasis": "list",
                },
                "claude-subagent": {
                    "inputTokens": 10,
                    "cacheReadInputTokens": 30,
                    "cacheCreationInputTokens": 2,
                    "outputTokens": 5,
                    "costBasis": "managed",
                },
            },
        },
    )


@pytest.mark.parametrize(
    ("version", "scope"),
    (("2.1.276", "invocation"), ("2.1.277", "session"), (None, "unknown")),
)
def test_claude_preserves_whole_tree_error_usage_and_version_scope(
    version: str | None, scope: str
) -> None:
    current = _usage.snapshot(_claude_events(version), "claude", "opus")
    assert current is not None
    assert current["input_tokens"] == 462
    assert current["cached_input_tokens"] == 330
    assert current["cache_write_input_tokens"] == 22
    assert current["output_tokens"] == 55
    assert current["reasoning_output_tokens"] is None
    assert current["cost_usd"] == "0.1234567890123456789"
    assert current["model"] is None
    assert current["scope"] == scope
    assert _usage.price({"cost_usd": "0.0234567890123456789"}, current)[0] == (
        "0.0234567890123456789"
    )


def test_claude_missing_whole_tree_usage_and_crash_are_not_zero_spend() -> None:
    current = _usage.snapshot(
        _events(
            {"type": "assistant", "message": {"usage": {"input_tokens": 10}}},
            {
                "type": "result",
                "subtype": "error_during_execution",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "modelUsage": {},
                "total_cost_usd": 0,
            },
        ),
        "claude",
        None,
    )
    assert current is not None
    assert current["input_tokens"] is None
    assert current["output_tokens"] is None
    assert current["cost_usd"] is None
    assert _usage.snapshot('{"type":"assistant"}', "claude", None) is None
    assert _usage.snapshot(_codex_events(), "custom", None) is None


def _request(path: Path) -> AgentRequest:
    return AgentRequest(
        prompt="prompt",
        profile_id=AgentProfileId("profile"),
        session_id=AgentSessionId("session"),
        persistent=True,
        provider_session_id=None,
        node_context=NodeContext[object](
            run_id=RunId("run"),
            graph_id=GraphId("graph"),
            node_id=NodeId("agent"),
            edge_id=EdgeId("enter"),
            output_dir=path,
            params=None,
        ),
        workspace=path,
        access=AgentAccess.READ_ONLY,
        artifact_dir=path,
    )


@pytest.mark.parametrize("failure", ("nonzero", "parse", "cancel", "success"))
def test_usage_capture_survives_every_provider_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    (tmp_path / "usage.json").write_text("{}", "utf-8")
    captured: list[dict[str, object] | None] = []

    def run(
        _args: Sequence[str], _request: AgentRequest, /
    ) -> _command.CommandResult:
        if failure == "cancel":
            (tmp_path / "events.jsonl").write_text(_codex_events(), "utf-8")
            raise KeyboardInterrupt
        return _command.CommandResult(
            returncode=1 if failure == "nonzero" else 0,
            events=_codex_events(),
            stderr="failure",
        )

    def parse(_result: _command.CommandResult) -> tuple[str, None, str]:
        if failure == "parse":
            raise AgentInvocationError("invalid reply")
        return "reply", None, ""

    def capture(_path: Path, value: dict[str, object] | None) -> None:
        captured.append(value)
        raise OSError("accounting failure cannot mask provider outcome")

    monkeypatch.setattr(_command, "run_command", run)
    monkeypatch.setattr("verdog_runtime._usage.capture", capture)

    def invoke() -> str:
        return _command.invoke_provider(
            _request(tmp_path), "codex", "unused", None, [], parse
        ).text

    if failure == "success":
        assert invoke() == "reply"
    else:
        expected = (
            KeyboardInterrupt if failure == "cancel" else AgentInvocationError
        )
        with pytest.raises(expected):
            invoke()
    assert len(captured) == 1
    assert captured[0] is not None
    assert captured[0]["input_tokens"] == 1000


@pytest.mark.parametrize("times_out", (False, True))
def test_claude_version_probe_is_bounded_cached_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, times_out: bool
) -> None:
    (tmp_path / "usage.json").write_text("{}", "utf-8")
    versions: list[list[str]] = []
    snapshots: list[dict[str, object] | None] = []
    cleaned: list[object] = []
    executable = str(tmp_path / "claude")

    class Probe:
        returncode = 0

        def __init__(self, command: list[str], **kwargs: object) -> None:
            versions.append(command)
            assert kwargs["stdin"] == subprocess.DEVNULL
            assert kwargs["cwd"] == tmp_path

        def __enter__(self) -> Probe:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def communicate(self, *, timeout: float) -> tuple[bytes, bytes]:
            assert timeout == 2
            if times_out:
                raise subprocess.TimeoutExpired([executable], timeout)
            return b"2.1.283 (Claude Code)", b""

    def cleanup(process: object, *, owns_group: bool) -> None:
        cleaned.append(process)

    def capture(_path: Path, value: dict[str, object] | None) -> None:
        snapshots.append(value)

    monkeypatch.setattr(_command.subprocess, "Popen", Probe)
    monkeypatch.setattr(
        "verdog_runtime._process.cleanup_after_interruption", cleanup
    )
    monkeypatch.setattr("verdog_runtime._usage.capture", capture)
    for version in (None, None, "2.1.276"):
        _command._capture_usage(  # pyright: ignore[reportPrivateUsage]
            _request(tmp_path),
            "claude",
            executable,
            None,
            _command.CommandResult(0, _claude_events(version), ""),
        )
    assert versions == [[executable, "--version"]]
    assert len(cleaned) == int(times_out)
    assert [value["scope"] for value in snapshots if value is not None] == [
        "unknown" if times_out else "session",
        "unknown" if times_out else "session",
        "invocation",
    ]


@pytest.mark.parametrize("interrupted", [False, True])
def test_cancelled_accounting_keeps_usage_without_launching_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool
) -> None:
    (tmp_path / "usage.json").write_text("{}", "utf-8")
    events = _claude_events(None)
    (tmp_path / "events.jsonl").write_text(events, "utf-8")
    request = _request(tmp_path)
    if not interrupted:
        request.cancellation.cancel()
    snapshots: list[dict[str, object] | None] = []

    def probe(_executable: str, _workspace: Path) -> None:
        pytest.fail("cancelled accounting must not launch a provider probe")

    def capture(_path: Path, value: dict[str, object] | None) -> None:
        snapshots.append(value)

    monkeypatch.setattr(_command, "_provider_version", probe)
    monkeypatch.setattr("verdog_runtime._usage.capture", capture)
    _command._capture_usage(  # pyright: ignore[reportPrivateUsage]
        request,
        "claude",
        "unused",
        None,
        None if interrupted else _command.CommandResult(0, events, ""),
    )
    assert len(snapshots) == 1
    assert snapshots[0] is not None
    assert snapshots[0]["scope"] == "unknown"
    assert snapshots[0]["cost_usd"] == "0.1234567890123456789"
