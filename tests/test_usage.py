"""Incremental accounting remains independent of reply and timing recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verdog_runtime import _usage
from verdog_runtime.agents import AgentRequest
from verdog_runtime.declarations import AgentAccess, NodeContext
from verdog_runtime.declarations.ids import (
    AgentProfileId,
    AgentSessionId,
    EdgeId,
    GraphId,
    NodeId,
    ProviderSessionId,
    RunId,
)


def _snapshot(**changes: object) -> dict[str, object]:
    return {
        "scope": "session",
        "provider": "claude",
        "provider_session_id": "source",
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "cache_write_input_tokens": 10,
        "output_tokens": 30,
        "reasoning_output_tokens": 5,
        "cost_usd": "0.001",
        **changes,
    }


def _request(root: Path, source: str | None = None) -> AgentRequest:
    visit = root / "agent/000001"
    artifact = visit / "invocations/000001"
    artifact.mkdir(parents=True)
    return AgentRequest(
        prompt="test",
        profile_id=AgentProfileId("default"),
        session_id=AgentSessionId("conversation"),
        persistent=True,
        provider_session_id=None
        if source is None
        else ProviderSessionId(source),
        node_context=NodeContext(
            run_id=RunId("run"),
            graph_id=GraphId("main"),
            node_id=NodeId("agent"),
            edge_id=EdgeId("incoming"),
            output_dir=visit,
            params=None,
        ),
        workspace=root,
        access=AgentAccess.READ_ONLY,
        artifact_dir=artifact,
    )


@pytest.mark.parametrize("source", [None, "source"])
def test_usage_differences_session_totals_and_keeps_failure_spend(
    tmp_path: Path, source: str | None
) -> None:
    request = _request(tmp_path, source)
    _usage.begin(request, _usage.Scope(tmp_path, "run", ".", "."), _snapshot())
    ending = _snapshot(
        provider_session_id="fork",
        input_tokens=160,
        cached_input_tokens=40,
        cache_write_input_tokens=15,
        output_tokens=45,
        reasoning_output_tokens=8,
        cost_usd="0.0017",
    )
    _usage.capture(request.artifact_dir, ending)
    assert _usage.finish(request.artifact_dir, "failed", None) == ending
    record = json.loads((request.artifact_dir / "usage.json").read_text())
    assert record["status"] == "failed"
    assert record["snapshot"] == ending
    assert record["usage"]["input_tokens"] == (160 if source is None else 60)
    assert record["usage"]["cached_input_tokens"] == (
        40 if source is None else 20
    )
    assert record["usage"]["output_tokens"] == (45 if source is None else 15)
    assert record["usage"]["cost_usd"] == (
        "0.0017" if source is None else "0.0007"
    )


@pytest.mark.parametrize(
    "baseline,ending",
    [
        (None, _snapshot()),
        (_snapshot(provider_session_id="different"), _snapshot()),
        (_snapshot(), _snapshot(scope="unknown")),
        (_snapshot(), _snapshot(input_tokens=0, output_tokens=50)),
    ],
)
def test_unproven_or_reset_baselines_never_charge_history(
    tmp_path: Path,
    baseline: dict[str, object] | None,
    ending: dict[str, object],
) -> None:
    request = _request(tmp_path, "source")
    _usage.begin(request, _usage.Scope(tmp_path, "run", ".", "."), baseline)
    _usage.capture(request.artifact_dir, ending)
    record = json.loads((request.artifact_dir / "usage.json").read_text())
    assert all(value is None for value in record["usage"].values())


def test_invocation_scope_and_partial_counters_are_independent(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, "source")
    _usage.begin(request, _usage.Scope(tmp_path, "run", ".", "."), None)
    _usage.capture(
        request.artifact_dir, _snapshot(scope="invocation", input_tokens=None)
    )
    record = json.loads((request.artifact_dir / "usage.json").read_text())
    assert record["usage"]["input_tokens"] is None
    assert record["usage"]["output_tokens"] == 30
    assert record["usage"]["cost_usd"] == "0.001"
    assert _usage.finish(request.artifact_dir, "succeeded", "different") is None
    assert (
        json.loads((request.artifact_dir / "usage.json").read_text())["usage"]
        is None
    )


def test_snapshots_survive_journal_replay(tmp_path: Path) -> None:
    from verdog_runtime.agents import AgentReply
    from verdog_runtime.interpreter._invocations import InvocationJournal

    (tmp_path / ".verdog").mkdir()
    request = _request(tmp_path)
    journal = InvocationJournal(tmp_path)
    address = journal.address(request, transition_epoch=10, slot=1)
    assert journal.prepare(address, request, "claude") is None
    journal.complete(
        address,
        request,
        "claude",
        AgentReply(text="done"),
        usage_snapshot=_snapshot(),
    )
    replay = InvocationJournal(tmp_path).prepare(address, request, "claude")
    assert replay is not None and replay.usage_snapshot == _snapshot()
    _usage.replay(request.artifact_dir)
    assert json.loads((request.artifact_dir / "usage.json").read_text())[
        "replayed"
    ]


def test_optional_snapshot_survives_restart() -> None:
    import cloudpickle

    from verdog_runtime.interpreter._continuation import SessionSnapshot
    from verdog_runtime.interpreter.execution import (
        _restart_session_resource,  # pyright: ignore[reportPrivateUsage]
    )

    snapshot = SessionSnapshot(
        resource_id="session",
        persistent=True,
        provider="claude",
        provider_session_id="source",
        access="read_only",
        usage_snapshot=_snapshot(),
    )
    assert cloudpickle.loads(cloudpickle.dumps(snapshot)) == snapshot
    _, resource = _restart_session_resource(
        {
            "resource_id": "session",
            "persistent": True,
            "provider": "claude",
            "provider_session_id": "source",
            "access": AgentAccess.READ_ONLY.value,
            "copy_on_write": False,
            "branch_supported": True,
            "tainted": False,
            "usage_snapshot": _snapshot(),
        },
        require_copy_on_write=False,
    )
    assert resource.usage_snapshot == _snapshot()


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_tokens", -1),
        ("output_tokens", True),
        ("cost_usd", "NaN"),
        ("scope", []),
    ],
)
def test_recovery_rejects_invalid_snapshots(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _usage.validate_snapshot(_snapshot(**{field: value}))
