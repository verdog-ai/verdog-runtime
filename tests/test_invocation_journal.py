"""External agent calls have durable ambiguity and replay semantics."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from verdog_runtime.agents import AgentReply, AgentRequest, AgentSessionAction
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
from verdog_runtime.interpreter._invocations import (
    InvocationJournal,
    InvocationJournalError,
)


def _request(output: Path, workspace: Path) -> AgentRequest:
    artifact = (
        output / "graph-main" / "agent" / "000001" / "invocations" / "000001"
    )
    artifact.mkdir(parents=True)
    return AgentRequest(
        prompt="find a proof",
        profile_id=AgentProfileId("default"),
        session_id=AgentSessionId("conversation"),
        persistent=True,
        provider_session_id=ProviderSessionId("source"),
        provider_session_action=AgentSessionAction.FORK,
        node_context=NodeContext(
            run_id=RunId("run"),
            graph_id=GraphId("main"),
            node_id=NodeId("agent"),
            edge_id=EdgeId("enter-agent"),
            output_dir=artifact.parent.parent,
            params=None,
        ),
        workspace=workspace.resolve(),
        access=AgentAccess.READ_ONLY,
        artifact_dir=artifact,
    )


def _journal_root(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "run"
    (output / ".verdog").mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return output, workspace


def test_started_invocation_refuses_plain_resume_and_retry_is_explicit(
    tmp_path: Path,
) -> None:
    output, workspace = _journal_root(tmp_path)
    request = _request(output, workspace)
    journal = InvocationJournal(output)
    address = journal.address(request, transition_epoch=9998, slot=1)

    assert journal.prepare(address, request, "test") is None
    record_path = next((output / ".verdog/invocations").glob("*.json"))
    assert json.loads(record_path.read_text("utf-8"))["status"] == "started"

    resumed = InvocationJournal(output)
    with pytest.raises(
        InvocationJournalError, match="--retry-incomplete"
    ) as captured:
        resumed.prepare(address, request, "test")
    assert captured.value.code == "invocation.ambiguous"

    retry = InvocationJournal(output, retry_incomplete=True)
    assert retry.prepare(address, request, "test") is None
    retried = json.loads(record_path.read_text("utf-8"))
    assert retried["status"] == "started"
    assert retried["attempt"] == 2


def test_completed_invocation_replays_reply_without_another_attempt(
    tmp_path: Path,
) -> None:
    output, workspace = _journal_root(tmp_path)
    request = _request(output, workspace)
    journal = InvocationJournal(output)
    address = journal.address(request, transition_epoch=9998, slot=1)
    assert journal.prepare(address, request, "test") is None
    reply = AgentReply(
        text="proved",
        provider_session_id=ProviderSessionId("branch"),
    )
    completed = journal.complete(address, request, "test", reply)
    assert completed.reply is reply
    assert not completed.replayed

    replay = InvocationJournal(output).prepare(address, request, "test")
    assert replay is not None and replay.replayed
    assert replay.reply == reply

    record_path = next((output / ".verdog/invocations").glob("*.json"))
    record = json.loads(record_path.read_text("utf-8"))
    assert record["status"] == "completed"
    assert record["request"]["prompt_sha256"] != request.prompt
    assert "find a proof" not in record_path.read_text("utf-8")


def test_resume_refuses_a_changed_request_at_the_same_boundary(
    tmp_path: Path,
) -> None:
    output, workspace = _journal_root(tmp_path)
    request = _request(output, workspace)
    journal = InvocationJournal(output)
    address = journal.address(request, transition_epoch=9998, slot=1)
    assert journal.prepare(address, request, "test") is None

    changed = AgentRequest(
        prompt="a different proof",
        profile_id=request.profile_id,
        session_id=request.session_id,
        persistent=request.persistent,
        provider_session_id=request.provider_session_id,
        provider_session_action=request.provider_session_action,
        node_context=request.node_context,
        workspace=request.workspace,
        access=request.access,
        artifact_dir=request.artifact_dir,
    )
    with pytest.raises(InvocationJournalError) as captured:
        journal.prepare(address, changed, "test")
    assert captured.value.code == "invocation.identity_mismatch"


def test_transition_epoch_separates_repeated_visits(tmp_path: Path) -> None:
    output, workspace = _journal_root(tmp_path)
    request = _request(output, workspace)
    journal = InvocationJournal(output)
    earlier = journal.address(request, transition_epoch=9998, slot=1)
    later = journal.address(request, transition_epoch=9996, slot=1)

    assert journal.prepare(earlier, request, "test") is None
    assert journal.prepare(later, request, "test") is None
    assert len(tuple((output / ".verdog/invocations").glob("*.json"))) == 2


@pytest.mark.parametrize(
    "workspace_relative", (Path("."), Path(".verdog/workspace"))
)
def test_attempt_specific_output_paths_are_normalized_for_replay(
    tmp_path: Path,
    workspace_relative: Path,
) -> None:
    output, workspace = _journal_root(tmp_path)
    first = _request(output, workspace)
    first_output = first.node_context.output_dir
    first_workspace = first_output / workspace_relative
    first_workspace.mkdir(parents=True, exist_ok=True)
    (first_workspace / "candidate.py").write_text("candidate", encoding="utf-8")
    first = replace(
        first,
        prompt=f"inspect {first_workspace / 'candidate.py'}",
        workspace=first_workspace,
    )
    journal = InvocationJournal(output)
    address = journal.address(first, transition_epoch=9998, slot=1)
    assert journal.prepare(address, first, "test") is None
    journal.complete(address, first, "test", AgentReply(text="done"))
    shutil.rmtree(first_workspace)

    second_output = first_output.parent / "000002"
    second_artifact = second_output / "invocations/000001"
    second_artifact.mkdir(parents=True)
    second_workspace = second_output / workspace_relative
    second_workspace.mkdir(parents=True, exist_ok=True)
    (second_workspace / "candidate.py").write_text(
        "candidate", encoding="utf-8"
    )
    second = replace(
        first,
        prompt=f"inspect {second_workspace / 'candidate.py'}",
        workspace=second_workspace,
        node_context=replace(first.node_context, output_dir=second_output),
        artifact_dir=second_artifact,
    )
    replay = journal.prepare(
        journal.address(second, transition_epoch=9998, slot=1),
        second,
        "test",
    )
    assert replay is not None and replay.replayed
    assert replay.reply.text == "done"
