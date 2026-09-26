"""Durable run metadata is usable without importing an authored workflow."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import verdog_runtime._artifact_references as artifact_references
import verdog_runtime._run_metadata as run_metadata
from verdog_runtime._run_store import (
    Boundary,
    CheckpointKind,
    CheckpointPolicy,
    CheckpointSummary,
    RunStatus,
    RunStore,
    RunStoreError,
    SessionState,
    checkpoint_summaries,
    load_checkpoint_summary,
    load_run_manifest,
    registered_runs,
    run_is_active,
)


def _store(tmp_path: Path, *, name: str = "custom-output") -> RunStore:
    project = tmp_path / "project"
    output = tmp_path / name
    project.mkdir(exist_ok=True)
    output.mkdir()
    return RunStore.create(
        output,
        project_root=project,
        workflow_id="main",
        definition_id="example.project.main",
        module="example.project.workflows.main",
        workflow_arguments=("--input.value", "7"),
        checkpointing=CheckpointPolicy.REQUIRED,
        run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        compatibility={"graph": "abc123", "python": "3.13"},
        started_at="2026-09-17T10:00:00Z",
    )


def _checkpoint(
    sequence: int,
    *,
    restorable: bool = True,
    branchable: bool | None = None,
    reason: str | None = None,
) -> CheckpointSummary:
    if branchable is None:
        branchable = restorable
    return CheckpointSummary(
        sequence=sequence,
        created_at=f"2026-09-17T10:00:0{sequence}Z",
        kind=CheckpointKind.NODE,
        completed=Boundary(".", "main", "propose", sequence, "."),
        next=Boundary(".", "main", "prove", sequence, "."),
        restore_available=restorable,
        fork_with_branch_available=branchable,
        fork_with_fresh_available=restorable,
        unavailable_code=None
        if restorable
        else "checkpoint.value_unserializable",
        unavailable_reason=reason,
    )


def test_create_writes_versioned_metadata_and_registers_custom_output(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    manifest = store.manifest()

    assert manifest.id == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    assert manifest.status is RunStatus.RUNNING
    assert manifest.workflow.id == "main"
    assert manifest.launch.workflow_arguments == ("--input.value", "7")
    assert manifest.launch.checkpointing is CheckpointPolicy.REQUIRED
    assert manifest.compatibility == {"graph": "abc123", "python": "3.13"}
    assert registered_runs(tmp_path / "project") == (store.output_dir,)

    run_document = json.loads(
        (store.control_dir / "run.json").read_text("utf-8")
    )
    assert run_document["schema_version"] == 2
    assert "checkpoints" not in run_document
    assert "directory_name" not in run_document
    assert "sessions" not in run_document

    assert not (store.control_dir / "launch.json").exists()
    assert not run_is_active(store.output_dir)
    with store.lease():
        assert run_is_active(store.output_dir)
        with (
            pytest.raises(RunStoreError, match="already active") as captured,
            RunStore.open(store.output_dir).lease(),
        ):
            pass
        assert captured.value.code == "run.active"
    assert not run_is_active(store.output_dir)
    with (
        pytest.raises(BlockingIOError, match="authored failure"),
        store.lease(),
    ):
        raise BlockingIOError("authored failure")

    updated = store.update(status=RunStatus.SUCCEEDED)
    assert updated.status is RunStatus.SUCCEEDED
    assert updated.updated_at >= manifest.updated_at


def test_checkpoint_commit_is_atomic_and_latest_boundary_controls_resume(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = store.commit_checkpoint(
        _checkpoint(1),
        shards={"root.pkl": b"root", "children/worker.pkl": b"worker"},
    )

    assert first.checkpoints.count == 1
    assert first.checkpoints.latest_completed == 1
    assert first.checkpoints.latest_restorable == 1
    assert first.checkpoints.resume_available
    assert store.checkpoint_shard(1, "root.pkl") == b"root"
    assert store.checkpoint_shard(1, "children/worker.pkl") == b"worker"
    assert store.checkpoint_shard_path(1, "root.pkl").is_file()
    checkpoint_manifest = json.loads(
        (store.checkpoint_directory(1) / "manifest.json").read_text("utf-8")
    )
    assert checkpoint_manifest["shards"][0]["name"] == "children/worker.pkl"
    assert checkpoint_manifest["fork_with_fresh_available"] is True
    assert checkpoint_manifest["shards"][1]["size"] == 4
    assert len(checkpoint_manifest["shards"][1]["sha256"]) == 64
    assert [
        item.sequence for item in checkpoint_summaries(store.output_dir)
    ] == [1]
    assert not tuple((store.control_dir / "staging").iterdir())

    second = store.commit_checkpoint(
        _checkpoint(
            2, restorable=False, reason="one value cannot be serialized"
        )
    )
    assert second.checkpoints.count == 2
    assert second.checkpoints.latest_completed == 2
    assert second.checkpoints.latest_restorable == 1
    assert not second.checkpoints.resume_available
    assert (
        second.checkpoints.unavailable_code == "checkpoint.value_unserializable"
    )

    with pytest.raises(RunStoreError, match="already committed"):
        store.commit_checkpoint(_checkpoint(2))

    with pytest.raises(RunStoreError, match="next sequence 3") as skipped:
        store.commit_checkpoint(_checkpoint(4))
    assert skipped.value.code == "checkpoint.sequence_invalid"

    store.checkpoint_shard_path(1, "root.pkl").write_bytes(b"changed")
    with pytest.raises(RunStoreError, match="integrity check") as corrupt:
        store.checkpoint_shard(1, "root.pkl")
    assert corrupt.value.code == "checkpoint.shard_corrupt"


def test_checkpoint_directory_is_authoritative_without_rewriting_run_manifest(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run_path = store.control_dir / "run.json"
    before = run_path.read_bytes()
    sessions = SessionState(persistent=1)
    checkpoint = replace(
        _checkpoint(1), created_at="2026-09-17T10:00:00.100000Z"
    )
    committed = store.commit_checkpoint(
        checkpoint,
        shards={"runtime.pkl": b"continuation"},
        sessions=sessions,
    )
    checkpoint_document = json.loads(
        (store.checkpoint_directory(1) / "manifest.json").read_text("utf-8")
    )
    assert run_path.read_bytes() == before
    assert checkpoint_document["schema_version"] == 3
    assert checkpoint_document["sessions"] == sessions.as_json()
    assert committed.sessions == sessions

    reopened = RunStore.open(store.output_dir).manifest()
    assert reopened.checkpoints.count == 1
    assert reopened.checkpoints.latest_completed == 1
    assert reopened.checkpoints.latest_restorable == 1
    assert reopened.checkpoints.resume_available
    assert reopened.sessions == sessions
    assert reopened.updated_at == checkpoint.created_at
    assert store.checkpoint_shard(1, "runtime.pkl") == b"continuation"


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "damage", ["missing_manifest", "file", "symlink", "broken_symlink"]
)
def test_damaged_latest_checkpoint_never_silently_rolls_back(
    tmp_path: Path, *, damage: str, cached: bool
) -> None:
    store = _store(tmp_path)
    store.commit_checkpoint(_checkpoint(1))
    store.commit_checkpoint(_checkpoint(2))
    reader = RunStore.open(store.output_dir) if cached else None
    directory = store.checkpoint_directory(2)
    expected_code = "checkpoint.directory_invalid"
    if damage == "missing_manifest":
        (directory / "manifest.json").unlink()
        expected_code = "checkpoint.manifest_unreadable"
    else:
        saved = store.control_dir / "saved-checkpoint"
        directory.rename(saved)
        if damage == "file":
            directory.write_text("not a checkpoint", encoding="utf-8")
        else:
            target = saved if damage == "symlink" else saved / "missing"
            directory.symlink_to(target, target_is_directory=True)

    with pytest.raises(RunStoreError) as captured:
        if reader is None:
            RunStore.open(store.output_dir)
        else:
            reader.manifest()
    assert captured.value.code == expected_code


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="requires POSIX named pipes"
)
@pytest.mark.parametrize("replaced_at_open", [False, True])
def test_registry_rejects_named_pipes_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    replaced_at_open: bool,
) -> None:
    store = _store(tmp_path)
    project = tmp_path / "project"
    registry = project / ".verdog/run-registry.json"
    real_open = os.open

    if not replaced_at_open:
        registry.unlink()
        os.mkfifo(registry)

    def guarded_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(path) == registry:
            if replaced_at_open:
                registry.unlink()
                os.mkfifo(registry)
            # Fail safely instead of hanging pytest if the reader regresses.
            assert flags & os.O_NONBLOCK, "opening a FIFO would block discovery"
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    with pytest.raises(RunStoreError) as captured:
        registered_runs(project)
    assert captured.value.code == "run.registry_unreadable"
    assert captured.value.details == {"path": str(registry)}
    assert store.output_dir.is_dir()


@pytest.mark.parametrize("version", [1, 2])
def test_old_checkpoint_formats_are_rejected(
    tmp_path: Path, version: int
) -> None:
    store = _store(tmp_path)
    store.commit_checkpoint(_checkpoint(1), shards={"runtime.pkl": b"state"})
    path = store.checkpoint_directory(1) / "manifest.json"
    document = json.loads(path.read_text("utf-8"))
    document["schema_version"] = version
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RunStoreError, match="unsupported.*schema"):
        RunStore.open(store.output_dir)


def test_preserves_independent_fresh_fork_availability(tmp_path: Path) -> None:
    store = _store(tmp_path)
    checkpoint = replace(
        _checkpoint(1, branchable=False),
        fork_with_fresh_available=False,
    )
    store.commit_checkpoint(checkpoint)
    path = store.checkpoint_directory(1) / "manifest.json"
    document = json.loads(path.read_text("utf-8"))

    assert document["fork_with_fresh_available"] is False
    assert not load_checkpoint_summary(path).fork_with_fresh_available
    reopened = RunStore.open(store.output_dir).checkpoints()[0]
    assert not reopened.fork_with_fresh_available


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        ".",
        "D:/outside.pkl",
        "D:outside.pkl",
        "child/D:outside.pkl",
    ],
)
def test_invalid_shards_never_publish_a_partial_checkpoint(
    tmp_path: Path, name: str
) -> None:
    store = _store(tmp_path)

    with pytest.raises(
        RunStoreError, match="unsafe checkpoint shard"
    ) as captured:
        store.commit_checkpoint(_checkpoint(1), shards={name: b"no"})
    assert captured.value.code == "checkpoint.shard_invalid"
    assert not store.checkpoint_directory(1).exists()
    assert not tuple((store.control_dir / "staging").iterdir())

    with pytest.raises(RunStoreError, match="unavailable"):
        store.checkpoint_shard(1, "missing.pkl")

    store.commit_checkpoint(_checkpoint(1), shards={"runtime.pkl": b"state"})
    manifest_path = store.checkpoint_directory(1) / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["shards"].append(dict(manifest["shards"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunStoreError, match="integrity check") as duplicate:
        store.checkpoint_shard(1, "runtime.pkl")
    assert duplicate.value.code == "checkpoint.shard_corrupt"


def test_open_reader_detects_a_new_authoritative_checkpoint(
    tmp_path: Path,
) -> None:
    writer = _store(tmp_path)
    reader = RunStore.open(writer.output_dir)
    assert not reader.checkpoints()

    writer.commit_checkpoint(_checkpoint(1), shards={"runtime.pkl": b"state"})

    refreshed = reader.manifest()
    assert refreshed.checkpoints.latest_completed == 1
    assert [item.sequence for item in reader.checkpoints()] == [1]
    assert reader.checkpoint_shard(1, "runtime.pkl") == b"state"


def test_checkpoint_is_parsed_once_for_summary_shards_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    trace = store.output_dir / "trace"
    trace.write_text("first\n", encoding="utf-8")
    store.commit_checkpoint(
        _checkpoint(1),
        shards={
            "root.pkl": b"root",
            "children/first.pkl": b"first",
            "children/second.pkl": b"second",
        },
        capture_artifacts=True,
    )
    manifest_path = store.checkpoint_directory(1) / "manifest.json"
    real_read = run_metadata._read_object  # pyright: ignore[reportPrivateUsage]
    reads = 0

    def counting_read(path: Path, *, code: str) -> dict[str, Any]:
        nonlocal reads
        if path == manifest_path:
            reads += 1
        return real_read(path, code=code)

    monkeypatch.setattr(run_metadata, "_read_object", counting_read)
    reader = RunStore.open(store.output_dir)
    assert reader.manifest().checkpoints.count == 1
    assert reader.checkpoint_shards(1) == {
        "root.pkl": b"root",
        "children/first.pkl": b"first",
        "children/second.pkl": b"second",
    }
    assert reader.artifact_references_available(1)
    assert [item.sequence for item in reader.checkpoints()] == [1]
    assert reads == 1

    (store.output_dir / "second").write_text("second\n", encoding="utf-8")
    reader.commit_checkpoint(_checkpoint(2), capture_artifacts=True)
    assert reads == 1


def test_checkpoint_replacement_during_its_single_parse_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    store.commit_checkpoint(_checkpoint(1), shards={"runtime.pkl": b"state"})
    manifest_path = store.checkpoint_directory(1) / "manifest.json"
    real_read = run_metadata._read_object  # pyright: ignore[reportPrivateUsage]

    def replacing_read(path: Path, *, code: str) -> dict[str, Any]:
        value = real_read(path, code=code)
        if path == manifest_path:
            replacement = path.with_name("replacement.json")
            replacement.write_text(json.dumps(value), encoding="utf-8")
            replacement.replace(path)
        return value

    monkeypatch.setattr(run_metadata, "_read_object", replacing_read)
    with pytest.raises(
        RunStoreError, match="changed while it was read"
    ) as race:
        load_checkpoint_summary(manifest_path)
    assert race.value.code == "checkpoint.manifest_unreadable"


def test_checkpoint_metadata_rejects_impossible_availability_and_sequence_gaps(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    impossible = replace(
        _checkpoint(1, restorable=False),
        fork_with_fresh_available=True,
    )
    with pytest.raises(
        RunStoreError, match="availability flags"
    ) as availability:
        store.commit_checkpoint(impossible)
    assert availability.value.code == "checkpoint.manifest_invalid"
    assert not store.checkpoint_directory(1).exists()

    with pytest.raises(RunStoreError, match="session availability") as sessions:
        store.commit_checkpoint(
            _checkpoint(1),
            sessions=SessionState(branch_available=False),
        )
    assert sessions.value.code == "checkpoint.manifest_invalid"

    store.commit_checkpoint(_checkpoint(1))
    store.commit_checkpoint(_checkpoint(2))
    shutil.rmtree(store.checkpoint_directory(1))
    with pytest.raises(RunStoreError, match="sequence has a gap") as gap:
        store.checkpoints()
    assert gap.value.code == "checkpoint.sequence_gap"

    # Validation must scale with the checkpoint count, not an untrusted
    # sequence.
    path = store.checkpoint_directory(2) / "manifest.json"
    document = json.loads(path.read_text("utf-8"))
    document["sequence"] = 10**12
    path.write_text(json.dumps(document), encoding="utf-8")
    path.parent.rename(store.checkpoint_directory(document["sequence"]))
    with pytest.raises(RunStoreError) as large_gap:
        store.checkpoints()
    assert large_gap.value.code == "checkpoint.sequence_gap"


def test_same_run_resume_requires_a_branchable_conversation_anchor(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    manifest = store.commit_checkpoint(_checkpoint(1, branchable=False))

    assert manifest.checkpoints.latest_completed == 1
    assert manifest.checkpoints.latest_restorable == 1
    assert not manifest.checkpoints.resume_available
    assert (
        manifest.checkpoints.unavailable_code
        == "checkpoint.session_branch_unavailable"
    )
    assert "provider conversations" in str(
        manifest.checkpoints.unavailable_reason
    )


def test_manifest_rejects_a_moved_or_retargeted_output(tmp_path: Path) -> None:
    store = _store(tmp_path)
    path = store.control_dir / "run.json"
    value = json.loads(path.read_text("utf-8"))
    value["output_dir"] = str(tmp_path / "someone-elses-run")
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RunStoreError, match="different output") as captured:
        load_run_manifest(store.output_dir)
    assert captured.value.code == "run.output_mismatch"


def test_manifest_refuses_a_symlinked_control_directory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    real = tmp_path / "moved-control"
    store.control_dir.rename(real)
    store.control_dir.symlink_to(real, target_is_directory=True)

    with pytest.raises(RunStoreError, match="unsafe") as captured:
        load_run_manifest(store.output_dir)
    assert captured.value.code == "run.control_invalid"


def test_artifact_references_only_write_metadata_and_reuse_verified_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    visit = store.output_dir / "graph-main/propose/000001"
    visit.mkdir(parents=True)
    payload = visit / "result.txt"
    payload.write_text("stable result\n", encoding="utf-8")
    payload.chmod(0o740)
    (visit / "empty").mkdir()
    inode = payload.stat().st_ino
    reads: list[Path] = []
    real_read = artifact_references._read_artifact  # pyright: ignore[reportPrivateUsage]

    def read(*args: Any, **kwargs: Any) -> Any:
        reads.append(args[1])
        return real_read(*args, **kwargs)

    def forbid_copy(*args: Any, **kwargs: Any) -> None:
        pytest.fail(
            "checkpointing must not copy, reflink, or hard-link artifacts"
        )

    monkeypatch.setattr(artifact_references, "_read_artifact", read)
    monkeypatch.setattr(
        artifact_references, "_materialize_artifact_bytes", forbid_copy
    )
    monkeypatch.setattr(os, "link", forbid_copy)
    for sequence in (1, 2):
        store.commit_checkpoint(_checkpoint(sequence), capture_artifacts=True)
        assert store.artifact_references_available(sequence)
        checkpoint = store.checkpoint_directory(sequence)
        assert {path.name for path in checkpoint.iterdir()} == {
            "manifest.json",
            "shards",
        }
        document = json.loads((checkpoint / "manifest.json").read_text("utf-8"))
        assert document["artifacts"]["kind"] == "references"
        assert [item["path"] for item in document["artifacts"]["files"]] == [
            "graph-main/propose/000001/result.txt"
        ]
        assert "graph-main/propose/000001/empty" in {
            item["path"] for item in document["artifacts"]["directories"]
        }
    assert reads == [payload.relative_to(store.output_dir)]
    assert payload.stat().st_ino == inode


def test_runtime_scratch_and_reports_are_excluded_by_ownership(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    graph = store.output_dir / "call/000001"
    graph.mkdir(parents=True)
    (graph / ".verdog-invocation.json").write_text("{}", encoding="utf-8")
    for directory in (store.output_dir, graph):
        (directory / "config.md").write_text("config", encoding="utf-8")
        (directory / "stats.md").write_text("stats", encoding="utf-8")
    (store.output_dir / "trace.log").write_text("trace", encoding="utf-8")
    (graph / "stacktrace.txt").write_text("failure", encoding="utf-8")
    scratch = graph / ".verdog/workspace"
    scratch.mkdir(parents=True)
    (scratch / "mutable.txt").write_text("scratch", encoding="utf-8")
    (scratch / "unsafe").symlink_to(tmp_path / "missing")
    ordinary = graph / "node/000001"
    ordinary.mkdir(parents=True)
    (ordinary / "config.md").write_text("user config", encoding="utf-8")
    (ordinary / "stats.md").write_text("user stats", encoding="utf-8")
    (ordinary / "trace.log").write_text("user trace", encoding="utf-8")
    first = store.capture_artifacts()
    files = {
        item["path"] for item in cast(list[dict[str, object]], first["files"])
    }
    assert files == {
        "call/000001/.verdog-invocation.json",
        "call/000001/stacktrace.txt",
        "call/000001/node/000001/config.md",
        "call/000001/node/000001/stats.md",
        "call/000001/node/000001/trace.log",
    }
    store.commit_checkpoint(_checkpoint(1), artifact_references=first)
    shutil.rmtree(scratch)
    (graph / "stats.md").write_text("new stats", encoding="utf-8")
    (store.output_dir / "trace.log").write_text("more trace", encoding="utf-8")
    store.commit_checkpoint(_checkpoint(2), capture_artifacts=True)
    store.validate_artifacts(1)


@pytest.mark.parametrize("damage", ["modify", "remove", "mode", "symlink"])
def test_failed_capture_preserves_immutable_files_and_previous_checkpoint(
    tmp_path: Path, damage: str
) -> None:
    store = _store(tmp_path)
    payload = store.output_dir / "result.txt"
    payload.write_text("stable", encoding="utf-8")
    store.commit_checkpoint(_checkpoint(1), capture_artifacts=True)
    if damage == "modify":
        payload.write_text("modified", encoding="utf-8")
    elif damage == "remove":
        payload.unlink()
    elif damage == "mode":
        payload.chmod(0o700)
    else:
        payload.unlink()
        payload.symlink_to(tmp_path / "missing")
    with pytest.raises(RunStoreError) as failure:
        store.commit_checkpoint(_checkpoint(2), capture_artifacts=True)
    assert failure.value.code in {
        "checkpoint.artifact_corrupt",
        "checkpoint.artifact_unsafe",
    }
    assert store.checkpoint_directory(1).is_dir()
    assert not store.checkpoint_directory(2).exists()
    assert not tuple((store.control_dir / "staging").iterdir())
    with pytest.raises(RunStoreError):
        RunStore.open(store.output_dir).validate_artifacts(1)
    with pytest.raises(RunStoreError):
        store.materialize_artifacts(1, tmp_path / "fork")
    assert not (tmp_path / "fork").exists()


def test_remote_inventory_is_persisted_without_late_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    (store.output_dir / "before.txt").write_text("before", encoding="utf-8")
    captured = store.capture_artifacts()
    (store.output_dir / "later.txt").write_text("later", encoding="utf-8")

    def forbid_scan(*args: Any, **kwargs: Any) -> None:
        pytest.fail("parent must preserve the child's checkpoint boundary")

    monkeypatch.setattr(artifact_references, "_scan_artifacts", forbid_scan)
    store.commit_checkpoint(_checkpoint(1), artifact_references=captured)
    restored = store.materialize_artifacts(1, tmp_path / "fork")
    assert (restored / "before.txt").read_text("utf-8") == "before"
    assert not (restored / "later.txt").exists()
    changed = json.loads(json.dumps(captured))
    changed["files"][0]["sha256"] = "0" * 64
    with pytest.raises(RunStoreError, match="integrity check"):
        store.commit_checkpoint(_checkpoint(2), artifact_references=changed)


def test_artifact_copy_is_independent_and_preserves_boundary_and_modes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    tools = store.output_dir / "graph-main/run/000001/tools"
    tools.mkdir(parents=True)
    script = tools / "verify.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o750)
    (tools / "empty").mkdir(mode=0o750)
    store.commit_checkpoint(_checkpoint(1), capture_artifacts=True)
    (tools / "later.txt").write_text("not at boundary", encoding="utf-8")
    store.commit_checkpoint(_checkpoint(2), capture_artifacts=True)

    destination = tmp_path / "fork-output"
    destination.mkdir()
    restored = store.materialize_artifacts(1, destination)
    restored_script = restored / "graph-main/run/000001/tools/verify.sh"
    assert restored == destination
    assert restored_script.read_text("utf-8") == "#!/bin/sh\nexit 0\n"
    assert restored_script.stat().st_mode & 0o777 == 0o750
    assert restored_script.stat().st_ino != script.stat().st_ino
    assert (restored_script.parent / "empty").stat().st_mode & 0o777 == 0o750
    assert not (restored_script.parent / "later.txt").exists()
    assert not (restored / ".verdog").exists()
    restored_script.write_text("fork mutation\n", encoding="utf-8")
    assert script.read_text("utf-8") == "#!/bin/sh\nexit 0\n"
    shutil.rmtree(store.output_dir)
    assert restored_script.read_text("utf-8") == "fork mutation\n"


def test_resume_validation_seeds_capture_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    (store.output_dir / "result.txt").write_text("stable", encoding="utf-8")
    store.commit_checkpoint(_checkpoint(1), capture_artifacts=True)
    reader = RunStore.open(store.output_dir)
    reader.validate_artifacts(1)

    def forbid_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("resume already verified unchanged files")

    monkeypatch.setattr(artifact_references, "_read_artifact", forbid_read)
    reader.commit_checkpoint(_checkpoint(2), capture_artifacts=True)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape",
        "node/.verdog/workspace/file",
        ".verdog/file",
        ".",
        "D:/outside.pkl",
        "D:outside.pkl",
        "child/D:outside.pkl",
    ],
)
def test_artifact_capture_and_materialization_reject_unsafe_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (store.output_dir / "unsafe-link").symlink_to(outside)
    with pytest.raises(
        RunStoreError, match="regular file or directory"
    ) as unsafe:
        store.commit_checkpoint(
            _checkpoint(1),
            shards={"runtime.pkl": b"state"},
            capture_artifacts=True,
        )
    assert unsafe.value.code == "checkpoint.artifact_unsafe"
    assert not store.checkpoint_directory(1).exists()
    assert not tuple((store.control_dir / "staging").iterdir())
    (store.output_dir / "unsafe-link").unlink()
    (store.output_dir / "result.txt").write_text("safe", encoding="utf-8")
    store.commit_checkpoint(_checkpoint(1), capture_artifacts=True)
    manifest_path = store.checkpoint_directory(1) / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["artifacts"]["files"][0]["path"] = unsafe_path
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination = tmp_path / "unsafe-fork"
    with pytest.raises(
        RunStoreError, match="unsafe checkpoint artifact"
    ) as traversal:
        store.materialize_artifacts(1, destination)
    assert traversal.value.code == "checkpoint.artifact_unsafe"
    assert not destination.exists()


def test_artifact_materialization_refuses_nonempty_or_overlapping_outputs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    (store.output_dir / "result.txt").write_text("safe", encoding="utf-8")
    store.commit_checkpoint(_checkpoint(1), capture_artifacts=True)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "mine.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(
        RunStoreError, match="not an empty directory"
    ) as nonempty:
        store.materialize_artifacts(1, occupied)
    assert nonempty.value.code == "checkpoint.artifact_destination_not_empty"
    assert (occupied / "mine.txt").read_text("utf-8") == "keep"
    with pytest.raises(RunStoreError, match="overlaps") as overlapping:
        store.materialize_artifacts(1, store.output_dir / "fork")
    assert overlapping.value.code == "checkpoint.artifact_destination_invalid"
