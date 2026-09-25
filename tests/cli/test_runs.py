"""Read-only CLI views over durable local workflow runs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from verdog.local import Clone
from verdog.runner import LifecycleRequest
from verdog.runs import (
    RunCommandError,
    discover_runs,
    fork_run,
    list_checkpoints,
    list_runs,
    restart_run,
    resume_run,
    select_run,
)
from verdog_runtime._run_store import (
    Boundary,
    CheckpointKind,
    CheckpointSummary,
    RunStatus,
    RunStore,
)


def _clone(tmp_path: Path) -> Clone:
    root = tmp_path / "project"
    root.mkdir()
    return Clone(root, "http://unused", None)


def _run(
    clone: Clone,
    output: Path,
    *,
    identifier: str,
    workflow: str = "main",
    status: RunStatus = RunStatus.RUNNING,
) -> RunStore:
    output.mkdir(parents=True)
    store = RunStore.create(
        output,
        project_root=clone.root,
        workflow_id=workflow,
        definition_id=f"example.project.{workflow}",
        module=f"example.project.workflows.{workflow}",
        workflow_arguments=("--input.seed", "3"),
        run_id=identifier,
        started_at="2026-09-17T10:00:00Z",
    )
    if status is not RunStatus.RUNNING:
        store.update(status=status)
    return store


def _checkpoint(sequence: int) -> CheckpointSummary:
    return CheckpointSummary(
        sequence=sequence,
        created_at="2026-09-17T10:01:00Z",
        kind=CheckpointKind.NODE,
        completed=Boundary(".", "main", "propose", 4, "."),
        next=Boundary(".", "main", "prove", 4, "."),
        restore_available=True,
        fork_with_branch_available=True,
        fork_with_fresh_available=True,
    )


def test_discovery_includes_default_and_registered_custom_outputs(
    tmp_path: Path,
) -> None:
    clone = _clone(tmp_path)
    default = _run(
        clone,
        clone.root / ".verdog/runs/20260917T100000Z-first",
        identifier="11111111-1111-4111-8111-111111111111",
        status=RunStatus.SUCCEEDED,
    )
    custom = _run(
        clone,
        tmp_path / "custom-output",
        identifier="22222222-2222-4222-8222-222222222222",
        workflow="review",
    )

    discovery = discover_runs(clone.root)
    assert not discovery.issues
    assert {run.output_dir for run in discovery.runs} == {
        default.output_dir,
        custom.output_dir,
    }
    by_id = {run.manifest.id: run for run in discovery.runs}
    assert by_id["11111111-1111-4111-8111-111111111111"].status is RunStatus.SUCCEEDED
    # A stale `running` manifest has no held lease and is presented honestly.
    assert by_id["22222222-2222-4222-8222-222222222222"].status is RunStatus.INTERRUPTED


def test_default_discovery_finds_workflow_groups_without_a_registry(tmp_path: Path) -> None:
    clone = _clone(tmp_path)
    root = clone.root / ".verdog/runs"
    outputs = (
        root / "legacy-run",
        root / "main/new-run",
        root / "main/ipc2023/sokoban/20260925T120000Z",
        root / "main__refine_choose/new-run",
    )
    excluded = (
        outputs[1] / "copied-run",
        root / ".hidden/run",
        tmp_path / "external-run",
    )
    for index, output in enumerate((*outputs, *excluded)):
        _run(clone, output, identifier=f"run-{index}", status=RunStatus.SUCCEEDED)
    (root / "linked-run").symlink_to(excluded[-1], target_is_directory=True)
    (clone.root / ".verdog/run-registry.json").unlink()

    discovery = discover_runs(clone.root)
    assert not discovery.issues
    assert {run.output_dir for run in discovery.runs} == set(outputs)


def test_runs_json_and_human_filters_are_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clone = _clone(tmp_path)
    _run(
        clone,
        clone.root / ".verdog/runs/first",
        identifier="11111111-1111-4111-8111-111111111111",
        status=RunStatus.SUCCEEDED,
    )
    _run(
        clone,
        tmp_path / "custom-output",
        identifier="22222222-2222-4222-8222-222222222222",
        workflow="review",
    )

    assert list_runs(clone, workflow="review", as_json=True) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == 1
    assert document["operation"] == "runs"
    assert document["project"] == str(clone.root)
    assert [run["id"] for run in document["runs"]] == [
        "22222222-2222-4222-8222-222222222222"
    ]
    summary = document["runs"][0]
    assert summary["status"] == "interrupted"
    assert summary["launch"] == {
        "workflow_arguments": ["--input.seed", "3"],
        "checkpointing": "auto",
    }

    assert list_runs(clone, statuses=("succeeded",)) == 0
    output = capsys.readouterr().out
    assert "RUN" in output and "CHECKPOINT" in output
    assert "11111111-1111-4111-8111-111111111111" in output
    assert "22222222-2222-4222-8222-222222222222" not in output


def test_checkpoint_selection_accepts_prefix_basename_and_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clone = _clone(tmp_path)
    store = _run(
        clone,
        tmp_path / "custom-output",
        identifier="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    )
    store.commit_checkpoint(_checkpoint(1), shards={"root.pkl": b"state"})

    identifier = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    selectors = (
        (identifier, None),
        ("aaaaaaaa", None),
        ("custom-output", None),
        ("./custom-output", tmp_path),
        (str(store.output_dir), None),
    )
    selected = [
        select_run(clone.root, reference, start=start)[0]
        for reference, start in selectors
    ]
    expected = selected[0]
    assert all(run.output_dir == store.output_dir for run in selected)
    assert all(run.manifest == expected.manifest for run in selected)
    assert all(run.checkpoints == expected.checkpoints for run in selected)
    assert all(run.manifest.checkpoints.resume_available for run in selected)

    requests: list[LifecycleRequest] = []

    def operate(_clone: Clone, request: LifecycleRequest) -> int:
        requests.append(request)
        return 0

    monkeypatch.setattr("verdog.runs.operate", operate)
    monkeypatch.chdir(tmp_path)
    for reference in (
        identifier,
        "aaaaaaaa",
        "custom-output",
        "./custom-output",
        str(store.output_dir),
    ):
        assert resume_run(clone, reference=reference) == 0
    assert all(request.source == store.output_dir for request in requests)
    assert all(request.checkpoint == 1 for request in requests)

    assert list_checkpoints(clone, reference="aaaaaaaa", as_json=True) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == 1
    assert document["operation"] == "checkpoints"
    assert document["run"]["id"] == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    assert document["run"]["checkpoints"]["resume_available"]
    assert document["checkpoints"] == [
        {
            "sequence": 1,
            "created_at": "2026-09-17T10:01:00Z",
            "kind": "node",
            "completed": {
                "project_path": ".",
                "graph": "main",
                "node": "propose",
                "visit": 4,
                "call_path": ".",
            },
            "next": {
                "project_path": ".",
                "graph": "main",
                "node": "prove",
                "visit": 4,
                "call_path": ".",
            },
            "restore_available": True,
            "fork_with_branch_available": True,
            "fork_with_fresh_available": True,
            "unavailable_code": None,
            "unavailable_reason": None,
        }
    ]

    assert list_checkpoints(clone, reference="custom-output") == 0
    output = capsys.readouterr().out
    assert "main/propose#4" in output
    assert "main/prove#4" in output


def test_corrupt_checkpoint_selectors_surface_the_same_error(
    tmp_path: Path,
) -> None:
    clone = _clone(tmp_path)
    store = _run(
        clone,
        tmp_path / "custom-output",
        identifier="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    )
    store.commit_checkpoint(_checkpoint(1))
    manifest_path = store.checkpoint_directory(1) / "manifest.json"
    document = json.loads(manifest_path.read_text("utf-8"))
    document["artifacts"] = {"schema_version": 999}
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    selectors = (
        ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", None),
        ("aaaaaaaa", None),
        ("custom-output", None),
        ("./custom-output", tmp_path),
        (str(store.output_dir), None),
    )
    errors: list[RunCommandError] = []
    for reference, start in selectors:
        with pytest.raises(RunCommandError) as captured:
            select_run(clone.root, reference, start=start)
        errors.append(captured.value)

    expected = errors[0]
    assert expected.code == "checkpoint.artifact_manifest_invalid"
    assert all(error.code == expected.code for error in errors)
    assert all(str(error) == str(expected) for error in errors)
    assert all(error.details == expected.details for error in errors)


def test_run_listing_uses_a_committed_directory_when_run_json_stays_unchanged(
    tmp_path: Path,
) -> None:
    clone = _clone(tmp_path)
    store = _run(
        clone,
        tmp_path / "custom-output",
        identifier="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    )
    manifest_path = store.control_dir / "run.json"
    before = manifest_path.read_bytes()
    store.commit_checkpoint(_checkpoint(1))
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest_path.read_bytes() == before
    assert "checkpoints" not in manifest
    assert "sessions" not in manifest

    [discovered] = discover_runs(clone.root).runs
    assert discovered.manifest.checkpoints.count == 1
    assert discovered.manifest.checkpoints.latest_completed == 1
    assert discovered.manifest.checkpoints.resume_available
    assert discovered.manifest.updated_at == _checkpoint(1).created_at


def test_omitted_or_short_selectors_refuse_ambiguity(tmp_path: Path) -> None:
    clone = _clone(tmp_path)
    for suffix in ("1", "2"):
        _run(
            clone,
            tmp_path / f"output-{suffix}",
            identifier=f"abc{suffix}0000-0000-4000-8000-000000000000",
        )

    with pytest.raises(RunCommandError) as omitted:
        select_run(clone.root, None)
    assert omitted.value.code == "run.selection_ambiguous"
    details = cast(dict[str, object], omitted.value.details)
    assert len(cast(list[object], details["candidates"])) == 2

    with pytest.raises(RunCommandError) as missing:
        select_run(clone.root, "not-a-run")
    assert missing.value.code == "run.not_found"


def test_unreadable_registered_outputs_remain_visible_as_selection_issues(
    tmp_path: Path,
) -> None:
    clone = _clone(tmp_path)
    registry = clone.root / ".verdog/run-registry.json"
    registry.parent.mkdir()
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "run_id": "missing-run",
                        "output_dir": str(tmp_path / "missing-output"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    discovery = discover_runs(clone.root)
    assert not discovery.runs
    assert [issue.code for issue in discovery.issues] == ["run.manifest_unreadable"]
    with pytest.raises(RunCommandError, match="1 unreadable run entry") as captured:
        select_run(clone.root, None)
    details = cast(dict[str, object], captured.value.details)
    issues = cast(list[dict[str, object]], details["issues"])
    assert issues[0]["code"] == "run.manifest_unreadable"


@pytest.mark.parametrize("include_healthy", (False, True))
def test_omitted_selector_counts_identified_corrupt_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    include_healthy: bool,
) -> None:
    import verdog.main as cli

    clone = _clone(tmp_path)
    corrupt_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    store = _run(clone, tmp_path / "corrupt", identifier=corrupt_id)
    store.commit_checkpoint(_checkpoint(1))
    manifest_path = store.checkpoint_directory(1) / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["artifacts"] = {"schema_version": 999}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    healthy_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    if include_healthy:
        _run(clone, tmp_path / "healthy", identifier=healthy_id)

    def open_clone(_workflow: str) -> Clone:
        return clone

    monkeypatch.setattr(cli, "open_clone", open_clone)

    assert cli.main(["checkpoints", "--json"]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "error"
    error = response["error"]
    if include_healthy:
        assert error["code"] == "run.selection_ambiguous"
        assert {item["id"] for item in error["details"]["candidates"]} == {
            healthy_id,
            corrupt_id,
        }
    else:
        assert error["code"] == "checkpoint.artifact_manifest_invalid"
        assert error["details"]["run_id"] == corrupt_id


def test_cli_registers_read_only_commands_and_emits_machine_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import verdog.main as cli

    clone = _clone(tmp_path)

    def open_clone(_workflow: str) -> Clone:
        return clone

    monkeypatch.setattr(cli, "open_clone", open_clone)

    parsed = cli.parse_arguments(
        ["runs", "main", "--status", "failed", "--status", "interrupted", "--json"]
    )
    assert parsed.workflow == "main"
    assert parsed.status == ["failed", "interrupted"]
    assert parsed.as_json

    assert cli.main(["checkpoints", "missing", "--json"]) == 1
    error = json.loads(capsys.readouterr().out)
    assert error["schema_version"] == 1
    assert error["operation"] == "checkpoints"
    assert error["status"] == "error"
    assert error["error"]["code"] == "run.not_found"
    assert error["error"]["details"]["reference"] == "missing"


def test_lifecycle_commands_forward_explicit_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = _clone(tmp_path)
    store = _run(
        clone,
        tmp_path / "source-output",
        identifier="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    )
    store.commit_checkpoint(_checkpoint(1), shards={"runtime.pkl": b"state"})
    calls: list[LifecycleRequest] = []

    def operate(_clone: Clone, request: LifecycleRequest) -> int:
        calls.append(request)
        return 0

    monkeypatch.setattr("verdog.runs.operate", operate)

    assert resume_run(clone, reference="aaaaaaaa", retry_incomplete=True) == 0
    resume = calls[-1]
    assert resume.operation == "resume"
    assert resume.source == store.output_dir
    assert resume.sessions == "restore"
    assert resume.checkpoint == 1
    assert resume.retry_incomplete is True

    # A newer serializable boundary may be unusable for independent provider
    # conversations. Restart should fall back to the newest branchable anchor.
    store.commit_checkpoint(
        replace(_checkpoint(2), fork_with_branch_available=False),
        shards={"runtime.pkl": b"newer-state"},
    )

    assert (
        restart_run(
            clone,
            reference="aaaaaaaa",
            sessions="branch",
            arguments=None,
        )
        == 0
    )
    restart = calls[-1]
    assert restart.operation == "restart"
    assert restart.source == store.output_dir
    assert restart.checkpoint == 1
    assert restart.arguments == ("--input.seed", "3")
    assert restart.arguments_mode == "reused"

    assert (
        restart_run(
            clone,
            reference="aaaaaaaa",
            sessions="fresh",
            arguments=(),
        )
        == 0
    )
    restart_empty = calls[-1]
    assert restart_empty.arguments == ()
    assert restart_empty.arguments_mode == "overridden"

    assert (
        fork_run(
            clone,
            reference="aaaaaaaa",
            checkpoint=1,
            sessions="fresh",
        )
        == 0
    )
    fork = calls[-1]
    assert fork.operation == "fork"
    assert fork.source == store.output_dir
    assert fork.checkpoint == 1
    assert fork.sessions == "fresh"
    assert fork.arguments_mode == "checkpoint"
