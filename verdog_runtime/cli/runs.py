"""Discover and describe durable local workflow runs."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from tabulate import tabulate
from .._run_model import RUN_HISTORY_SCHEMA_VERSION
from .._run_store import (
    CONTROL_DIRECTORY,
    RUN_MANIFEST,
    CheckpointSummary,
    RunManifest,
    RunStatus,
    RunStore,
    RunStoreError,
    registered_runs,
    run_is_active,
)

from .local import Clone, WorkspaceError
from .runner import LifecycleRequest, operate


class RunCommandError(WorkspaceError):
    """A run selection or metadata failure with a stable machine code."""

    def __init__(self, message: str, *, code: str, details: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True, slots=True)
class RunRecord:
    manifest: RunManifest
    output_dir: Path
    status: RunStatus
    checkpoints: tuple[CheckpointSummary, ...]

    def as_summary(self) -> dict[str, object]:
        return self.manifest.as_summary(status=self.status)


@dataclass(frozen=True, slots=True)
class DiscoveryIssue:
    path: Path
    code: str
    message: str
    details: object = None
    run_id: str | None = None
    directory_name: str | None = None


@dataclass(frozen=True, slots=True)
class RunDiscovery:
    runs: tuple[RunRecord, ...]
    issues: tuple[DiscoveryIssue, ...]


def _issue(path: Path, error: RunStoreError) -> DiscoveryIssue:
    raw_details: object = error.details
    details = (
        cast(dict[str, object], raw_details)
        if isinstance(raw_details, dict)
        else {}
    )
    run_id = details.get("run_id")
    directory_name = details.get("directory_name")
    return DiscoveryIssue(
        path=path,
        code=error.code,
        message=str(error),
        details=cast(object, raw_details),
        run_id=run_id if isinstance(run_id, str) else None,
        directory_name=(
            directory_name if isinstance(directory_name, str) else None
        ),
    )


def _default_outputs(project_root: Path) -> tuple[Path, ...]:
    root = project_root.resolve() / ".verdog" / "runs"
    if not root.is_dir() or root.is_symlink():
        return ()
    outputs: list[Path] = []
    for directory, children, _ in os.walk(root):
        path = Path(directory)
        if (path / CONTROL_DIRECTORY / RUN_MANIFEST).exists():
            outputs.append(path.resolve())
            # A run's artifacts may be large and may contain copied run metadata.
            children.clear()
        else:
            children[:] = [
                name
                for name in children
                if not name.startswith(".") and not (path / name).is_symlink()
            ]
    return tuple(outputs)


def _registered_outputs(
    project_root: Path,
) -> tuple[tuple[Path, ...], DiscoveryIssue | None]:
    try:
        return registered_runs(project_root), None
    except RunStoreError as error:
        return (), _issue(project_root.resolve() / ".verdog/run-registry.json", error)


def _load_record(project_root: Path, output_dir: Path, /) -> RunRecord:
    project = project_root.resolve()
    output = output_dir.resolve()
    store = RunStore(output)
    manifest = store.manifest()
    if Path(manifest.project_root).resolve() != project:
        raise RunStoreError(
            f"run belongs to another project: {output}",
            code="run.project_mismatch",
            details={
                "output_dir": str(output),
                "project_root": manifest.project_root,
            },
        )
    status = manifest.status
    if status is RunStatus.RUNNING and not run_is_active(output):
        status = RunStatus.INTERRUPTED
    return RunRecord(manifest, output, status, store.checkpoints())


def _updated_sort_key(value: str, /) -> tuple[int, datetime, str]:
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        instant = datetime.fromisoformat(normalized)
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=UTC)
    except ValueError:
        return 0, datetime.min.replace(tzinfo=UTC), value
    return 1, instant.astimezone(UTC), value


def discover_runs(project_root: Path) -> RunDiscovery:
    """Find default and registered custom outputs without importing workflows."""

    project = project_root.resolve()
    registered, registry_issue = _registered_outputs(project)
    candidates = {*_default_outputs(project), *registered}
    found: list[RunRecord] = []
    issues = [] if registry_issue is None else [registry_issue]
    for output in sorted(candidates, key=str):
        try:
            found.append(_load_record(project, output))
        except RunStoreError as error:
            issues.append(_issue(output, error))
    found.sort(
        key=lambda run: (*_updated_sort_key(run.manifest.updated_at), run.manifest.id),
        reverse=True,
    )
    return RunDiscovery(tuple(found), tuple(issues))


def _workflow_matches(manifest: RunManifest, requested: str) -> bool:
    identifiers = {
        manifest.workflow.id,
        manifest.workflow.definition_id,
        manifest.workflow.id.rsplit(".", 1)[-1],
        manifest.workflow.definition_id.rsplit(".", 1)[-1],
    }
    return requested in identifiers


def filtered_runs(
    discovery: RunDiscovery,
    *,
    workflow: str | None = None,
    statuses: Iterable[RunStatus] = (),
) -> tuple[RunRecord, ...]:
    selected_statuses = frozenset(statuses)
    return tuple(
        run
        for run in discovery.runs
        if (workflow is None or _workflow_matches(run.manifest, workflow))
        and (not selected_statuses or run.status in selected_statuses)
    )


def _path_reference(reference: str) -> bool:
    return (
        reference.startswith((".", "~", os.sep))
        or os.sep in reference
        or (os.altsep is not None and os.altsep in reference)
    )


def _record_from_path(
    project_root: Path,
    reference: str,
    start: Path,
    known: Sequence[RunRecord],
) -> RunRecord:
    raw = Path(reference).expanduser()
    output = (start / raw).resolve() if not raw.is_absolute() else raw.resolve()
    existing = next((run for run in known if run.output_dir == output), None)
    if existing is not None:
        return existing
    try:
        return _load_record(project_root, output)
    except RunStoreError as error:
        raise RunCommandError(
            str(error), code=error.code, details=error.details
        ) from error


def _issue_matches(issue: DiscoveryIssue, reference: str) -> bool:
    return (
        issue.run_id == reference
        or (issue.run_id is not None and issue.run_id.startswith(reference))
        or issue.directory_name == reference
    )


def select_run(
    project_root: Path,
    reference: str | None,
    *,
    start: Path | None = None,
) -> tuple[RunRecord, RunDiscovery]:
    """Resolve a path, UUID/prefix, or managed directory basename."""

    discovery = discover_runs(project_root)
    if reference is not None and _path_reference(reference):
        return _record_from_path(
            project_root, reference, start or Path.cwd(), discovery.runs
        ), discovery
    matches = discovery.runs
    if reference is not None:
        matches = tuple(
            run
            for run in matches
            if run.manifest.id == reference
            or run.manifest.id.startswith(reference)
            or run.manifest.directory_name == reference
        )
    issue_matches = tuple(
        issue
        for issue in discovery.issues
        if (
            issue.run_id is not None
            if reference is None
            else _issue_matches(issue, reference)
        )
    )
    match_count = len(matches) + len(issue_matches)
    if match_count == 1 and matches:
        return matches[0], discovery
    if match_count == 1 and issue_matches:
        issue = issue_matches[0]
        raise RunCommandError(
            issue.message, code=issue.code, details=issue.details
        )
    candidate_runs = matches or (() if issue_matches else discovery.runs)
    candidates: list[dict[str, object]] = [
        {
            "id": run.manifest.id,
            "directory_name": run.manifest.directory_name,
            "output_dir": str(run.output_dir),
        }
        for run in candidate_runs
    ]
    candidates.extend(
        {
            "id": issue.run_id,
            "directory_name": issue.directory_name,
            "output_dir": str(issue.path),
        }
        for issue in issue_matches
    )
    details = {
        "reference": reference,
        "candidates": candidates,
        "issues": _issue_details(discovery.issues),
    }
    if match_count == 0:
        description = (
            "no local run was found"
            if reference is None
            else f"run not found: {reference}"
        )
        if discovery.issues:
            count = len(discovery.issues)
            noun = "entry" if count == 1 else "entries"
            verb = "was" if count == 1 else "were"
            description += f"; {count} unreadable run {noun} {verb} ignored"
        raise RunCommandError(
            description,
            code="run.not_found",
            details=details,
        )
    raise RunCommandError(
        "more than one run matches; provide a run id, unique prefix, or path",
        code="run.selection_ambiguous",
        details=details,
    )


def _warn(issues: Sequence[DiscoveryIssue]) -> None:
    for issue in issues:
        print(
            f"verdog: ignored {issue.path}: {issue.message} [{issue.code}]",
            file=sys.stderr,
        )


def _issue_details(issues: Sequence[DiscoveryIssue]) -> list[dict[str, str]]:
    return [
        {"path": str(issue.path), "code": issue.code, "message": issue.message}
        for issue in issues
    ]


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    return str(
        tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True)
    )


def _checkpoint_label(sequence: int | None) -> str:
    return "-" if sequence is None else str(sequence)


def _runs_human(runs: Sequence[RunRecord]) -> str:
    if not runs:
        return "No runs found."
    rows = [
        (
            run.manifest.id,
            run.status.value,
            run.manifest.workflow.id,
            _checkpoint_label(run.manifest.checkpoints.latest_completed),
            run.manifest.updated_at,
            run.output_dir,
        )
        for run in runs
    ]
    return _table(
        ("RUN", "STATUS", "WORKFLOW", "CHECKPOINT", "UPDATED", "OUTPUT"), rows
    )


def list_runs(
    clone: Clone,
    *,
    workflow: str | None = None,
    statuses: Sequence[str] = (),
    as_json: bool = False,
) -> int:
    discovery = discover_runs(clone.root)
    selected = filtered_runs(
        discovery,
        workflow=workflow,
        statuses=tuple(RunStatus(status) for status in statuses),
    )
    _warn(discovery.issues)
    if as_json:
        print(
            json.dumps(
                {
                    "schema_version": RUN_HISTORY_SCHEMA_VERSION,
                    "operation": "runs",
                    "project": str(clone.root.resolve()),
                    "runs": [run.as_summary() for run in selected],
                },
                indent=2,
            )
        )
    else:
        print(_runs_human(selected))
    return 0


def _boundary_label(value: object) -> str:
    if value is None:
        return "-"
    boundary = value
    graph = getattr(boundary, "graph", "?")
    node = getattr(boundary, "node", "?")
    visit = getattr(boundary, "visit", "?")
    call_path = getattr(boundary, "call_path", ".")
    prefix = "" if call_path in {"", "."} else f"{call_path}:"
    return f"{prefix}{graph}/{node}#{visit}"


def _checkpoints_human(run: RunRecord, checkpoints: Sequence[CheckpointSummary]) -> str:
    heading = f"Run {run.manifest.id} ({run.status.value}) — {run.output_dir}"
    if not checkpoints:
        return heading + "\n\nNo checkpoints found."
    rows = [
        (
            checkpoint.sequence,
            checkpoint.kind.value,
            _boundary_label(checkpoint.completed),
            _boundary_label(checkpoint.next),
            "yes" if checkpoint.restore_available else "no",
            "yes" if checkpoint.fork_with_branch_available else "no",
            checkpoint.created_at,
        )
        for checkpoint in checkpoints
    ]
    return (
        heading
        + "\n\n"
        + _table(
            ("CHECKPOINT", "KIND", "COMPLETED", "NEXT", "RESTORE", "BRANCH", "CREATED"),
            rows,
        )
    )


def list_checkpoints(
    clone: Clone,
    *,
    reference: str | None = None,
    as_json: bool = False,
    start: Path | None = None,
) -> int:
    run, discovery = select_run(clone.root, reference, start=start)
    _warn(discovery.issues)
    checkpoints = run.checkpoints
    if as_json:
        print(
            json.dumps(
                {
                    "schema_version": RUN_HISTORY_SCHEMA_VERSION,
                    "operation": "checkpoints",
                    "run": run.as_summary(),
                    "checkpoints": [item.as_summary() for item in checkpoints],
                },
                indent=2,
            )
        )
    else:
        print(_checkpoints_human(run, checkpoints))
    return 0


def _workflow_local_id(run: RunRecord) -> str:
    return run.manifest.workflow.id.rsplit(".", 1)[-1]


def _require_inactive(run: RunRecord) -> None:
    if run_is_active(run.output_dir):
        raise RunCommandError(
            f"run is active: {run.manifest.id}",
            code="run.active",
            details={"run_id": run.manifest.id, "output_dir": str(run.output_dir)},
        )


def _run_checkpoints(run: RunRecord, /) -> tuple[CheckpointSummary, ...]:
    return run.checkpoints


def _checkpoint_by_sequence(run: RunRecord, sequence: int) -> CheckpointSummary:
    found = next(
        (
            checkpoint
            for checkpoint in _run_checkpoints(run)
            if checkpoint.sequence == sequence
        ),
        None,
    )
    if found is None:
        raise RunCommandError(
            f"checkpoint {sequence} does not exist for run {run.manifest.id}",
            code="checkpoint.not_found",
            details={"run_id": run.manifest.id, "sequence": sequence},
        )
    return found


def resume_run(
    clone: Clone,
    *,
    reference: str | None,
    retry_incomplete: bool = False,
    as_json: bool = False,
) -> int:
    """Resume the selected run at its latest exactly committed boundary."""

    run, discovery = select_run(clone.root, reference)
    _warn(discovery.issues)
    _require_inactive(run)
    if run.status is RunStatus.SUCCEEDED:
        raise RunCommandError(
            "a succeeded run cannot be resumed; fork a checkpoint or restart it",
            code="run.already_succeeded",
            details={"run_id": run.manifest.id},
        )
    checkpoint = run.manifest.checkpoints.latest_completed
    if checkpoint is None or not run.manifest.checkpoints.resume_available:
        raise RunCommandError(
            run.manifest.checkpoints.unavailable_reason
            or "the run has no exactly resumable checkpoint",
            code=run.manifest.checkpoints.unavailable_code
            or "checkpoint.resume_unavailable",
            details={"run_id": run.manifest.id, "checkpoint": checkpoint},
        )
    summary = _checkpoint_by_sequence(run, checkpoint)
    if not summary.fork_with_branch_available:
        raise RunCommandError(
            "the latest checkpoint cannot restore its persistent conversations",
            code="session.branch_unavailable",
            details={"run_id": run.manifest.id, "checkpoint": checkpoint},
        )
    return operate(
        clone,
        LifecycleRequest(
            operation="resume",
            source=run.output_dir,
            workflow_id=_workflow_local_id(run),
            sessions="restore",
            checkpoint=checkpoint,
            arguments=run.manifest.launch.workflow_arguments,
            arguments_mode="checkpoint",
            retry_incomplete=retry_incomplete,
            as_json=as_json,
        ),
    )


def restart_run(
    clone: Clone,
    *,
    reference: str | None,
    sessions: str,
    arguments: tuple[str, ...] | None,
    as_json: bool = False,
) -> int:
    """Start a new lineage child from the beginning of a prior launch."""

    run, discovery = select_run(clone.root, reference)
    _warn(discovery.issues)
    _require_inactive(run)
    if sessions not in {"branch", "fresh"}:
        raise RunCommandError(
            f"unsupported session policy: {sessions}", code="session.policy_invalid"
        )
    session_policy: Literal["branch", "fresh"] = (
        "branch" if sessions == "branch" else "fresh"
    )
    checkpoint: int | None = None
    if sessions == "branch":
        summary = next(
            (
                item
                for item in reversed(_run_checkpoints(run))
                if item.fork_with_branch_available
            ),
            None,
        )
        if summary is None:
            raise RunCommandError(
                "this run has no checkpoint from which conversations can branch",
                code="session.branch_unavailable",
                details={"run_id": run.manifest.id},
            )
        checkpoint = summary.sequence
    selected_arguments = (
        run.manifest.launch.workflow_arguments if arguments is None else arguments
    )
    return operate(
        clone,
        LifecycleRequest(
            operation="restart",
            source=run.output_dir,
            workflow_id=_workflow_local_id(run),
            sessions=session_policy,
            checkpoint=checkpoint,
            arguments=tuple(selected_arguments),
            arguments_mode="reused" if arguments is None else "overridden",
            as_json=as_json,
        ),
    )


def fork_run(
    clone: Clone,
    *,
    reference: str | None,
    checkpoint: int,
    sessions: str,
    as_json: bool = False,
) -> int:
    """Continue a committed workflow state in a new lineage child."""

    run, discovery = select_run(clone.root, reference)
    _warn(discovery.issues)
    _require_inactive(run)
    if sessions not in {"branch", "fresh"}:
        raise RunCommandError(
            f"unsupported session policy: {sessions}", code="session.policy_invalid"
        )
    session_policy: Literal["branch", "fresh"] = (
        "branch" if sessions == "branch" else "fresh"
    )
    summary = _checkpoint_by_sequence(run, checkpoint)
    available = (
        summary.fork_with_branch_available
        if sessions == "branch"
        else summary.fork_with_fresh_available
    )
    if not available:
        raise RunCommandError(
            summary.unavailable_reason
            or f"checkpoint {checkpoint} cannot be forked with {sessions} sessions",
            code=summary.unavailable_code or "checkpoint.fork_unavailable",
            details={
                "run_id": run.manifest.id,
                "checkpoint": checkpoint,
                "sessions": sessions,
            },
        )
    return operate(
        clone,
        LifecycleRequest(
            operation="fork",
            source=run.output_dir,
            workflow_id=_workflow_local_id(run),
            sessions=session_policy,
            checkpoint=checkpoint,
            arguments=run.manifest.launch.workflow_arguments,
            arguments_mode="checkpoint",
            as_json=as_json,
        ),
    )


__all__ = [
    "DiscoveryIssue",
    "RunCommandError",
    "RunDiscovery",
    "RunRecord",
    "discover_runs",
    "filtered_runs",
    "fork_run",
    "list_checkpoints",
    "list_runs",
    "restart_run",
    "resume_run",
    "select_run",
]
