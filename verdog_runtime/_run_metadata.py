"""Encoding, validation, locking, and discovery for durable run metadata."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from ._artifact_references import ArtifactReferences

from ._file_lock import FileLockUnavailable as LockUnavailable
from ._file_lock import locked_file
from ._relative_path import strict_posix_relative_parts
from ._run_model import (
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    CONTROL_DIRECTORY,
    REGISTRY_LOCK,
    REGISTRY_PATH,
    RUN_MANIFEST,
    RUN_MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Boundary,
    CheckpointKind,
    CheckpointPolicy,
    CheckpointState,
    CheckpointSummary,
    LaunchRecord,
    ParentRun,
    RunManifest,
    RunStatus,
    RunStoreError,
    SessionIssue,
    SessionState,
    WorkflowIdentity,
)


FileSignature: TypeAlias = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class CheckpointShard:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """One immutable, fully decoded checkpoint manifest."""

    summary: CheckpointSummary
    shards: Mapping[str, CheckpointShard]
    artifacts: ArtifactReferences | None
    sessions: SessionState | None
    manifest_path: Path
    manifest_signature: FileSignature


@dataclass(frozen=True, slots=True)
class LoadedCheckpointIndex:
    checkpoints: tuple[LoadedCheckpoint, ...]
    directory_signature: FileSignature | None


@dataclass(frozen=True, slots=True)
class LoadedRun:
    manifest: RunManifest
    checkpoints: tuple[LoadedCheckpoint, ...]
    manifest_signature: FileSignature
    checkpoint_directory_signature: FileSignature | None


def file_signature(value: os.stat_result) -> FileSignature:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def latest_timestamp(values: Sequence[str], /) -> str:
    parsed: list[tuple[datetime, str]] = []
    try:
        for value in values:
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            instant = datetime.fromisoformat(normalized)
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=UTC)
            parsed.append((instant.astimezone(UTC), value))
    except ValueError:
        # Schema v1 accepted arbitrary non-empty strings. Preserve readability
        # for such metadata while all runtime-written timestamps remain RFC 3339.
        return max(values)
    return max(parsed, key=lambda item: item[0])[1]


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    staged = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(staged, 0o600)
        staged.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def remove_private_tree(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    for directory, names, files in os.walk(path, topdown=True, followlinks=False):
        current = Path(directory)
        current.chmod(0o700)
        for name in list(names):
            child = current / name
            if child.is_symlink():
                child.unlink(missing_ok=True)
                names.remove(name)
            else:
                child.chmod(0o700)
        for name in files:
            child = current / name
            if child.is_symlink():
                child.unlink(missing_ok=True)
            else:
                child.chmod(0o600)
    shutil.rmtree(path, ignore_errors=True)


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    descriptor = -1
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("not a regular file")
        descriptor = os.open(
            path,
            # A file replaced with a FIFO after lstat must not block discovery.
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        before = os.fstat(descriptor)
        if file_signature(metadata) != file_signature(before):
            raise OSError("metadata changed while it was opened")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            document = stream.read()
            after = os.fstat(stream.fileno())
        path_signature = file_signature(path.lstat())
        if (
            file_signature(before) != file_signature(after)
            or path_signature != file_signature(after)
        ):
            raise OSError("metadata changed while it was read")
        value: object = json.loads(document)
    except (OSError, UnicodeError, ValueError) as error:
        raise RunStoreError(
            f"run metadata is unreadable: {path}",
            code=code,
            details={"path": str(path)},
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise RunStoreError(
            f"run metadata is not an object: {path}",
            code=code,
            details={"path": str(path)},
        )
    return cast(dict[str, Any], value)


def _read_manifest(
    path: Path, *, code: str
) -> tuple[dict[str, Any], FileSignature]:
    """Bind decoded metadata to the file signature used by its cache."""

    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise OSError("not a regular file")
        value = _read_object(path, code=code)
        signature = file_signature(path.lstat())
    except OSError as error:
        raise RunStoreError(
            f"run metadata is unreadable: {path}",
            code=code,
            details={"path": str(path)},
        ) from error
    if signature != file_signature(before):
        raise RunStoreError(
            f"run metadata changed while it was read: {path}",
            code=code,
            details={"path": str(path)},
        )
    return value, signature


def existing_control_directory(output_dir: Path) -> Path:
    output = output_dir.resolve()
    control = output / CONTROL_DIRECTORY
    try:
        mode = control.lstat().st_mode
    except OSError as error:
        raise RunStoreError(
            f"run metadata directory is unavailable: {control}",
            code="run.manifest_unreadable",
            details={"path": str(control)},
        ) from error
    if not stat.S_ISDIR(mode):
        raise RunStoreError(
            f"run metadata directory is unsafe: {control}",
            code="run.control_invalid",
            details={"path": str(control)},
        )
    return control


def _version(
    value: Mapping[str, Any],
    path: Path,
    *,
    supported: Sequence[int] = (SCHEMA_VERSION,),
) -> int:
    version = value.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in supported
    ):
        raise RunStoreError(
            f"unsupported run metadata schema in {path}",
            code="run.schema_unsupported",
            details={"path": str(path), "schema_version": version},
        )
    return version


def _string(value: Mapping[str, Any], key: str, path: Path) -> str:
    found = value.get(key)
    if not isinstance(found, str) or not found:
        raise RunStoreError(
            f"run metadata has no valid {key}: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": key},
        )
    return found


def _optional_string(value: Mapping[str, Any], key: str, path: Path) -> str | None:
    found = value.get(key)
    if found is not None and not isinstance(found, str):
        raise RunStoreError(
            f"run metadata has an invalid {key}: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": key},
        )
    return found


def _object(value: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    found = value.get(key)
    if not isinstance(found, dict):
        raise RunStoreError(
            f"run metadata has no valid {key}: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": key},
        )
    return cast(dict[str, Any], found)


def _integer(value: Mapping[str, Any], key: str, path: Path) -> int:
    found = value.get(key)
    if not isinstance(found, int) or isinstance(found, bool) or found < 0:
        raise RunStoreError(
            f"run metadata has no valid {key}: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": key},
        )
    return found


def _optional_integer(value: Mapping[str, Any], key: str, path: Path) -> int | None:
    found = value.get(key)
    if found is None:
        return None
    return _integer(value, key, path)


def _boolean(value: Mapping[str, Any], key: str, path: Path) -> bool:
    found = value.get(key)
    if not isinstance(found, bool):
        raise RunStoreError(
            f"run metadata has no valid {key}: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": key},
        )
    return found


def _workflow(value: Mapping[str, Any], path: Path) -> WorkflowIdentity:
    workflow = _object(value, "workflow", path)
    return WorkflowIdentity(
        id=_string(workflow, "id", path),
        definition_id=_string(workflow, "definition_id", path),
        module=_string(workflow, "module", path),
    )


def _launch(value: Mapping[str, Any], path: Path) -> LaunchRecord:
    launch = _object(value, "launch", path)
    arguments = launch.get("workflow_arguments")
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) for item in cast(list[object], arguments)
    ):
        raise RunStoreError(
            f"run metadata has invalid workflow arguments: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": "launch.workflow_arguments"},
        )
    try:
        policy = CheckpointPolicy(_string(launch, "checkpointing", path))
    except ValueError as error:
        raise RunStoreError(
            f"run metadata has an invalid checkpoint policy: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": "launch.checkpointing"},
        ) from error
    return LaunchRecord(tuple(cast(list[str], arguments)), policy)


def _parent(value: Mapping[str, Any], path: Path) -> ParentRun | None:
    raw = value.get("parent")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RunStoreError(
            f"run metadata has an invalid parent: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": "parent"},
        )
    parent = cast(dict[str, Any], raw)
    checkpoint = _optional_integer(parent, "checkpoint", path)
    return ParentRun(
        run_id=_string(parent, "run_id", path),
        operation=_string(parent, "operation", path),
        checkpoint=checkpoint,
        arguments=_string(parent, "arguments", path),
    )


def _checkpoint_state(value: Mapping[str, Any], path: Path) -> CheckpointState:
    raw = _object(value, "checkpoints", path)
    return CheckpointState(
        count=_integer(raw, "count", path),
        latest_completed=_optional_integer(raw, "latest_completed", path),
        latest_restorable=_optional_integer(raw, "latest_restorable", path),
        resume_available=_boolean(raw, "resume_available", path),
        unavailable_code=_optional_string(raw, "unavailable_code", path),
        unavailable_reason=_optional_string(raw, "unavailable_reason", path),
    )


def _session_state(value: Mapping[str, Any], path: Path) -> SessionState:
    raw = _object(value, "sessions", path)
    issues_value = raw.get("issues")
    if not isinstance(issues_value, list):
        raise RunStoreError(
            f"run metadata has invalid session issues: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": "sessions.issues"},
        )
    issues: list[SessionIssue] = []
    for item in cast(list[object], issues_value):
        if not isinstance(item, dict):
            raise RunStoreError(
                f"run metadata has invalid session issues: {path}",
                code="run.manifest_invalid",
                details={"path": str(path), "field": "sessions.issues"},
            )
        issue = cast(dict[str, Any], item)
        issues.append(
            SessionIssue(
                address=_string(issue, "address", path),
                provider=_string(issue, "provider", path),
                code=_string(issue, "code", path),
                message=_string(issue, "message", path),
            )
        )
    return SessionState(
        persistent=_integer(raw, "persistent", path),
        model=_string(raw, "model", path),
        branch_available=_boolean(raw, "branch_available", path),
        issues=tuple(issues),
    )


def load_run(output_dir: Path) -> LoadedRun:
    output = output_dir.resolve()
    path = existing_control_directory(output) / RUN_MANIFEST
    value, signature = _read_manifest(path, code="run.manifest_unreadable")
    version = _version(
        value, path, supported=(SCHEMA_VERSION, RUN_MANIFEST_SCHEMA_VERSION)
    )
    try:
        status = RunStatus(_string(value, "status", path))
    except ValueError as error:
        raise RunStoreError(
            f"run metadata has an invalid status: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": "status"},
        ) from error
    compatibility_value = value.get("compatibility", {})
    if not isinstance(compatibility_value, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in cast(dict[object, object], compatibility_value).items()
    ):
        raise RunStoreError(
            f"run metadata has invalid compatibility data: {path}",
            code="run.manifest_invalid",
            details={"path": str(path), "field": "compatibility"},
        )
    launch = _launch(value, path)
    legacy_sessions = SessionState()
    if version == SCHEMA_VERSION:
        # Validate the old shape, but never trust its cached checkpoint state.
        # Its session cache is needed only because v1 checkpoints did not yet
        # carry their own session summary.
        _checkpoint_state(value, path)
        legacy_sessions = _session_state(value, path)
    manifest = RunManifest(
        id=_string(value, "id", path),
        project_root=_string(value, "project_root", path),
        directory_name=(
            _string(value, "directory_name", path)
            if version == SCHEMA_VERSION
            else output.name
        ),
        workflow=_workflow(value, path),
        status=status,
        started_at=_string(value, "started_at", path),
        updated_at=_string(value, "updated_at", path),
        output_dir=_string(value, "output_dir", path),
        launch=launch,
        parent=_parent(value, path),
        compatibility=cast(dict[str, str], compatibility_value),
        storage_schema_version=version,
    )
    if Path(manifest.output_dir).resolve() != output:
        raise RunStoreError(
            f"run metadata names a different output directory: {path}",
            code="run.output_mismatch",
            details={"path": str(path), "output_dir": manifest.output_dir},
        )
    try:
        index = loaded_checkpoint_index(output)
    except RunStoreError as error:
        details: dict[str, object] = {}
        raw_details: object = error.details
        if isinstance(raw_details, dict):
            details.update(cast(dict[str, object], raw_details))
        elif raw_details is not None:
            details["cause_details"] = raw_details
        details.update(
            {
                "run_id": manifest.id,
                "directory_name": manifest.directory_name,
                "output_dir": str(output),
            }
        )
        raise RunStoreError(
            str(error), code=error.code, details=details
        ) from error
    checkpoints = index.checkpoints
    summaries = tuple(item.summary for item in checkpoints)
    sessions = (
        checkpoints[-1].sessions
        if checkpoints and checkpoints[-1].sessions is not None
        else legacy_sessions
    )
    updated_at = latest_timestamp(
        (manifest.updated_at, *(item.summary.created_at for item in checkpoints))
    )
    return LoadedRun(
        manifest=replace(
            manifest,
            updated_at=updated_at,
            checkpoints=checkpoint_state(summaries, launch.checkpointing),
            sessions=sessions,
        ),
        checkpoints=checkpoints,
        manifest_signature=signature,
        checkpoint_directory_signature=index.directory_signature,
    )


def load_run_manifest(output_dir: Path) -> RunManifest:
    """Load one run with state derived from committed checkpoint directories."""

    return load_run(output_dir).manifest


def _boundary(value: object, path: Path, key: str) -> Boundary | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RunStoreError(
            f"checkpoint metadata has an invalid {key}: {path}",
            code="checkpoint.manifest_invalid",
            details={"path": str(path), "field": key},
        )
    raw = cast(dict[str, Any], value)
    visit = _integer(raw, "visit", path)
    if visit <= 0:
        raise RunStoreError(
            f"checkpoint metadata has an invalid {key}: {path}",
            code="checkpoint.manifest_invalid",
            details={"path": str(path), "field": f"{key}.visit"},
        )
    return Boundary(
        project_path=_string(raw, "project_path", path),
        graph=_string(raw, "graph", path),
        node=_string(raw, "node", path),
        visit=visit,
        call_path=_string(raw, "call_path", path),
    )


def validate_checkpoint_availability(
    summary: CheckpointSummary,
    path: Path,
    /,
) -> None:
    invalid = (
        (not summary.restore_available and summary.fork_with_branch_available)
        or (not summary.restore_available and summary.fork_with_fresh_available)
        or (
            summary.fork_with_branch_available and not summary.fork_with_fresh_available
        )
    )
    if invalid:
        raise RunStoreError(
            f"checkpoint availability flags are inconsistent: {path}",
            code="checkpoint.manifest_invalid",
            details={"path": str(path), "sequence": summary.sequence},
        )


def _checkpoint_summary(
    value: Mapping[str, Any], path: Path, /
) -> CheckpointSummary:
    sequence = _integer(value, "sequence", path)
    if sequence <= 0:
        raise RunStoreError(
            f"checkpoint sequence must be positive: {path}",
            code="checkpoint.manifest_invalid",
            details={"path": str(path), "field": "sequence"},
        )
    try:
        kind = CheckpointKind(_string(value, "kind", path))
    except ValueError as error:
        raise RunStoreError(
            f"checkpoint metadata has an invalid kind: {path}",
            code="checkpoint.manifest_invalid",
            details={"path": str(path), "field": "kind"},
        ) from error
    restore_available = _boolean(value, "restore_available", path)
    fresh_available = _boolean(value, "fork_with_fresh_available", path)

    summary = CheckpointSummary(
        sequence=sequence,
        created_at=_string(value, "created_at", path),
        kind=kind,
        completed=_boundary(value.get("completed"), path, "completed"),
        next=_boundary(value.get("next"), path, "next"),
        restore_available=restore_available,
        fork_with_branch_available=_boolean(value, "fork_with_branch_available", path),
        fork_with_fresh_available=fresh_available,
        unavailable_code=_optional_string(value, "unavailable_code", path),
        unavailable_reason=_optional_string(value, "unavailable_reason", path),
    )
    validate_checkpoint_availability(summary, path)
    return summary


def _checkpoint_shard_name(name: str, path: Path, /) -> str:
    if strict_posix_relative_parts(name) is None:
        raise RunStoreError(
            f"unsafe checkpoint shard name: {name or '<empty>'}",
            code="checkpoint.shard_invalid",
            details={"path": str(path), "name": name},
        )
    return name


def _checkpoint_shards(
    value: Mapping[str, Any], path: Path, /
) -> Mapping[str, CheckpointShard]:
    raw_records = value.get("shards")
    if not isinstance(raw_records, list):
        raise RunStoreError(
            f"checkpoint shard manifest is invalid: {path}",
            code="checkpoint.manifest_invalid",
            details={"path": str(path), "field": "shards"},
        )
    found: dict[str, CheckpointShard] = {}
    for raw_record in cast(list[object], raw_records):
        if not isinstance(raw_record, dict):
            raise RunStoreError(
                f"checkpoint shard manifest is invalid: {path}",
                code="checkpoint.manifest_invalid",
                details={"path": str(path), "field": "shards"},
            )
        record = cast(dict[str, Any], raw_record)
        raw_name = record.get("name")
        size = record.get("size")
        digest = record.get("sha256")
        if (
            not isinstance(raw_name, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
        ):
            raise RunStoreError(
                f"checkpoint shard manifest is invalid: {path}",
                code="checkpoint.manifest_invalid",
                details={"path": str(path), "field": "shards"},
            )
        name = _checkpoint_shard_name(raw_name, path)
        if name in found:
            raise RunStoreError(
                f"checkpoint shard failed its integrity check: {name}",
                code="checkpoint.shard_corrupt",
                details={"path": str(path), "name": name},
            )
        found[name] = CheckpointShard(name, size, digest)
    return MappingProxyType(found)


def _checkpoint_sessions(
    value: Mapping[str, Any], path: Path, /
) -> SessionState:
    raw = value.get("sessions")
    try:
        return _session_state({"sessions": raw}, path)
    except RunStoreError as error:
        raise RunStoreError(
            f"checkpoint metadata has invalid session state: {path}",
            code="checkpoint.manifest_invalid",
            details={"path": str(path), "field": "sessions"},
        ) from error


def _checkpoint_artifacts(
    value: Mapping[str, Any], path: Path, /
) -> ArtifactReferences | None:
    # Defer this import because the artifact module shares our fsync primitive.
    from ._artifact_references import decode_artifact_references

    return decode_artifact_references(value.get("artifacts"), path)


def validate_checkpoint_sessions(
    summary: CheckpointSummary,
    sessions: SessionState | None,
    path: Path,
    /,
) -> None:
    if (
        sessions is not None
        and summary.fork_with_branch_available
        and not sessions.branch_available
    ):
        raise RunStoreError(
            f"checkpoint session availability is inconsistent: {path}",
            code="checkpoint.manifest_invalid",
            details={"path": str(path), "field": "sessions.branch_available"},
        )


def load_checkpoint(path: Path, /) -> LoadedCheckpoint:
    value, signature = _read_manifest(path, code="checkpoint.manifest_unreadable")
    _version(value, path, supported=(CHECKPOINT_MANIFEST_SCHEMA_VERSION,))
    summary = _checkpoint_summary(value, path)
    sessions = _checkpoint_sessions(value, path)
    validate_checkpoint_sessions(summary, sessions, path)
    return LoadedCheckpoint(
        summary=summary,
        shards=_checkpoint_shards(value, path),
        artifacts=_checkpoint_artifacts(value, path),
        sessions=sessions,
        manifest_path=path,
        manifest_signature=signature,
    )


def checkpoint_directory_signature(output_dir: Path) -> FileSignature | None:
    root = existing_control_directory(output_dir) / CHECKPOINT_DIRECTORY
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RunStoreError(
            f"checkpoint directory is invalid: {root}",
            code="checkpoint.directory_invalid",
            details={"path": str(root)},
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise RunStoreError(
            f"checkpoint directory is invalid: {root}",
            code="checkpoint.directory_invalid",
            details={"path": str(root)},
        )
    return file_signature(metadata)


def loaded_checkpoint_index(output_dir: Path) -> LoadedCheckpointIndex:
    root = existing_control_directory(output_dir) / CHECKPOINT_DIRECTORY
    for attempt in range(2):
        before = checkpoint_directory_signature(output_dir)
        if before is None:
            if checkpoint_directory_signature(output_dir) is None:
                return LoadedCheckpointIndex((), None)
            continue
        try:
            directories = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError as error:
            if attempt == 0:
                continue
            raise RunStoreError(
                f"checkpoint directory is invalid: {root}",
                code="checkpoint.directory_invalid",
                details={"path": str(root)},
            ) from error
        found: list[LoadedCheckpoint] = []
        for directory in directories:
            manifest_path = directory / "manifest.json"
            if not directory.name.isdecimal():
                if (
                    not directory.is_dir()
                    or directory.is_symlink()
                    or not manifest_path.exists()
                ):
                    continue
            # Numbered directories are committed atomically from staging. An
            # incomplete or unsafe entry is corruption, not an older boundary.
            try:
                if not stat.S_ISDIR(directory.lstat().st_mode):
                    raise OSError("not a checkpoint directory")
            except OSError as error:
                raise RunStoreError(
                    f"checkpoint directory is invalid: {directory}",
                    code="checkpoint.directory_invalid",
                    details={"path": str(directory)},
                ) from error
            checkpoint = load_checkpoint(manifest_path)
            if directory.name != f"{checkpoint.summary.sequence:06d}":
                raise RunStoreError(
                    f"checkpoint directory does not match its sequence: {directory}",
                    code="checkpoint.sequence_mismatch",
                    details={
                        "path": str(directory),
                        "sequence": checkpoint.summary.sequence,
                    },
                )
            found.append(checkpoint)
        after = checkpoint_directory_signature(output_dir)
        if after != before:
            continue
        ordered = tuple(sorted(found, key=lambda item: item.summary.sequence))
        sequences = [item.summary.sequence for item in ordered]
        if len(sequences) != len(set(sequences)):
            raise RunStoreError(
                f"duplicate checkpoint sequences in {root}",
                code="checkpoint.sequence_duplicate",
                details={"path": str(root)},
            )
        if sequences != list(range(1, len(sequences) + 1)):
            raise RunStoreError(
                f"checkpoint sequence has a gap in {root}",
                code="checkpoint.sequence_gap",
                details={"path": str(root), "sequences": sequences},
            )
        return LoadedCheckpointIndex(ordered, after)
    raise RunStoreError(
        f"checkpoint directory changed while it was read: {root}",
        code="checkpoint.directory_invalid",
        details={"path": str(root)},
    )


def load_checkpoint_summary(path: Path) -> CheckpointSummary:
    """Load one committed checkpoint summary."""

    return load_checkpoint(path).summary


def checkpoint_summaries(output_dir: Path) -> tuple[CheckpointSummary, ...]:
    return tuple(item.summary for item in loaded_checkpoint_index(output_dir).checkpoints)


def checkpoint_state(
    summaries: Sequence[CheckpointSummary],
    checkpointing: CheckpointPolicy,
    /,
) -> CheckpointState:
    """Derive all run-level checkpoint availability from committed summaries."""

    if not summaries:
        disabled = checkpointing is CheckpointPolicy.OFF
        return CheckpointState(
            unavailable_code="checkpoint.disabled" if disabled else "checkpoint.none",
            unavailable_reason=(
                "checkpointing is disabled"
                if disabled
                else "this run has no committed checkpoint"
            ),
        )
    latest = summaries[-1]
    latest_restorable = next(
        (item.sequence for item in reversed(summaries) if item.restore_available),
        None,
    )
    resume_available = latest.restore_available and latest.fork_with_branch_available
    unavailable_code = latest.unavailable_code
    unavailable_reason = latest.unavailable_reason
    if latest.restore_available and not latest.fork_with_branch_available:
        unavailable_code = "checkpoint.session_branch_unavailable"
        unavailable_reason = (
            "one or more persistent provider conversations cannot be restored exactly"
        )
    return CheckpointState(
        count=len(summaries),
        latest_completed=latest.sequence,
        latest_restorable=latest_restorable,
        resume_available=resume_available,
        unavailable_code=None if resume_available else unavailable_code,
        unavailable_reason=None if resume_available else unavailable_reason,
    )


@contextmanager
def exclusive_file(path: Path, *, blocking: bool) -> Generator[object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RunStoreError(
            f"lock path is unsafe: {path}",
            code="run.lock_invalid",
            details={"path": str(path)},
        )
    with path.open("a+b") as stream:
        with locked_file(stream, blocking=blocking):
            yield stream


def run_is_active(output_dir: Path) -> bool:
    """Whether a cooperating runtime currently owns this run's lease."""

    try:
        lock = existing_control_directory(output_dir) / "lock"
    except RunStoreError:
        return False
    if not lock.is_file() or lock.is_symlink():
        return False
    try:
        with exclusive_file(lock, blocking=False):
            return False
    except LockUnavailable:
        return True


def _registry_entries(project_root: Path) -> tuple[tuple[str, str], ...]:
    project = project_root.resolve()
    control = project / REGISTRY_PATH.parts[0]
    if control.is_symlink() or (control.exists() and not control.is_dir()):
        raise RunStoreError(
            f"run registry directory is unsafe: {control}",
            code="run.registry_invalid",
            details={"path": str(control)},
        )
    path = project / REGISTRY_PATH
    if not path.exists():
        return ()
    value = _read_object(path, code="run.registry_unreadable")
    _version(value, path)
    raw = value.get("runs")
    if not isinstance(raw, list):
        raise RunStoreError(
            f"run registry is invalid: {path}",
            code="run.registry_invalid",
            details={"path": str(path)},
        )
    found: list[tuple[str, str]] = []
    for item in cast(list[object], raw):
        if not isinstance(item, dict):
            raise RunStoreError(
                f"run registry is invalid: {path}",
                code="run.registry_invalid",
                details={"path": str(path)},
            )
        entry = cast(dict[str, Any], item)
        found.append(
            (_string(entry, "run_id", path), _string(entry, "output_dir", path))
        )
    return tuple(found)


def registered_runs(project_root: Path) -> tuple[Path, ...]:
    """All custom/default output roots registered by this project."""

    return tuple(
        Path(output).resolve() for _, output in _registry_entries(project_root)
    )


def register_run(project_root: Path, output_dir: Path, run_id: str) -> None:
    """Atomically make a custom output discoverable from its project."""

    project = project_root.resolve()
    output = output_dir.resolve()
    control = project / REGISTRY_PATH.parts[0]
    if control.is_symlink() or (control.exists() and not control.is_dir()):
        raise RunStoreError(
            f"run registry directory is unsafe: {control}",
            code="run.registry_invalid",
            details={"path": str(control)},
        )
    control.mkdir(parents=True, exist_ok=True)
    locks = (project / REGISTRY_LOCK).parent
    if locks.is_symlink() or (locks.exists() and not locks.is_dir()):
        raise RunStoreError(
            f"run registry lock directory is unsafe: {locks}",
            code="run.registry_invalid",
            details={"path": str(locks)},
        )
    locks.mkdir(parents=True, exist_ok=True)
    lock = project / REGISTRY_LOCK
    with exclusive_file(lock, blocking=True):
        entries = list(_registry_entries(project))
        entries = [entry for entry in entries if Path(entry[1]).resolve() != output]
        entries.append((run_id, str(output)))
        entries.sort(key=lambda entry: (entry[1], entry[0]))
        atomic_json(
            project / REGISTRY_PATH,
            {
                "schema_version": SCHEMA_VERSION,
                "runs": [
                    {"run_id": identifier, "output_dir": path}
                    for identifier, path in entries
                ],
            },
        )
