"""Durable metadata for locally executed workflows.

The run store deliberately contains no interpreter state.  It is the small,
versioned index around checkpoint shards: the runtime writes it, while the CLI
can inspect it without importing the workflow which produced it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import pathlib
import stat
import tempfile
import types
import uuid
from collections.abc import Generator, Mapping, Sequence

from verdog_runtime._artifact_references import (
    ArtifactCache,
    ArtifactReferences,
    decode_artifact_references,
    materialize_artifact_references,
    validate_artifact_extension,
)
from verdog_runtime._artifact_references import (
    capture_artifacts as _capture_artifacts,
)
from verdog_runtime._artifact_references import (
    validate_artifacts as _validate_artifacts,
)
from verdog_runtime._file_lock import FileLockUnavailable as _LockUnavailable
from verdog_runtime._relative_path import strict_posix_relative_parts
from verdog_runtime._run_metadata import (
    FileSignature,
    LoadedCheckpoint,
    checkpoint_directory_signature,
    checkpoint_state,
    checkpoint_summaries,
    file_signature,
    latest_timestamp,
    load_checkpoint,
    load_checkpoint_summary,
    load_run,
    load_run_manifest,
    loaded_checkpoint_index,
    register_run,
    registered_runs,
    run_is_active,
)
from verdog_runtime._run_metadata import (
    atomic_json as _atomic_json,
)
from verdog_runtime._run_metadata import (
    exclusive_file as _exclusive_file,
)
from verdog_runtime._run_metadata import (
    existing_control_directory as _existing_control_directory,
)
from verdog_runtime._run_metadata import (
    fsync_directory as _fsync_directory,
)
from verdog_runtime._run_metadata import (
    remove_private_tree as _remove_private_tree,
)
from verdog_runtime._run_metadata import (
    validate_checkpoint_availability as _validate_checkpoint_availability,
)
from verdog_runtime._run_metadata import (
    validate_checkpoint_sessions as _validate_checkpoint_sessions,
)
from verdog_runtime._run_model import (
    CHECKPOINT_DIRECTORY,
    CONTROL_DIRECTORY,
    RUN_MANIFEST,
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
    utc_now,
)
from verdog_runtime._run_model import (
    EMPTY_COMPATIBILITY as _EMPTY_COMPATIBILITY,
)
from verdog_runtime._run_model import (
    EMPTY_SHARDS as _EMPTY_SHARDS,
)

_EMPTY_SESSIONS = SessionState()


def _shard_relative(name: str) -> pathlib.Path:
    parts = strict_posix_relative_parts(name)
    if parts is None:
        raise RunStoreError(
            f"unsafe checkpoint shard name: {name or '<empty>'}",
            code="checkpoint.shard_invalid",
            details={"name": name},
        )
    return pathlib.Path(*parts)


def _write_binary(path: pathlib.Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def _shard_bytes(value: object, name: str, /) -> bytes:
    if not isinstance(value, bytes):
        raise RunStoreError(
            f"checkpoint shard is not bytes: {name}",
            code="checkpoint.shard_invalid",
            details={"name": name},
        )
    return value


def _write_checkpoint_stage(
    staged: pathlib.Path,
    summary: CheckpointSummary,
    shards: Mapping[str, bytes],
    *,
    sessions: SessionState,
    artifact_references: Mapping[str, object] | None,
) -> None:
    shard_root = staged / "shards"
    shard_root.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    for name, value in sorted(shards.items()):
        raw_value = _shard_bytes(value, name)
        _write_binary(shard_root / _shard_relative(name), raw_value)
        records.append(
            {
                "name": name,
                "size": len(raw_value),
                "sha256": hashlib.sha256(raw_value).hexdigest(),
            }
        )
    # The manifest is written last inside staging; the directory rename is the
    # single point at which readers can observe the checkpoint.
    _atomic_json(
        staged / "manifest.json",
        {
            **summary.as_json(),
            "shards": records,
            "artifacts": artifact_references,
            "sessions": sessions.as_json(),
        },
    )
    directories = {shard_root}
    directories.update(
        path.parent for path in shard_root.rglob("*") if path.is_file()
    )
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        _fsync_directory(directory)
    _fsync_directory(staged)


class RunStore:
    """The mutable, atomic metadata wrapper for one local run."""

    def __init__(self, output_dir: pathlib.Path) -> None:
        self.output_dir = output_dir.resolve()
        self.control_dir = self.output_dir / CONTROL_DIRECTORY
        self._checkpoint_cache: dict[int, LoadedCheckpoint] = {}
        self._artifact_cache: ArtifactCache = {}
        self._manifest_cache: RunManifest | None = None
        self._run_manifest_signature: FileSignature | None = None
        self._checkpoint_root_signature: FileSignature | None = None
        self._checkpoint_index_complete = False

    @classmethod
    def create(
        cls,
        output_dir: pathlib.Path,
        *,
        project_root: pathlib.Path,
        workflow_id: str,
        definition_id: str,
        module: str,
        workflow_arguments: Sequence[str] = (),
        checkpointing: CheckpointPolicy = CheckpointPolicy.AUTO,
        run_id: str | None = None,
        parent: ParentRun | None = None,
        compatibility: Mapping[str, str] = _EMPTY_COMPATIBILITY,
        started_at: str | None = None,
    ) -> RunStore:
        output = output_dir.resolve()
        if not output.is_dir():
            raise RunStoreError(
                f"run output directory does not exist: {output}",
                code="run.output_missing",
                details={"output_dir": str(output)},
            )
        control = output / CONTROL_DIRECTORY
        if control.exists() or control.is_symlink():
            raise RunStoreError(
                f"run metadata already exists: {control}",
                code="run.already_initialized",
                details={"output_dir": str(output)},
            )
        control.mkdir(mode=0o700)
        (control / CHECKPOINT_DIRECTORY).mkdir(mode=0o700)
        (control / "staging").mkdir(mode=0o700)
        identifier = run_id or str(uuid.uuid4())
        timestamp = started_at or utc_now()
        policy = CheckpointPolicy(checkpointing)
        manifest = RunManifest(
            id=identifier,
            project_root=str(project_root.resolve()),
            directory_name=output.name,
            workflow=WorkflowIdentity(workflow_id, definition_id, module),
            status=RunStatus.RUNNING,
            started_at=timestamp,
            updated_at=timestamp,
            output_dir=str(output),
            launch=LaunchRecord(tuple(workflow_arguments), policy),
            parent=parent,
            checkpoints=checkpoint_state((), policy),
            compatibility=dict(compatibility),
        )
        store = cls(output)
        store.save(manifest)
        register_run(project_root, output, identifier)
        return store

    @classmethod
    def open(cls, output_dir: pathlib.Path) -> RunStore:
        store = cls(output_dir)
        store.manifest()
        return store

    def _manifest_cache_is_current(self) -> bool:
        if self._manifest_cache is None or self._run_manifest_signature is None:
            return False
        try:
            run_signature = file_signature(
                (self.control_dir / RUN_MANIFEST).lstat()
            )
        except OSError:
            return False
        if run_signature != self._run_manifest_signature:
            return False
        return self._checkpoint_cache_is_current()

    def _checkpoint_cache_is_current(self) -> bool:
        if (
            not self._checkpoint_index_complete
            or checkpoint_directory_signature(self.output_dir)
            != self._checkpoint_root_signature
        ):
            return False
        for checkpoint in self._checkpoint_cache.values():
            try:
                signature = file_signature(checkpoint.manifest_path.lstat())
            except OSError:
                return False
            if signature != checkpoint.manifest_signature:
                return False
        return True

    def manifest(self) -> RunManifest:
        if (
            self._manifest_cache is not None
            and self._manifest_cache_is_current()
        ):
            return self._manifest_cache
        loaded = load_run(self.output_dir)
        self._checkpoint_cache = {
            item.summary.sequence: item for item in loaded.checkpoints
        }
        self._manifest_cache = loaded.manifest
        self._run_manifest_signature = loaded.manifest_signature
        self._checkpoint_root_signature = loaded.checkpoint_directory_signature
        self._checkpoint_index_complete = True
        return loaded.manifest

    def save(self, manifest: RunManifest) -> None:
        if pathlib.Path(manifest.output_dir).resolve() != self.output_dir:
            raise RunStoreError(
                "cannot write a manifest for another output directory",
                code="run.output_mismatch",
                details={"output_dir": manifest.output_dir},
            )
        _atomic_json(self.control_dir / RUN_MANIFEST, manifest.as_json())
        self._manifest_cache = None
        self._run_manifest_signature = None

    def update(
        self,
        *,
        status: RunStatus | None = None,
    ) -> RunManifest:
        manifest = self.manifest()
        updated = dataclasses.replace(
            manifest,
            status=manifest.status if status is None else RunStatus(status),
            updated_at=utc_now(),
        )
        self.save(updated)
        return updated

    @contextlib.contextmanager
    def lease(self) -> Generator[None]:
        try:
            with _exclusive_file(self.control_dir / "lock", blocking=False):
                yield
        except _LockUnavailable as error:
            raise RunStoreError(
                f"run is already active: {self.output_dir}",
                code="run.active",
                details={"output_dir": str(self.output_dir)},
            ) from error

    def checkpoint_directory(self, sequence: int) -> pathlib.Path:
        if sequence <= 0:
            raise ValueError("checkpoint sequence must be positive")
        return self.control_dir / CHECKPOINT_DIRECTORY / f"{sequence:06d}"

    def _loaded_checkpoint(self, sequence: int) -> LoadedCheckpoint:
        path = self.checkpoint_directory(sequence) / "manifest.json"
        cached = self._checkpoint_cache.get(sequence)
        if cached is not None:
            try:
                signature = file_signature(path.lstat())
            except OSError:
                signature = None
            if signature == cached.manifest_signature:
                return cached
        loaded = load_checkpoint(path)
        if loaded.summary.sequence != sequence:
            raise RunStoreError(
                (
                    f"checkpoint directory does not match its sequence: "
                    f"{path.parent}"
                ),
                code="checkpoint.sequence_mismatch",
                details={
                    "path": str(path.parent),
                    "sequence": loaded.summary.sequence,
                },
            )
        if cached is None:
            self._checkpoint_index_complete = False
        self._checkpoint_cache[sequence] = loaded
        self._manifest_cache = None
        return loaded

    def _all_checkpoints(self) -> tuple[LoadedCheckpoint, ...]:
        if not self._checkpoint_cache_is_current():
            index = loaded_checkpoint_index(self.output_dir)
            checkpoints = index.checkpoints
            self._checkpoint_cache = {
                item.summary.sequence: item for item in checkpoints
            }
            self._checkpoint_root_signature = index.directory_signature
            self._checkpoint_index_complete = True
            self._manifest_cache = None
            return checkpoints
        return tuple(
            self._checkpoint_cache[sequence]
            for sequence in sorted(self._checkpoint_cache)
        )

    def next_checkpoint_sequence(self) -> int:
        existing = self._all_checkpoints()
        return 1 if not existing else existing[-1].summary.sequence + 1

    def commit_checkpoint(
        self,
        summary: CheckpointSummary,
        *,
        shards: Mapping[str, bytes] = _EMPTY_SHARDS,
        sessions: SessionState = _EMPTY_SESSIONS,
        capture_artifacts: bool = False,
        artifact_references: Mapping[str, object] | None = None,
    ) -> RunManifest:
        _validate_checkpoint_availability(
            summary,
            self.checkpoint_directory(summary.sequence) / "manifest.json",
        )
        _validate_checkpoint_sessions(
            summary,
            sessions,
            self.checkpoint_directory(summary.sequence) / "manifest.json",
        )
        manifest = self.manifest()
        directory = self.checkpoint_directory(summary.sequence)
        if directory.exists():
            raise RunStoreError(
                f"checkpoint is already committed: {summary.sequence}",
                code="checkpoint.already_committed",
                details={"sequence": summary.sequence},
            )
        expected = self.next_checkpoint_sequence()
        if summary.sequence != expected:
            raise RunStoreError(
                f"checkpoint sequence {summary.sequence} is not the next "
                f"sequence {expected}",
                code="checkpoint.sequence_invalid",
                details={"sequence": summary.sequence, "expected": expected},
            )
        checkpoint_root = directory.parent
        checkpoint_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging_root = self.control_dir / "staging"
        staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        staged = pathlib.Path(
            tempfile.mkdtemp(
                prefix=f"checkpoint-{summary.sequence:06d}-", dir=staging_root
            )
        )
        committed = False
        try:
            if capture_artifacts and artifact_references is not None:
                raise ValueError(
                    "capture_artifacts and artifact_references are "
                    "mutually exclusive"
                )
            if capture_artifacts:
                artifact_references = self.capture_artifacts()
            if artifact_references is not None:
                # A child captures at its boundary. Do not rescan its output
                # after it has continued executing in the other process.
                references = decode_artifact_references(
                    dict(artifact_references), staged / "manifest.json"
                )
                assert references is not None
                validate_artifact_extension(
                    self._latest_artifacts(), references
                )
            _write_checkpoint_stage(
                staged,
                summary,
                shards,
                sessions=sessions,
                artifact_references=artifact_references,
            )
            staged.replace(directory)
            committed = True
            _fsync_directory(checkpoint_root)
        finally:
            if not committed:
                _remove_private_tree(staged)
        loaded = load_checkpoint(directory / "manifest.json")
        self._checkpoint_cache[summary.sequence] = loaded
        self._checkpoint_root_signature = checkpoint_directory_signature(
            self.output_dir
        )
        self._checkpoint_index_complete = True
        summaries = tuple(
            self._checkpoint_cache[sequence].summary
            for sequence in sorted(self._checkpoint_cache)
        )
        committed_sessions = loaded.sessions
        if committed_sessions is None:
            raise AssertionError("a checkpoint must contain session state")
        updated = dataclasses.replace(
            manifest,
            checkpoints=checkpoint_state(
                summaries, manifest.launch.checkpointing
            ),
            sessions=committed_sessions,
            updated_at=latest_timestamp(
                (manifest.updated_at, loaded.summary.created_at)
            ),
        )
        self._manifest_cache = updated
        return updated

    def checkpoint_shard_path(self, sequence: int, name: str) -> pathlib.Path:
        control = _existing_control_directory(self.output_dir)
        shard_root = (
            control / CHECKPOINT_DIRECTORY / f"{sequence:06d}" / "shards"
        )
        relative = _shard_relative(name)
        target = shard_root / relative
        try:
            root = shard_root.resolve(strict=True)
            resolved = target.resolve(strict=True)
            mode = target.lstat().st_mode
        except OSError as error:
            raise RunStoreError(
                f"checkpoint shard is unavailable: {sequence}/{name}",
                code="checkpoint.shard_unavailable",
                details={"sequence": sequence, "name": name},
            ) from error
        if (
            shard_root.is_symlink()
            or target.is_symlink()
            or not stat.S_ISREG(mode)
            or not resolved.is_relative_to(root)
            or resolved != root.joinpath(*relative.parts)
        ):
            raise RunStoreError(
                f"checkpoint shard is unsafe: {sequence}/{name}",
                code="checkpoint.shard_invalid",
                details={"sequence": sequence, "name": name},
            )
        return resolved

    def _checkpoint_shard(
        self,
        checkpoint: LoadedCheckpoint,
        name: str,
        /,
        *,
        path: pathlib.Path | None = None,
    ) -> bytes:
        sequence = checkpoint.summary.sequence
        try:
            target = path or self.checkpoint_shard_path(sequence, name)
            value = target.read_bytes()
        except OSError as error:
            raise RunStoreError(
                f"checkpoint shard is unreadable: {sequence}/{name}",
                code="checkpoint.shard_unavailable",
                details={"sequence": sequence, "name": name},
            ) from error
        integrity = checkpoint.shards.get(name)
        if (
            integrity is None
            or integrity.size != len(value)
            or integrity.sha256 != hashlib.sha256(value).hexdigest()
        ):
            raise RunStoreError(
                (
                    f"checkpoint shard failed its integrity check: "
                    f"{sequence}/{name}"
                ),
                code="checkpoint.shard_corrupt",
                details={"sequence": sequence, "name": name},
            )
        return value

    def checkpoint_shard(self, sequence: int, name: str) -> bytes:
        path = self.checkpoint_shard_path(sequence, name)
        return self._checkpoint_shard(
            self._loaded_checkpoint(sequence), name, path=path
        )

    def checkpoint_shards(self, sequence: int) -> Mapping[str, bytes]:
        """Return every integrity-checked shard in one committed checkpoint."""
        checkpoint = self._loaded_checkpoint(sequence)
        return types.MappingProxyType(
            {
                name: self._checkpoint_shard(checkpoint, name)
                for name in checkpoint.shards
            }
        )

    def _latest_artifacts(self) -> ArtifactReferences | None:
        return next(
            (
                checkpoint.artifacts
                for checkpoint in reversed(self._all_checkpoints())
                if checkpoint.artifacts is not None
            ),
            None,
        )

    def capture_artifacts(self) -> dict[str, object]:
        """Capture references at the executing process's checkpoint boundary."""
        return _capture_artifacts(
            self.output_dir,
            cache=self._artifact_cache,
            previous=self._latest_artifacts(),
        )

    def validate_artifacts(self, sequence: int) -> None:
        """Verify the existing output files before resuming this checkpoint."""
        references = self._loaded_checkpoint(sequence).artifacts
        if references is None:
            raise RunStoreError(
                f"checkpoint has no artifact references: {sequence}",
                code="checkpoint.artifacts_unavailable",
                details={"sequence": sequence},
            )
        _validate_artifacts(
            self.output_dir, references, cache=self._artifact_cache
        )

    def artifact_references_available(self, sequence: int) -> bool:
        checkpoint = self.checkpoint_directory(sequence)
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            raise RunStoreError(
                f"checkpoint is unavailable: {sequence}",
                code="checkpoint.not_found",
                details={"sequence": sequence},
            )
        return self._loaded_checkpoint(sequence).artifacts is not None

    def materialize_artifacts(
        self, sequence: int, destination: pathlib.Path
    ) -> pathlib.Path:
        """Atomically restore checkpoint files into a new output."""
        checkpoint = self.checkpoint_directory(sequence)
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            raise RunStoreError(
                f"checkpoint is unavailable: {sequence}",
                code="checkpoint.not_found",
                details={"sequence": sequence},
            )
        references = self._loaded_checkpoint(sequence).artifacts
        if references is None:
            manifest_path = checkpoint / "manifest.json"
            raise RunStoreError(
                f"checkpoint has no valid artifact references: {manifest_path}",
                code="checkpoint.artifacts_unavailable",
                details={"path": str(manifest_path)},
            )
        requested = destination.absolute()
        if requested.is_symlink():
            raise RunStoreError(
                f"artifact destination is unsafe: {requested}",
                code="checkpoint.artifact_destination_invalid",
                details={"path": str(requested)},
            )
        target = requested.resolve(strict=False)
        if (
            target == self.output_dir
            or target.is_relative_to(self.output_dir)
            or self.output_dir.is_relative_to(target)
        ):
            raise RunStoreError(
                f"artifact destination overlaps its source run: {target}",
                code="checkpoint.artifact_destination_invalid",
                details={"path": str(target)},
            )
        existed = target.exists()
        original_mode = 0o700
        if existed:
            metadata = target.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or next(target.iterdir(), None) is not None
            ):
                raise RunStoreError(
                    f"artifact destination is not an empty directory: {target}",
                    code="checkpoint.artifact_destination_not_empty",
                    details={"path": str(target)},
                )
            original_mode = stat.S_IMODE(metadata.st_mode)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = pathlib.Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.artifacts-", dir=target.parent
            )
        )
        committed = False
        removed_empty_target = False
        try:
            materialize_artifact_references(
                self.output_dir, staged, *references
            )
            if existed:
                target.rmdir()
                removed_empty_target = True
            staged.replace(target)
            committed = True
            target.chmod(original_mode)
            _fsync_directory(target.parent)
        finally:
            if not committed:
                _remove_private_tree(staged)
                if removed_empty_target and not target.exists():
                    target.mkdir(mode=original_mode)
        return target

    def checkpoints(self) -> tuple[CheckpointSummary, ...]:
        return tuple(item.summary for item in self._all_checkpoints())


__all__ = [
    "Boundary",
    "CheckpointKind",
    "CheckpointPolicy",
    "CheckpointState",
    "CheckpointSummary",
    "CONTROL_DIRECTORY",
    "ParentRun",
    "RUN_MANIFEST",
    "RunManifest",
    "RunStatus",
    "RunStore",
    "RunStoreError",
    "SessionIssue",
    "SessionState",
    "WorkflowIdentity",
    "checkpoint_summaries",
    "load_checkpoint_summary",
    "load_run_manifest",
    "register_run",
    "registered_runs",
    "run_is_active",
    "utc_now",
]
