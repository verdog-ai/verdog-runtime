"""Reference immutable run artifacts without copying checkpoint files."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import stat
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

from verdog_runtime import _relative_path, _run_metadata, _run_model

_ARTIFACT_REFERENCES_VERSION = 1
_COPY_CHUNK_SIZE = 1024 * 1024
_FICLONE = 0x40049409
_INVOCATION_METADATA = ".verdog-invocation.json"


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactDirectory:
    relative: pathlib.Path
    mode: int


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactFile:
    relative: pathlib.Path
    mode: int
    size: int
    sha256: str


ArtifactReferences: TypeAlias = tuple[
    tuple[ArtifactDirectory, ...], tuple[ArtifactFile, ...]
]
ArtifactCache: TypeAlias = dict[
    pathlib.Path, tuple[_run_metadata.FileSignature, ArtifactFile]
]


def _artifact_relative(name: str) -> pathlib.Path:
    parts = _relative_path.strict_posix_relative_parts(name)
    if parts is None or _run_model.CONTROL_DIRECTORY in parts:
        raise _run_model.RunStoreError(
            f"unsafe checkpoint artifact path: {name or '<empty>'}",
            code="checkpoint.artifact_unsafe",
            details={"path": name},
        )
    return pathlib.Path(*parts)


def _artifact_mode(value: os.stat_result) -> int:
    return stat.S_IMODE(value.st_mode) & 0o777


def _integrity_error(relative: pathlib.Path) -> None:
    raise _run_model.RunStoreError(
        "checkpoint artifact failed its integrity check: "
        f"{relative.as_posix()}",
        code="checkpoint.artifact_corrupt",
        details={"path": relative.as_posix()},
    )


def _artifact_path(
    root: pathlib.Path, relative: pathlib.Path, *, directory: bool
) -> pathlib.Path:
    target = root.joinpath(*relative.parts)
    try:
        canonical_root = root.resolve(strict=True)
        resolved = target.resolve(strict=True)
        mode = target.lstat().st_mode
    except OSError as error:
        raise _run_model.RunStoreError(
            f"checkpoint artifact is unavailable: {relative.as_posix()}",
            code="checkpoint.artifact_unavailable",
            details={"path": relative.as_posix()},
        ) from error
    if (
        root.is_symlink()
        or target.is_symlink()
        or not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode))
        or resolved != canonical_root.joinpath(*relative.parts)
    ):
        raise _run_model.RunStoreError(
            f"checkpoint artifact is unsafe: {relative.as_posix()}",
            code="checkpoint.artifact_unsafe",
            details={"path": relative.as_posix()},
        )
    return resolved


def _artifact_file_path(
    root: pathlib.Path, relative: pathlib.Path
) -> pathlib.Path:
    return _artifact_path(root, relative, directory=False)


def _scan_artifacts(
    root: pathlib.Path,
) -> tuple[
    dict[pathlib.Path, _run_metadata.FileSignature],
    dict[pathlib.Path, _run_metadata.FileSignature],
]:
    _artifact_path(root, pathlib.Path(), directory=True)
    directories: dict[pathlib.Path, _run_metadata.FileSignature] = {}
    files: dict[pathlib.Path, _run_metadata.FileSignature] = {}
    pending = [pathlib.Path()]
    while pending:
        base = pending.pop()
        directory = root / base
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda item: item.name)
            report_directory = base == pathlib.Path() or any(
                entry.name == _INVOCATION_METADATA for entry in entries
            )
            for entry in entries:
                if entry.name == _run_model.CONTROL_DIRECTORY:
                    continue
                if report_directory and entry.name in ("config.md", "stats.md"):
                    continue
                if base == pathlib.Path() and entry.name == "trace.log":
                    continue
                relative = _artifact_relative((base / entry.name).as_posix())
                metadata = entry.stat(follow_symlinks=False)
                signature = _run_metadata.file_signature(metadata)
                if stat.S_ISDIR(metadata.st_mode):
                    directories[relative] = signature
                    pending.append(relative)
                elif stat.S_ISREG(metadata.st_mode):
                    files[relative] = signature
                else:
                    raise _run_model.RunStoreError(
                        (
                            f"artifact is not a regular file or directory: "
                            f"{root / relative}"
                        ),
                        code="checkpoint.artifact_unsafe",
                        details={"path": str(root / relative)},
                    )
        except OSError as error:
            raise _run_model.RunStoreError(
                f"artifact directory changed while checkpointing: {directory}",
                code="checkpoint.artifact_unstable",
                details={"path": str(directory)},
            ) from error
    return directories, files


def _read_artifact(
    root: pathlib.Path,
    relative: pathlib.Path,
    signature: _run_metadata.FileSignature,
    *,
    flush: bool,
) -> ArtifactFile:
    path = _artifact_file_path(root, relative)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if (
                _run_metadata.file_signature(os.fstat(stream.fileno()))
                != signature
            ):
                _integrity_error(relative)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if flush:
                os.fsync(stream.fileno())
            if (
                _run_metadata.file_signature(os.fstat(stream.fileno()))
                != signature
                or _run_metadata.file_signature(path.lstat()) != signature
            ):
                _integrity_error(relative)
    except OSError as error:
        raise _run_model.RunStoreError(
            f"checkpoint artifact is unreadable: {relative.as_posix()}",
            code="checkpoint.artifact_unavailable",
            details={"path": relative.as_posix()},
        ) from error
    return ArtifactFile(
        relative, stat.S_IMODE(signature[2]) & 0o777, signature[3], digest
    )


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("artifact write made no progress")
        offset += written


def _transfer_and_hash(source: int, target: int | None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(source, 0, os.SEEK_SET)
    while chunk := os.read(source, _COPY_CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
        if target is not None:
            _write_all(target, chunk)
    return size, digest.hexdigest()


def _reference_mode(value: Mapping[str, Any], path: pathlib.Path) -> int:
    mode = value.get("mode")
    if (
        not isinstance(mode, int)
        or isinstance(mode, bool)
        or mode < 0
        or mode > 0o777
    ):
        raise _run_model.RunStoreError(
            f"checkpoint artifact has an invalid mode: {path}",
            code="checkpoint.artifact_manifest_invalid",
            details={"path": str(path), "field": "mode"},
        )
    return mode


def _reference_string(
    value: Mapping[str, Any], key: str, path: pathlib.Path
) -> str:
    found = value.get(key)
    if not isinstance(found, str) or not found:
        raise _run_model.RunStoreError(
            f"checkpoint artifact has an invalid {key}: {path}",
            code="checkpoint.artifact_manifest_invalid",
            details={"path": str(path), "field": key},
        )
    return found


def _reference_size(value: Mapping[str, Any], path: pathlib.Path) -> int:
    found = value.get("size")
    if not isinstance(found, int) or isinstance(found, bool) or found < 0:
        raise _run_model.RunStoreError(
            f"checkpoint artifact has an invalid size: {path}",
            code="checkpoint.artifact_manifest_invalid",
            details={"path": str(path), "field": "size"},
        )
    return found


def _reference_records(
    value: Mapping[str, Any], key: str, path: pathlib.Path
) -> list[object]:
    records = value.get(key)
    if not isinstance(records, list):
        raise _run_model.RunStoreError(
            f"checkpoint artifact manifest has no valid {key}: {path}",
            code="checkpoint.artifact_manifest_invalid",
            details={"path": str(path), "field": key},
        )
    return cast(list[object], records)


def _reference_directories(
    value: Mapping[str, Any], path: pathlib.Path
) -> tuple[ArtifactDirectory, ...]:
    found: list[ArtifactDirectory] = []
    for item in _reference_records(value, "directories", path):
        if not isinstance(item, dict):
            raise _run_model.RunStoreError(
                f"checkpoint artifact directory is invalid: {path}",
                code="checkpoint.artifact_manifest_invalid",
                details={"path": str(path)},
            )
        record = cast(dict[str, Any], item)
        found.append(
            ArtifactDirectory(
                _artifact_relative(_reference_string(record, "path", path)),
                _reference_mode(record, path),
            )
        )
    return tuple(sorted(found, key=lambda item: item.relative.as_posix()))


def _reference_files(
    value: Mapping[str, Any], path: pathlib.Path
) -> tuple[ArtifactFile, ...]:
    found: list[ArtifactFile] = []
    for item in _reference_records(value, "files", path):
        if not isinstance(item, dict):
            raise _run_model.RunStoreError(
                f"checkpoint artifact file is invalid: {path}",
                code="checkpoint.artifact_manifest_invalid",
                details={"path": str(path)},
            )
        record = cast(dict[str, Any], item)
        digest = _reference_string(record, "sha256", path)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise _run_model.RunStoreError(
                f"checkpoint artifact digest is invalid: {path}",
                code="checkpoint.artifact_manifest_invalid",
                details={"path": str(path), "field": "sha256"},
            )
        found.append(
            ArtifactFile(
                _artifact_relative(_reference_string(record, "path", path)),
                _reference_mode(record, path),
                _reference_size(record, path),
                digest,
            )
        )
    return tuple(sorted(found, key=lambda item: item.relative.as_posix()))


def _validate_reference_layout(
    directories: Sequence[ArtifactDirectory],
    files: Sequence[ArtifactFile],
    manifest_path: pathlib.Path,
) -> None:
    directory_paths = {item.relative for item in directories}
    file_paths = {item.relative for item in files}
    if (
        len(directory_paths) != len(directories)
        or len(file_paths) != len(files)
        or directory_paths & file_paths
    ):
        raise _run_model.RunStoreError(
            f"checkpoint artifact paths are ambiguous: {manifest_path}",
            code="checkpoint.artifact_manifest_invalid",
            details={"path": str(manifest_path)},
        )
    missing_parent = next(
        (
            relative
            for relative in (*directory_paths, *file_paths)
            if relative.parent != pathlib.Path()
            and relative.parent not in directory_paths
        ),
        None,
    )
    if missing_parent is not None:
        raise _run_model.RunStoreError(
            (
                f"checkpoint artifact parent is missing: "
                f"{missing_parent.as_posix()}"
            ),
            code="checkpoint.artifact_manifest_invalid",
            details={"path": missing_parent.as_posix()},
        )


def decode_artifact_references(
    raw: object, manifest_path: pathlib.Path, /
) -> ArtifactReferences | None:
    """Decode artifact metadata already read from one checkpoint manifest."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _run_model.RunStoreError(
            f"checkpoint has no valid artifact references: {manifest_path}",
            code="checkpoint.artifact_manifest_invalid",
            details={"path": str(manifest_path)},
        )
    artifact = cast(dict[str, Any], raw)
    if (
        type(artifact.get("schema_version")) is not int
        or artifact.get("schema_version") != _ARTIFACT_REFERENCES_VERSION
        or artifact.get("kind") != "references"
    ):
        raise _run_model.RunStoreError(
            f"checkpoint has no valid artifact references: {manifest_path}",
            code="checkpoint.artifact_manifest_invalid",
            details={"path": str(manifest_path)},
        )
    directories = _reference_directories(artifact, manifest_path)
    files = _reference_files(artifact, manifest_path)
    _validate_reference_layout(directories, files, manifest_path)
    return directories, files


def validate_artifact_extension(
    previous: ArtifactReferences | None, current: ArtifactReferences
) -> None:
    """Validate immutable artifacts; directories may gain children."""
    if previous is None:
        return
    for old, new in zip(previous, current, strict=True):
        indexed = {item.relative: item for item in new}
        for item in old:
            if indexed.get(item.relative) != item:
                _integrity_error(item.relative)


def capture_artifacts(
    source_root: pathlib.Path,
    *,
    cache: ArtifactCache,
    previous: ArtifactReferences | None = None,
) -> dict[str, object]:
    """Flush and inventory outputs; retain verified hashes in ``cache``."""
    before_directories, before_files = _scan_artifacts(source_root)
    directories = tuple(
        ArtifactDirectory(relative, stat.S_IMODE(signature[2]) & 0o777)
        for relative, signature in sorted(before_directories.items())
    )
    files: list[ArtifactFile] = []
    changed_directories: set[pathlib.Path] = {pathlib.Path()}
    for relative, signature in sorted(before_files.items()):
        cached = cache.get(relative)
        if cached is not None and cached[0] == signature:
            record = cached[1]
        else:
            record = _read_artifact(
                source_root, relative, signature, flush=True
            )
            cache[relative] = (signature, record)
            changed_directories.add(relative.parent)
        files.append(record)
    current = directories, tuple(files)
    validate_artifact_extension(previous, current)
    if (before_directories, before_files) != _scan_artifacts(source_root):
        raise _run_model.RunStoreError(
            (
                "run artifacts changed while checkpoint references were "
                "being captured"
            ),
            code="checkpoint.artifact_unstable",
            details={"output_dir": str(source_root)},
        )
    previous_directories: set[pathlib.Path] = (
        set() if previous is None else {item.relative for item in previous[0]}
    )
    for record in directories:
        if record.relative not in previous_directories:
            changed_directories.update(
                (record.relative, record.relative.parent)
            )
    for relative in sorted(
        changed_directories, key=lambda item: len(item.parts), reverse=True
    ):
        _run_metadata.fsync_directory(source_root / relative)
    return {
        "schema_version": _ARTIFACT_REFERENCES_VERSION,
        "kind": "references",
        "directories": [
            {"path": item.relative.as_posix(), "mode": item.mode}
            for item in directories
        ],
        "files": [
            {
                "path": item.relative.as_posix(),
                "mode": item.mode,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in files
        ],
    }


def _validate_directories(
    root: pathlib.Path, directories: Sequence[ArtifactDirectory]
) -> None:
    _artifact_path(root, pathlib.Path(), directory=True)
    for record in directories:
        path = _artifact_path(root, record.relative, directory=True)
        if _artifact_mode(path.lstat()) != record.mode:
            _integrity_error(record.relative)


def validate_artifacts(
    source_root: pathlib.Path,
    references: ArtifactReferences,
    *,
    cache: ArtifactCache,
) -> None:
    """Verify referenced artifacts independently of later outputs."""
    directories, files = references
    _validate_directories(source_root, directories)
    for record in files:
        path = _artifact_file_path(source_root, record.relative)
        signature = _run_metadata.file_signature(path.lstat())
        current = _read_artifact(
            source_root, record.relative, signature, flush=False
        )
        if current != record:
            _integrity_error(record.relative)
        cache[record.relative] = (signature, current)


def _try_reflink(source: int, target: int) -> bool:
    if os.name != "posix":
        return False
    try:
        import fcntl

        fcntl.ioctl(target, _FICLONE, source)
    except OSError:
        os.ftruncate(target, 0)
        return False
    return True


def _materialize_artifact_bytes(
    source: pathlib.Path,
    target: pathlib.Path,
    record: ArtifactFile,
    expected_signature: _run_metadata.FileSignature,
) -> None:
    source_descriptor = os.open(
        source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        if (
            _run_metadata.file_signature(os.fstat(source_descriptor))
            != expected_signature
        ):
            raise _run_model.RunStoreError(
                "checkpoint artifact changed during materialization: "
                f"{record.relative.as_posix()}",
                code="checkpoint.artifact_corrupt",
                details={"path": record.relative.as_posix()},
            )
        target_descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            cloned = _try_reflink(source_descriptor, target_descriptor)
            size, digest = _transfer_and_hash(
                source_descriptor, None if cloned else target_descriptor
            )
            os.fsync(target_descriptor)
        finally:
            os.close(target_descriptor)
        current_signature = _run_metadata.file_signature(source.lstat())
        if (
            _run_metadata.file_signature(os.fstat(source_descriptor))
            != expected_signature
            or current_signature != expected_signature
            or size != record.size
            or digest != record.sha256
        ):
            raise _run_model.RunStoreError(
                "checkpoint artifact failed its integrity check: "
                f"{record.relative.as_posix()}",
                code="checkpoint.artifact_corrupt",
                details={"path": record.relative.as_posix()},
            )
    finally:
        os.close(source_descriptor)


def _copy_referenced_artifact(
    source_root: pathlib.Path, record: ArtifactFile, target: pathlib.Path
) -> None:
    source = _artifact_file_path(source_root, record.relative)
    metadata = source.lstat()
    if _artifact_mode(metadata) != record.mode:
        _integrity_error(record.relative)
    expected_signature = _run_metadata.file_signature(metadata)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        _materialize_artifact_bytes(source, target, record, expected_signature)
    except _run_model.RunStoreError:
        target.unlink(missing_ok=True)
        raise
    except OSError as error:
        target.unlink(missing_ok=True)
        raise _run_model.RunStoreError(
            "checkpoint artifact could not be materialized: "
            f"{record.relative.as_posix()}",
            code="checkpoint.artifact_unavailable",
            details={"path": record.relative.as_posix()},
        ) from error
    target.chmod(record.mode)


def materialize_artifact_references(
    source_root: pathlib.Path,
    destination: pathlib.Path,
    directories: Sequence[ArtifactDirectory],
    files: Sequence[ArtifactFile],
) -> None:
    _validate_directories(source_root, directories)
    for record in sorted(
        directories, key=lambda item: len(item.relative.parts)
    ):
        destination.joinpath(*record.relative.parts).mkdir(mode=0o700)
    for record in files:
        _copy_referenced_artifact(
            source_root,
            record,
            destination.joinpath(*record.relative.parts),
        )
    for record in sorted(
        directories, key=lambda item: len(item.relative.parts), reverse=True
    ):
        directory = destination.joinpath(*record.relative.parts)
        _run_metadata.fsync_directory(directory)
        directory.chmod(record.mode)
    _run_metadata.fsync_directory(destination)
