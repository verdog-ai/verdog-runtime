"""Reject checkpoints when known Python code inputs have drifted.

Continuation snapshots contain serialized instances of authored workflow and
runtime classes.  Loading one under different source or dependency versions is
not a migration: it can fail during deserialization or, worse, continue with
different behavior.  A run therefore records this conservative fingerprint and
exact resume/fork operations compare it before reading the snapshot.

This is deliberately separate from run-store checksums, which detect accidental
byte corruption.  This module provides an additional, deliberately conservative
signal that the current process can interpret those bytes.  It is not an
authenticity mechanism or a complete description of process behavior.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import cast

CHECKPOINT_COMPATIBILITY_VERSION = "3"
_EXECUTION_MODEL = "synchronous-activation-stack-v1"
_ENVIRONMENT_MARKER = Path(sys.prefix) / ".verdog-environment.json"


def _digest_parts(parts: Iterable[bytes]) -> str:
    digest = sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _tree_parts(
    path: Path,
    logical_path: Path,
    ancestors: frozenset[tuple[int, int]] = frozenset(),
) -> Iterable[bytes]:
    """Yield a deterministic tree, following source symlinks without looping."""

    relative = logical_path.as_posix().encode("utf-8")
    if "__pycache__" in logical_path.parts or logical_path.suffix == ".pyc":
        return
    yield b"path"
    yield relative
    if path.is_symlink():
        yield b"symlink"
        yield str(path.readlink()).encode("utf-8")
    if path.is_dir():
        yield b"directory"
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in ancestors:
            yield b"cycle"
            return
        nested_ancestors = ancestors | {identity}
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            yield from _tree_parts(
                child,
                logical_path / child.name,
                nested_ancestors,
            )
        return
    if path.is_file():
        yield b"file"
        yield path.read_bytes()
        return
    yield b"other"


def _source_roots(root: Path) -> tuple[Path, ...]:
    """Return in-process source roots declared by ``verdog sync``.

    Programmatic runtime use has no environment marker, so the owning project's
    source directory remains the safe fallback.  A marker is only accepted when
    it names that directory; this avoids incorporating an unrelated workflow
    environment when a dispatcher is embedded in another process.
    """

    owned = (root / "src").resolve()
    try:
        marker: object = json.loads(_ENVIRONMENT_MARKER.read_text("utf-8"))
    except (OSError, ValueError):
        return (owned,)
    if not isinstance(marker, dict):
        return (owned,)
    raw_roots = cast(dict[object, object], marker).get("source_roots")
    if not isinstance(raw_roots, list):
        return (owned,)
    declared_values: list[Path] = []
    for value in cast(list[object], raw_roots):
        if not isinstance(value, str) or not Path(value).is_absolute():
            return (owned,)
        declared_values.append(Path(value).resolve())
    declared = tuple(dict.fromkeys(declared_values))
    return declared if owned in declared else (owned,)


def _source_parts(root: Path) -> Iterable[bytes]:
    for index, source in enumerate(_source_roots(root)):
        yield b"source-root"
        yield str(index).encode("ascii")
        yield from _tree_parts(source, Path("src"))


def _environment_digest() -> str:
    installed: list[tuple[str, str]] = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if isinstance(name, str) and name:
            installed.append((name.casefold(), distribution.version))
    encoded = json.dumps(sorted(set(installed)), separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _runtime_parts() -> Iterable[bytes]:
    root = Path(__file__).resolve().parent
    yield from _tree_parts(root, Path("verdog_runtime"))


def checkpoint_compatibility(project_root: Path, /) -> Mapping[str, str]:
    """Fingerprint known code domains that may define a serialized object."""

    root = project_root.resolve()
    cache_tag = sys.implementation.cache_tag or "unavailable"
    try:
        runtime_version = metadata.version("verdog-runtime")
    except metadata.PackageNotFoundError:  # pragma: no cover - source-only development
        runtime_version = "uninstalled"
    values = {
        "format": CHECKPOINT_COMPATIBILITY_VERSION,
        "execution_model": _EXECUTION_MODEL,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_implementation": sys.implementation.name,
        "python_cache_tag": cache_tag,
        "platform": sys.platform,
        "runtime": runtime_version,
        "runtime_sha256": _digest_parts(_runtime_parts()),
        "source_sha256": _digest_parts(_source_parts(root)),
        "environment_sha256": _environment_digest(),
    }
    return MappingProxyType(values)


def compatibility_drift(
    expected: Mapping[str, str], actual: Mapping[str, str], /
) -> str | None:
    """Describe exact compatibility drift, or return ``None``."""

    missing = sorted(expected.keys() - actual.keys())
    added = sorted(actual.keys() - expected.keys())
    changed = sorted(
        key for key in expected.keys() & actual.keys() if expected[key] != actual[key]
    )
    if not (missing or added or changed):
        return None
    parts: list[str] = []
    if changed:
        parts.append("changed " + ", ".join(changed))
    if missing:
        parts.append("missing " + ", ".join(missing))
    if added:
        parts.append("added " + ", ".join(added))
    return "; ".join(parts)


__all__ = ["checkpoint_compatibility", "compatibility_drift"]
