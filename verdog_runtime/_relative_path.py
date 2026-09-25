"""Shared grammar for strict portable relative paths in durable manifests."""

from __future__ import annotations

from pathlib import PurePosixPath


def strict_posix_relative_parts(value: str, /) -> tuple[str, ...] | None:
    """Return normalized parts only when ``value`` is already strict POSIX form."""

    path = PurePosixPath(value)
    if (
        not value
        or "\0" in value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return None
    return path.parts
