"""Shared grammar for strict portable relative paths in durable manifests."""

from __future__ import annotations

import pathlib


def strict_posix_relative_parts(value: str, /) -> tuple[str, ...] | None:
    """Return path parts if ``value`` has strict POSIX form."""
    path = pathlib.PurePosixPath(value)
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
