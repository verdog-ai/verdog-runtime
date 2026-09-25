"""Small cross-platform advisory-file-lock primitive."""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Generator
from typing import BinaryIO, Protocol, cast


class FileLockUnavailable(Exception):
    """The requested nonblocking advisory lock is already held."""


class _WindowsLockApi(Protocol):
    LK_LOCK: int
    LK_NBLCK: int
    LK_UNLCK: int

    def locking(
        self, file_descriptor: int, mode: int, byte_count: int, /
    ) -> None: ...


def _windows_lock_api() -> _WindowsLockApi:
    return cast(_WindowsLockApi, importlib.import_module("msvcrt"))


def _lock_windows(stream: BinaryIO, *, blocking: bool) -> None:
    msvcrt = _windows_lock_api()

    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
    try:
        msvcrt.locking(stream.fileno(), mode, 1)
    except OSError as error:
        raise FileLockUnavailable from error


def _unlock_windows(stream: BinaryIO) -> None:
    msvcrt = _windows_lock_api()

    stream.seek(0)
    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _lock_posix(stream: BinaryIO, *, blocking: bool) -> None:
    import fcntl

    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(stream.fileno(), flags)
    except BlockingIOError as error:
        raise FileLockUnavailable from error


def _unlock_posix(stream: BinaryIO) -> None:
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def locked_file(stream: BinaryIO, *, blocking: bool) -> Generator[None]:
    """Hold one advisory lock for the lifetime of the context."""
    if os.name == "nt":
        _lock_windows(stream, blocking=blocking)
    else:
        _lock_posix(stream, blocking=blocking)
    try:
        yield
    finally:
        if os.name == "nt":
            _unlock_windows(stream)
        else:
            _unlock_posix(stream)
