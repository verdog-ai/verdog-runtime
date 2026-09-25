from __future__ import annotations

import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TypedDict, cast

from .declarations.ids import GraphId, ParameterAddress

_PROCESS_TREE_ENV = "_VERDOG_PROCESS_TREE"
_TERMINATION_GRACE_SECONDS = 0.5


class ProcessOptions(TypedDict, total=False):
    creationflags: int
    start_new_session: bool


def normalize_project_path(value: str, /) -> str:
    """Return one safe owner-relative path in platform-neutral form."""

    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("child project path must be owner-relative")
    parts = tuple(part for part in relative.parts if part not in ("", "."))
    return PurePosixPath(*parts).as_posix() if parts else "."


def compose_project_path(owner: str, descendant: str, /) -> str:
    owner = normalize_project_path(owner)
    descendant = normalize_project_path(descendant)
    if owner == ".":
        return descendant
    if descendant == ".":
        return owner
    return f"{owner}/{descendant}"


def normalize_parameter_address(value: object, /) -> ParameterAddress:
    parts = cast(tuple[object, ...], value) if isinstance(value, tuple) else ()
    if (
        len(parts) != 2
        or not isinstance(parts[0], str)
        or not isinstance(parts[1], str)
        or not parts[1]
    ):
        raise TypeError("a parameter address must be (project_path, GraphId)")
    return normalize_project_path(parts[0]), GraphId(parts[1])


def resolve_project_root(root: Path, project_path: str, /) -> Path:
    """Resolve one child project without letting it escape its owner."""

    relative = Path(project_path)
    normalize_project_path(project_path)
    owner = root.resolve()
    target = (owner / relative).resolve()
    if not target.is_relative_to(owner):
        raise ValueError("child project path escapes its owner")
    return target


def wait_until_reaped(
    process: subprocess.Popen[bytes], /, timeout: float | None = None
) -> bool:
    """Reap a child within an optional caller-owned polling interval."""

    try:
        process.wait(timeout=None if timeout is None else max(timeout, 0.0))
    except subprocess.TimeoutExpired:
        return False
    return True


def process_options(environment: dict[str, str], /) -> tuple[ProcessOptions, bool]:
    """Mark and isolate the outermost process in one cancellable process tree."""

    nested = environment.get(_PROCESS_TREE_ENV) == "1"
    environment[_PROCESS_TREE_ENV] = "1"
    options: ProcessOptions
    if os.name == "nt":
        options = {
            "creationflags": (
                0 if nested else getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        }
    else:
        options = {"start_new_session": not nested}
    return options, not nested


def _posix_descendants(pid: int, /) -> tuple[int, ...]:
    """Snapshot descendants deepest-first without third-party dependencies."""

    children: dict[int, list[int]] = {}
    proc_children = Path(f"/proc/{pid}/task/{pid}/children")
    if proc_children.exists():
        pending = [pid]
        seen = {pid}
        while pending:
            parent = pending.pop()
            try:
                direct = [
                    int(item)
                    for item in Path(f"/proc/{parent}/task/{parent}/children")
                    .read_text("ascii")
                    .split()
                ]
            except (FileNotFoundError, PermissionError, ValueError):
                direct = []
            children[parent] = direct
            for child in direct:
                if child not in seen:
                    seen.add(child)
                    pending.append(child)
    else:
        try:
            snapshot = subprocess.run(
                ["ps", "-A", "-o", "pid=,ppid="],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        except OSError:
            snapshot = ""
        for line in snapshot.splitlines():
            try:
                child, parent = (int(item) for item in line.split())
            except (TypeError, ValueError):
                continue
            children.setdefault(parent, []).append(child)

    ordered: list[int] = []
    seen_descendants = {pid}
    traversal: list[tuple[int, bool]] = [(pid, False)]
    while traversal:
        current, expanded = traversal.pop()
        if expanded:
            if current != pid:
                ordered.append(current)
            continue
        traversal.append((current, True))
        for child in reversed(children.get(current, ())):
            if child in seen_descendants:
                continue
            seen_descendants.add(child)
            traversal.append((child, False))
    return tuple(ordered)


def _signal_processes(pids: tuple[int, ...], signum: signal.Signals, /) -> None:
    for pid in dict.fromkeys(pids):
        with suppress(ProcessLookupError):
            os.kill(pid, signum)


def terminate_process_tree(
    process: subprocess.Popen[bytes],
    /,
    *,
    owns_group: bool,
) -> None:
    """Stop a blocking child and every process it started, then reap it."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            pass
        if process.poll() is None:
            with suppress(ProcessLookupError):
                process.kill()
        wait_until_reaped(process)
        return

    descendants = _posix_descendants(process.pid)
    if owns_group:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    else:
        _signal_processes((*descendants, process.pid), signal.SIGTERM)
    try:
        wait_until_reaped(process, _TERMINATION_GRACE_SECONDS)
    finally:
        if owns_group:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            _signal_processes(descendants, signal.SIGKILL)
        else:
            remaining = (*descendants, *_posix_descendants(process.pid))
            _signal_processes((*remaining, process.pid), signal.SIGKILL)
        wait_until_reaped(process)


def cleanup_after_interruption(
    process: subprocess.Popen[bytes],
    /,
    *,
    owns_group: bool,
) -> None:
    with suppress(BaseException):
        terminate_process_tree(process, owns_group=owns_group)
