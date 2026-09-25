"""Cancellation must stop installer descendants before environment recovery."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from verdog_runtime.cli.sync import (
    _install,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.skipif(
    os.name != "posix", reason="POSIX process groups and file locks"
)
def test_installer_descendants_stop_before_interruption_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    ready = tmp_path / "ready"
    lock_path = tmp_path / "child.lock"
    child_code = """
import fcntl, os, signal, sys
from pathlib import Path
with open(sys.argv[1], "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    ready = Path(sys.argv[2])
    ready.with_suffix(".tmp").write_text(str(os.getpid()))
    ready.with_suffix(".tmp").replace(ready)
    signal.pause()
"""
    parent_code = """
import subprocess, sys
subprocess.Popen([sys.executable, "-c", *sys.argv[1:]]).wait()
"""
    command = [
        sys.executable,
        "-c",
        parent_code,
        child_code,
        str(lock_path),
        str(ready),
    ]
    wait = subprocess.Popen[bytes].wait
    parents: list[subprocess.Popen[bytes]] = []

    def interrupt_once(
        process: subprocess.Popen[bytes], timeout: float | None = None
    ) -> int:
        if not parents:
            parents.append(process)
            deadline = time.monotonic() + 5
            while not ready.exists():
                assert process.poll() is None, (
                    "dummy installer exited before its child was ready"
                )
                assert time.monotonic() < deadline, (
                    "dummy installer child did not start"
                )
                time.sleep(0.01)
            raise KeyboardInterrupt
        return wait(process, timeout=5 if timeout is None else timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupt_once)
    try:
        with pytest.raises(KeyboardInterrupt):
            _install(command, tmp_path)
        assert len(parents) == 1 and parents[0].poll() is not None
        # An exited child releases its lock even if temporarily a zombie.
        # The parent's exit status does not prove the backend stopped.
        with lock_path.open("rb") as lock:
            deadline = time.monotonic() + 2
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline, (
                        "installer child survived cancellation"
                    )
                    time.sleep(0.01)
    finally:
        # Clean up even if only the parent was killed.
        if ready.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(ready.read_text()), signal.SIGKILL)
        if parents and parents[0].poll() is None:
            parents[0].kill()
            wait(parents[0], timeout=5)
