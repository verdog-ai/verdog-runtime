"""Process cleanup retains group ownership after the launcher exits."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import cast

import pytest

from verdog_runtime import _process


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_exited_group_leader_does_not_leave_its_descendant_running() -> None:
    worker = "import os, time; print(os.getpid(), flush=True); time.sleep(30)"
    launcher = (
        "import subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {worker!r}], "
        "stdout=subprocess.PIPE); "
        "sys.stdout.buffer.write(child.stdout.readline())"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", launcher],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 0
        descendant = int(output)
        _process.terminate_process_tree(process, owns_group=True)
        deadline = time.monotonic() + 3
        while True:
            status = subprocess.run(
                ["ps", "-p", str(descendant), "-o", "stat="],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if not status or status.startswith("Z"):
                break
            assert time.monotonic() < deadline, "descendant survived cleanup"
            time.sleep(0.01)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
@pytest.mark.parametrize("exited", [False, True])
def test_nested_cleanup_never_signals_the_callers_group(
    monkeypatch: pytest.MonkeyPatch, exited: bool
) -> None:
    signals: list[int] = []
    process = cast(
        subprocess.Popen[bytes],
        SimpleNamespace(pid=101, poll=lambda: 0 if exited else None),
    )

    def no_group_signal(*_args: object) -> None:
        pytest.fail("nested cleanup must not kill its caller or siblings")

    def descendants(_pid: int) -> tuple[int, ...]:
        return (102,)

    def reaped(_process: object, _timeout: object = None) -> bool:
        return True

    def kill(pid: int, _signal: int) -> None:
        signals.append(pid)

    monkeypatch.setattr(_process.os, "killpg", no_group_signal)
    monkeypatch.setattr(_process.os, "kill", kill)
    monkeypatch.setattr(_process, "_posix_descendants", descendants)
    monkeypatch.setattr(_process, "wait_until_reaped", reaped)
    _process.terminate_process_tree(process, owns_group=False)
    assert set(signals) == (set() if exited else {101, 102})
