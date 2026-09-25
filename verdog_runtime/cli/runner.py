"""Run a workflow through the interpreter in its isolated root environment."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .._lifecycle import (
    ArgumentMode,
    LifecycleCommand,
    Operation,
    SessionMode,
    encode_lifecycle_command,
)
from .local import Clone, WorkspaceError, interpreter_in
from .sync import require_current_environment


@dataclass(frozen=True, slots=True)
class LifecycleRequest:
    operation: Operation
    source: Path
    workflow_id: str
    sessions: SessionMode
    checkpoint: int | None = None
    arguments: tuple[str, ...] = ()
    arguments_mode: ArgumentMode = "checkpoint"
    retry_incomplete: bool = False
    as_json: bool = False


def _isolated_environment(environment: Path, interpreter: Path, /) -> dict[str, str]:
    process_environment = os.environ.copy()
    process_environment.pop("PYTHONHOME", None)
    process_environment.pop("PYTHONPATH", None)
    process_environment["PYTHONNOUSERSITE"] = "1"
    process_environment["VIRTUAL_ENV"] = str(environment)
    path = process_environment.get("PATH")
    process_environment["PATH"] = str(interpreter.parent) + (
        os.pathsep + path if path else ""
    )
    return process_environment


def _runtime(
    clone: Clone, workflow_id: str | None, /
) -> tuple[str, str, Path, dict[str, str]]:
    workflow = clone.workflow_definition(workflow_id)
    environment = require_current_environment(clone, workflow)
    interpreter = interpreter_in(environment)
    if interpreter is None:
        raise WorkspaceError(
            f"{environment} has no Python interpreter; run `verdog sync`"
        )
    return (
        workflow.id,
        workflow.module,
        interpreter,
        _isolated_environment(environment, interpreter),
    )


def run(
    clone: Clone,
    output_dir: Path | None = None,
    *,
    workflow_id: str | None = None,
    arguments: tuple[str, ...] = (),
    checkpointing: Literal["off", "auto", "required"] = "auto",
) -> int:
    """Trampoline into one workflow definition's own interpreter."""

    definition_id, definition_module, interpreter, process_environment = _runtime(
        clone, workflow_id
    )
    requested = "" if output_dir is None else str(output_dir.resolve())
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            str(interpreter),
            "-m",
            "verdog_runtime.entry",
            str(clone.root),
            definition_id,
            definition_module,
            requested,
            f"--verdog-checkpointing={checkpointing}",
            *arguments,
        ],
        cwd=clone.root,
        env=process_environment,
        check=False,
    ).returncode


def operate(
    clone: Clone,
    request: LifecycleRequest,
    /,
) -> int:
    """Trampoline a run lifecycle operation into its workflow environment."""

    definition_id, definition_module, interpreter, process_environment = _runtime(
        clone, request.workflow_id
    )
    lifecycle = LifecycleCommand(
        operation=request.operation,
        root=clone.root.resolve(),
        definition_id=definition_id,
        definition_module=definition_module,
        source_output=request.source.resolve(),
        sessions=request.sessions,
        checkpoint=request.checkpoint,
        arguments_mode=request.arguments_mode,
        retry_incomplete=request.retry_incomplete,
        as_json=request.as_json,
        arguments=request.arguments,
    )
    command = [
        str(interpreter),
        "-m",
        "verdog_runtime.entry",
        "--verdog-lifecycle",
        encode_lifecycle_command(lifecycle),
    ]
    return subprocess.run(  # noqa: S603 - fixed interpreter and argv, no shell
        command,
        cwd=clone.root,
        env=process_environment,
        check=False,
    ).returncode
