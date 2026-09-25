"""Run a workflow through the interpreter in its isolated root environment."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
from typing import Literal

from verdog_runtime import _lifecycle
from verdog_runtime.cli import local, sync


@dataclasses.dataclass(frozen=True, slots=True)
class LifecycleRequest:
    """A resume, restart, or fork request for an isolated workflow process."""

    operation: _lifecycle.Operation
    source: pathlib.Path
    workflow_id: str
    sessions: _lifecycle.SessionMode
    checkpoint: int | None = None
    arguments: tuple[str, ...] = ()
    arguments_mode: _lifecycle.ArgumentMode = "checkpoint"
    retry_incomplete: bool = False
    as_json: bool = False


def _isolated_environment(
    environment: pathlib.Path, interpreter: pathlib.Path, /
) -> dict[str, str]:
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
    clone: local.Clone, workflow_id: str | None, /
) -> tuple[str, str, pathlib.Path, dict[str, str]]:
    workflow = clone.workflow_definition(workflow_id)
    environment = sync.require_current_environment(clone, workflow)
    interpreter = local.interpreter_in(environment)
    if interpreter is None:
        raise local.WorkspaceError(
            f"{environment} has no Python interpreter; run `verdog sync`"
        )
    return (
        workflow.id,
        workflow.module,
        interpreter,
        _isolated_environment(environment, interpreter),
    )


def run(
    clone: local.Clone,
    output_dir: pathlib.Path | None = None,
    *,
    workflow_id: str | None = None,
    arguments: tuple[str, ...] = (),
    checkpointing: Literal["off", "auto", "required"] = "auto",
) -> int:
    """Trampoline into one workflow definition's own interpreter."""
    definition_id, definition_module, interpreter, process_environment = (
        _runtime(clone, workflow_id)
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
    clone: local.Clone,
    request: LifecycleRequest,
    /,
) -> int:
    """Trampoline a run lifecycle operation into its workflow environment."""
    definition_id, definition_module, interpreter, process_environment = (
        _runtime(clone, request.workflow_id)
    )
    lifecycle = _lifecycle.LifecycleCommand(
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
        _lifecycle.encode_lifecycle_command(lifecycle),
    ]
    return subprocess.run(  # noqa: S603 - fixed interpreter and argv, no shell
        command,
        cwd=clone.root,
        env=process_environment,
        check=False,
    ).returncode
