"""Private entry point for running one workflow in its own environment."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import chdir, contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Literal, TypeVar, cast
from uuid import uuid4

from ._definitions import load_definition
from ._lifecycle import (
    ArgumentMode,
    LifecycleCommand,
    Operation,
    decode_lifecycle_command,
)
from ._run_model import RUN_HISTORY_SCHEMA_VERSION
from ._run_store import (
    RunManifest,
    RunStore,
    RunStatus,
    load_run_manifest,
    run_is_active,
)
from .cli import WorkflowArguments, parse_arguments
from .declarations import (
    WorkflowConfiguration,
    WorkflowDefinition,
)
from .declarations.ids import GraphId
from .interpreter import CheckpointPolicy, Dispatcher, SessionPolicy

RUN_DIRECTORY = ".verdog/runs"
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ParamsT = TypeVar("ParamsT")
ScopeT = TypeVar("ScopeT")


@dataclass(frozen=True, slots=True)
class _RunRequest:
    root: Path
    definition_id: str
    definition_module: str
    requested_output: Path | None
    arguments: tuple[str, ...]
    checkpointing: CheckpointPolicy


@dataclass(slots=True)
class _LifecycleState:
    target_output: Path | None = None
    source: RunManifest | None = None
    mode: ArgumentMode | None = None
    problem: BaseException | None = None
    outcome: object = None
    manifest: RunManifest | None = None


@dataclass(frozen=True, slots=True)
class _PreparedLifecycle:
    request: LifecycleCommand
    definition: WorkflowDefinition[object, object, object, object]
    parsed: WorkflowArguments[object, object]
    source: RunManifest
    source_store: RunStore
    mode: ArgumentMode
    selected_arguments: tuple[str, ...]
    target_output: Path


def _arguments(
    definition: WorkflowDefinition[InputT, OutputT, ParamsT, ScopeT],
    declaration: ModuleType,
    arguments: tuple[str, ...],
    /,
) -> tuple[
    WorkflowDefinition[InputT, OutputT, ParamsT, ScopeT],
    WorkflowArguments[InputT, object],
]:
    options_value: object = getattr(declaration, "RuntimeOptions", None)
    configure_value: object = getattr(declaration, "configure", None)
    if not isinstance(options_value, type):
        raise RuntimeError("workflow declaration RuntimeOptions must be a type")
    options_type: type[object] = options_value
    if not callable(configure_value):
        raise RuntimeError("workflow declaration configure is not callable")
    parsed = parse_arguments(
        definition.input_type,
        arguments,
        params_types=definition.params_types,
        runtime_options=options_type,
        prog=f"verdog run {definition.id}",
    )
    if not isinstance(parsed.runtime, options_type):
        raise RuntimeError(
            "workflow declaration returned runtime options of the wrong type"
        )
    configure = cast(Callable[[object], object], configure_value)
    configuration = configure(parsed.runtime)
    if not isinstance(configuration, WorkflowConfiguration):
        raise RuntimeError(
            "workflow declaration configure() did not return WorkflowConfiguration"
        )
    return replace(definition, configuration=configuration), parsed


def _output_directory(root: Path, requested: Path | None, workflow_id: str) -> Path:
    output = requested
    if output is None:
        started = datetime.now(UTC)
        workflow = workflow_id.rsplit(".", 1)[-1]
        output = (
            root / RUN_DIRECTORY / workflow / f"{started:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        )
    output = output.resolve()
    if output.exists():
        if not output.is_dir():
            raise RuntimeError(f"output directory is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise RuntimeError(f"output directory is not empty: {output}")
    return output


def _run(request: _RunRequest, /) -> int:
    with chdir(request.root):
        definition, parsed = _configured_definition(
            request.root,
            request.definition_id,
            request.definition_module,
            request.arguments,
        )
        output = _output_directory(request.root, request.requested_output, definition.id)
        try:
            metadata = cast(
                object,
                json.loads((request.root / "project.json").read_text("utf-8")),
            )
        except (OSError, ValueError):
            metadata = None
        project = (
            cast(dict[str, object], metadata) if isinstance(metadata, dict) else {}
        )
        revision = project.get("generated_from")
        suffix = (
            f" at {revision[:12]}…" if isinstance(revision, str) and revision else ""
        )
        print(f"Running {definition.id}{suffix}")
        try:
            outcome = (
                Dispatcher(project_root=request.root).run(
                    definition,
                    parsed.input,
                    output_dir=output,
                    params=parsed.params,
                    runtime_options=parsed.runtime,
                    checkpointing=request.checkpointing,
                    workflow_arguments=request.arguments,
                    _record_run=True,
                )
            )
        except Exception as error:  # noqa: BLE001 - the project's own failure
            print(f"verdog: the workflow failed: {error}", file=sys.stderr)
            print(f"Outputs: {output}", file=sys.stderr)
            return 1
        print(f"Succeeded. Outputs: {output}")
        print(f"Result: {outcome!r}"[:400])
        return 0


def _configured_definition(
    root: Path,
    definition_id: str,
    definition_module: str,
    arguments: tuple[str, ...],
    /,
) -> tuple[
    WorkflowDefinition[object, object, object, object],
    WorkflowArguments[object, object],
]:
    definition_type: type[WorkflowDefinition[object, object, object, object]] = (
        WorkflowDefinition
    )
    definition, declaration = load_definition(
        root,
        definition_module,
        GraphId(definition_id),
        definition_type,
        lambda loaded: loaded.id,
    )
    return _arguments(definition, declaration, arguments)


def _require_lifecycle_contract(
    request: LifecycleCommand,
    source: RunManifest,
    /,
) -> tuple[ArgumentMode, tuple[str, ...]]:
    mode = request.arguments_mode
    if request.checkpoint is not None and request.checkpoint <= 0:
        raise RuntimeError("internal lifecycle checkpoint must be positive")
    if request.retry_incomplete and request.operation != "resume":
        raise RuntimeError("only resume can retry an incomplete invocation")
    if request.operation == "resume":
        if (
            request.sessions != "restore"
            or mode != "checkpoint"
            or request.checkpoint is None
        ):
            raise RuntimeError("invalid internal resume invocation")
    elif request.operation == "fork":
        if request.sessions not in {"branch", "fresh"} or mode != "checkpoint":
            raise RuntimeError("invalid internal fork invocation")
        if request.checkpoint is None:
            raise RuntimeError("an internal fork needs a checkpoint")
    else:
        if request.sessions not in {"branch", "fresh"} or mode == "checkpoint":
            raise RuntimeError("invalid internal restart invocation")
        if request.sessions == "branch" and request.checkpoint is None:
            raise RuntimeError("a branching restart needs a session checkpoint")
        if request.sessions == "fresh" and request.checkpoint is not None:
            raise RuntimeError("a fresh restart must not name a session checkpoint")

    recorded = source.launch.workflow_arguments
    if mode in {"checkpoint", "reused"}:
        if request.arguments != recorded:
            raise RuntimeError("recorded workflow arguments changed during launch")
        return mode, recorded
    return mode, request.arguments


def _effective_status(manifest: RunManifest, output: Path, /) -> RunStatus:
    if manifest.status is RunStatus.RUNNING and not run_is_active(output):
        return RunStatus.INTERRUPTED
    return manifest.status


@contextmanager
def _machine_output() -> Generator[None]:
    """Keep stdout as one JSON document, including across native child writes."""

    try:
        # Native code and subprocesses inherit the process descriptors, even when
        # Python's ``sys.stdout`` has been replaced by a capture or logging stream.
        os.fstat(1)
        os.fstat(2)
    except OSError:
        with redirect_stdout(sys.stderr):
            yield
        return
    sys.stdout.flush()
    duplicate = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield
        sys.stderr.flush()
    finally:
        os.dup2(duplicate, 1)
        os.close(duplicate)


def _error_detail(error: BaseException, /) -> dict[str, object]:
    code = getattr(error, "code", None)
    details = getattr(error, "details", None)
    message = str(error)
    if isinstance(error, SystemExit):
        code = "workflow.arguments_invalid"
        message = "the workflow arguments were rejected"
    detail: dict[str, object] = {
        "code": code if isinstance(code, str) else "run.operation_failed",
        "message": message or type(error).__name__,
    }
    if details is not None:
        detail["details"] = details
    return detail


def _error_document(operation: Operation, error: BaseException, /) -> dict[str, object]:
    return {
        "schema_version": RUN_HISTORY_SCHEMA_VERSION,
        "operation": operation,
        "status": "error",
        "error": _error_detail(error),
    }


def _operation_document(
    request: LifecycleCommand,
    state: _LifecycleState,
    /,
) -> dict[str, object]:
    assert state.source is not None
    assert state.mode is not None
    assert state.manifest is not None
    assert state.target_output is not None
    status = _effective_status(state.manifest, state.target_output)
    document: dict[str, object] = {
        "schema_version": RUN_HISTORY_SCHEMA_VERSION,
        "operation": request.operation,
        "status": status.value,
        "source_run_id": state.source.id,
        "source_checkpoint": request.checkpoint,
        "sessions": request.sessions,
        "arguments": state.mode,
        "run": state.manifest.as_summary(status=status),
    }
    if state.problem is not None:
        document["error"] = _error_detail(state.problem)
    return document


def _dispatch_lifecycle(
    prepared: _PreparedLifecycle,
    /,
) -> object:
    request = prepared.request
    dispatcher = Dispatcher(project_root=request.root)
    if request.operation == "resume":
        print(
            f"Resuming {prepared.definition.id} from {prepared.source.directory_name}"
        )
        return dispatcher.resume(
            prepared.definition,
            output_dir=prepared.target_output,
            retry_incomplete=request.retry_incomplete,
            _source_store=prepared.source_store,
        )

    if request.operation == "restart":
        print(
            f"Restarting {prepared.definition.id} from {prepared.source.directory_name}"
        )
        return dispatcher.restart(
            prepared.definition,
            prepared.parsed.input,
            source_output_dir=request.source_output,
            output_dir=prepared.target_output,
            _source_store=prepared.source_store,
            sessions=SessionPolicy(request.sessions),
            source_checkpoint=request.checkpoint,
            params=prepared.parsed.params,
            runtime_options=prepared.parsed.runtime,
            workflow_arguments=prepared.selected_arguments,
            arguments_mode=cast(Literal["reused", "overridden"], prepared.mode),
        )

    print(
        f"Forking {prepared.definition.id} from {prepared.source.directory_name} "
        f"checkpoint {request.checkpoint}"
    )
    assert request.checkpoint is not None
    return dispatcher.fork(
        prepared.definition,
        source_output_dir=request.source_output,
        checkpoint=request.checkpoint,
        output_dir=prepared.target_output,
        sessions=SessionPolicy(request.sessions),
        _source_store=prepared.source_store,
    )


def _attempt_lifecycle(
    request: LifecycleCommand,
    /,
) -> _LifecycleState:
    state = _LifecycleState(
        target_output=(request.source_output if request.operation == "resume" else None)
    )
    try:
        source_store = RunStore.open(request.source_output)
        state.source = source_store.manifest()
        state.mode, selected_arguments = _require_lifecycle_contract(
            request,
            state.source,
        )
        definition, parsed = _configured_definition(
            request.root,
            request.definition_id,
            request.definition_module,
            selected_arguments,
        )
        if state.target_output is None:
            state.target_output = _output_directory(request.root, None, definition.id)
        state.outcome = _dispatch_lifecycle(
            _PreparedLifecycle(
                request=request,
                definition=definition,
                parsed=parsed,
                source=state.source,
                source_store=source_store,
                mode=state.mode,
                selected_arguments=selected_arguments,
                target_output=state.target_output,
            )
        )
    except BaseException as error:  # noqa: BLE001 - the workflow is untrusted
        state.problem = error
    return state


def _load_lifecycle_manifest(state: _LifecycleState, /) -> None:
    if state.target_output is None:
        return
    try:
        state.manifest = load_run_manifest(state.target_output)
    except Exception as error:  # noqa: BLE001 - preserve the original failure
        if state.problem is None:
            state.problem = error


def _has_manifest_response(
    operation: Operation,
    state: _LifecycleState,
    /,
) -> bool:
    return (
        state.source is not None
        and state.mode is not None
        and state.manifest is not None
        and (
            operation != "resume"
            or state.problem is None
            or state.manifest.updated_at != state.source.updated_at
        )
    )


def _emit_manifest_response(
    request: LifecycleCommand,
    state: _LifecycleState,
    /,
) -> None:
    document = _operation_document(request, state)
    if request.as_json:
        print(json.dumps(document, indent=2))
        return
    status = cast(str, document["status"])
    assert state.target_output is not None
    if state.problem is None:
        print(
            f"{request.operation.capitalize()} {status}. Outputs: {state.target_output}"
        )
        print(f"Result: {state.outcome!r}"[:400])
    else:
        print(
            f"verdog: {request.operation} {status}: {state.problem}",
            file=sys.stderr,
        )
        print(f"Outputs: {state.target_output}", file=sys.stderr)


def _emit_error_response(
    operation: Operation,
    problem: BaseException,
    as_json: bool,
    /,
) -> None:
    if as_json:
        print(json.dumps(_error_document(operation, problem), indent=2))
    else:
        print(f"verdog: {operation} failed: {problem}", file=sys.stderr)


def _finish_lifecycle(
    request: LifecycleCommand,
    state: _LifecycleState,
    /,
) -> None:
    if _has_manifest_response(request.operation, state):
        _emit_manifest_response(request, state)
        return
    if state.problem is not None:
        _emit_error_response(request.operation, state.problem, request.as_json)
        return
    missing = RuntimeError("the lifecycle operation produced no run manifest")
    if request.as_json:
        print(json.dumps(_error_document(request.operation, missing), indent=2))
    else:
        print(f"verdog: {missing}", file=sys.stderr)
    state.problem = missing


def _lifecycle_exit_status(problem: BaseException | None, /) -> int:
    if problem is None:
        return 0
    if isinstance(problem, KeyboardInterrupt):
        return 130
    if isinstance(problem, SystemExit) and isinstance(problem.code, int):
        return problem.code
    return 1


def _lifecycle(
    request: LifecycleCommand,
    /,
) -> int:
    output_context = _machine_output() if request.as_json else nullcontext()
    with chdir(request.root), output_context:
        state = _attempt_lifecycle(request)
    _load_lifecycle_manifest(state)
    _finish_lifecycle(request, state)
    return _lifecycle_exit_status(state.problem)


def _lifecycle_main(argv: Sequence[str], /) -> int:
    if len(argv) != 3 or argv[1] != "--verdog-lifecycle":
        print("verdog: invalid internal lifecycle invocation", file=sys.stderr)
        return 2
    try:
        request = decode_lifecycle_command(argv[2])
    except ValueError as error:
        print(f"verdog: {error}", file=sys.stderr)
        return 2
    return _lifecycle(request)


def _run_main(argv: Sequence[str], /) -> int:
    if len(argv) < 6:
        print("verdog: invalid internal run invocation", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    requested = Path(argv[4]) if argv[4] else None
    prefix = "--verdog-checkpointing="
    if not argv[5].startswith(prefix):
        print("verdog: invalid internal checkpoint policy", file=sys.stderr)
        return 2
    try:
        checkpointing = CheckpointPolicy(argv[5].removeprefix(prefix))
    except ValueError:
        print("verdog: invalid internal checkpoint policy", file=sys.stderr)
        return 2
    try:
        return _run(
            _RunRequest(
                root=root,
                definition_id=argv[2],
                definition_module=argv[3],
                requested_output=requested,
                arguments=tuple(argv[6:]),
                checkpointing=checkpointing,
            )
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"verdog: {error}", file=sys.stderr)
        return 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--verdog-lifecycle":
        return _lifecycle_main(sys.argv)
    return _run_main(sys.argv)


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI process
    raise SystemExit(main())
