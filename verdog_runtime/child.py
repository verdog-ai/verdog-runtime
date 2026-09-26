"""The private process boundary used by `WorkflowCall` nodes."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    NoReturn,
    cast,
)

import cloudpickle

from verdog_runtime import (
    _checkpoint_compatibility,
    _child_checkpoint,
    _definitions,
    _process,
    _run_model,
    _statistics,
    declarations,
)
from verdog_runtime import _protocol as protocol_module
from verdog_runtime.declarations import ids

if TYPE_CHECKING:
    from verdog_runtime.declarations import GraphDefinition
    from verdog_runtime.interpreter._calls import Budget, CallScope
    from verdog_runtime.interpreter.execution import (
        Dispatcher,
        ExecutionEvent,
        _CheckpointEmission,  # pyright: ignore[reportPrivateUsage]
        _ParameterRegistry,  # pyright: ignore[reportPrivateUsage]
    )

type _EmitFrame = Callable[
    [
        protocol_module.EventFrame
        | protocol_module.TimingFrame
        | protocol_module.CheckpointFrame
    ],
    None,
]
type _CancellationCheck = Callable[[], None]
_OMITTED = object()
_WAIT_SLICE_SECONDS = 0.05
_READER_JOIN_SECONDS = 1.0


@dataclasses.dataclass(frozen=True, slots=True)
class _ResponseChunk:
    value: bytes
    consumed: threading.Event


@dataclasses.dataclass(frozen=True, slots=True)
class _ResponseFailure:
    error: BaseException


class _ResponseEnd:
    pass


type _ResponseItem = _ResponseChunk | _ResponseFailure | _ResponseEnd
_RESPONSE_END = _ResponseEnd()
_REQUEST_WRITTEN = object()


class _ChildStopped(BaseException):
    """Unwind a child synchronously when its parent terminates it."""


def _register_project_by_value(project: pathlib.Path, /) -> None:
    root = (project / "src").resolve()
    if not root.is_relative_to(project.resolve()):
        raise ValueError("project source root escapes its project")
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            continue
        try:
            if pathlib.Path(source).resolve().is_relative_to(root):
                cloudpickle.register_pickle_by_value(module)
        except OSError:
            continue


def _local_definition_id(
    definition_id: ids.GraphId, definition_module: str, /
) -> str:
    package, separator, name = str(definition_id).rpartition(".")
    if (
        not separator
        or not package
        or not definition_module.startswith(package + ".")
    ):
        raise ValueError("child definition does not belong to its module")
    if not ids.is_valid_definition_id(name):
        raise ValueError("child definition id is invalid")
    return name


def _environment(
    project: pathlib.Path, definition_id: ids.GraphId, definition_module: str, /
) -> pathlib.Path:
    name = _local_definition_id(definition_id, definition_module)
    return project / ".verdog" / "environments" / name


def _interpreter(environment: pathlib.Path, /) -> pathlib.Path | None:
    for relative in ("bin/python", "Scripts/python.exe"):
        candidate = environment / relative
        if candidate.is_file():
            return candidate
    return None


def _request(
    definition_id: ids.GraphId,
    definition_module: str,
    input: object,
    run_id: ids.RunId,
    transitions_remaining: int,
    output_dir: pathlib.Path,
    call_path: pathlib.Path,
    project_path: str,
    params_override: object = _OMITTED,
    *,
    started_at: float | None = None,
    checkpointing: _run_model.CheckpointPolicy = (
        _run_model.CheckpointPolicy.OFF
    ),
    resume_continuation: bytes | None = None,
    retry_incomplete: bool = False,
) -> bytes:
    return protocol_module.encode_frame(
        protocol_module.CallFrame(
            definition_id=definition_id,
            definition_module=definition_module,
            input=protocol_module.encode_payload(input),
            run_id=run_id,
            transitions_remaining=transitions_remaining,
            output_dir=str(output_dir),
            call_path=call_path.as_posix(),
            project_path=project_path,
            started_at=protocol_module.seconds(
                time.monotonic() if started_at is None else started_at
            ),
            checkpointing=checkpointing,
            retry_incomplete=retry_incomplete,
            params_override=(
                None
                if params_override is _OMITTED
                else protocol_module.encode_payload(params_override)
            ),
            resume_continuation=(
                None
                if resume_continuation is None
                else protocol_module.encode_binary_payload(resume_continuation)
            ),
        )
    )


def _detail(returncode: int | None) -> str:
    return f"child exited with {returncode}"


def _record_reply(
    frame: protocol_module.ReplyFrame,
    transitions_remaining: int,
    event_handler: protocol_module.EventHandler | None,
    checkpoint_handler: protocol_module.CheckpointHandler | None,
    timing_handler: Callable[[_statistics.TimingRecord], None] | None,
    /,
) -> tuple[protocol_module.TerminalFrame | None, int]:
    if isinstance(frame, protocol_module.TimingFrame):
        if timing_handler is not None:
            timing_handler(frame.record)
        return None, transitions_remaining
    if isinstance(frame, protocol_module.EventFrame):
        if event_handler is not None:
            event_handler(frame)
        return None, frame.transitions_remaining
    if isinstance(frame, protocol_module.CheckpointFrame):
        if checkpoint_handler is not None:
            checkpoint_handler(frame)
        return None, frame.transitions_remaining
    return frame, frame.transitions_remaining


class _FramedResponseReader:
    """Read a blocking pipe without blocking cancellation checks in caller."""

    def __init__(self, stream: BinaryIO, /) -> None:
        self._stream = stream
        self._items: queue.Queue[_ResponseItem] = queue.Queue()
        self._stopped = threading.Event()
        self._pending_lock = threading.Lock()
        self._pending: _ResponseChunk | None = None
        self._thread = threading.Thread(
            target=self._read,
            name="verdog-child-protocol-reader",
            daemon=True,
        )
        self._thread.start()

    def _read(self) -> None:
        try:
            while not self._stopped.is_set():
                value = self._stream.read(64 * 1024)
                if not value:
                    break
                item = _ResponseChunk(value=value, consumed=threading.Event())
                with self._pending_lock:
                    self._pending = item
                self._items.put(item)
                item.consumed.wait()
                with self._pending_lock:
                    if self._pending is item:
                        self._pending = None
        except BaseException as error:
            if not self._stopped.is_set():
                self._items.put(_ResponseFailure(error))
        finally:
            self._items.put(_RESPONSE_END)

    def receive(
        self,
        check_cancelled: _CancellationCheck | None,
        deadline: float | None,
        /,
    ) -> _ResponseItem:
        while True:
            wait = _cooperate(check_cancelled, deadline)
            try:
                return self._items.get(timeout=wait)
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._stopped.set()
        with self._pending_lock:
            if self._pending is not None:
                self._pending.consumed.set()

    def close(self) -> None:
        self.stop()
        self._thread.join(timeout=_READER_JOIN_SECONDS)


class _RequestWriter:
    """Keep a large request write from hiding cancellation or its deadline."""

    def __init__(self, stream: BinaryIO, request: bytes, /) -> None:
        self._stream = stream
        self._request = request
        self._result: queue.Queue[object | BaseException] = queue.Queue(
            maxsize=1
        )
        self._thread = threading.Thread(
            target=self._write,
            name="verdog-child-protocol-writer",
            daemon=True,
        )
        self._thread.start()

    def _write(self) -> None:
        outcome: object | BaseException = _REQUEST_WRITTEN
        try:
            self._stream.write(self._request)
            self._stream.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except BaseException as error:
            outcome = error
        finally:
            try:
                self._stream.close()
            except BaseException as error:
                if outcome is _REQUEST_WRITTEN:
                    outcome = error
            self._result.put(outcome)

    def wait(
        self,
        check_cancelled: _CancellationCheck | None,
        deadline: float | None,
        /,
    ) -> None:
        while True:
            wait = _cooperate(check_cancelled, deadline)
            try:
                outcome = self._result.get(timeout=wait)
            except queue.Empty:
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            return

    def close(self) -> None:
        self._thread.join(timeout=_READER_JOIN_SECONDS)


def _cooperate(
    check_cancelled: _CancellationCheck | None,
    deadline: float | None,
    /,
) -> float:
    if check_cancelled is not None:
        check_cancelled()
    if deadline is None:
        return _WAIT_SLICE_SECONDS
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(
            "child workflow deadline expired [child_process_timeout]"
        )
    return min(_WAIT_SLICE_SECONDS, remaining)


def _decode_response_chunk(
    value: bytes,
    buffer: bytearray,
    terminal: protocol_module.TerminalFrame | None,
    protocol_error: Exception | None,
    transitions_remaining: int,
    *,
    run_id: ids.RunId,
    call_path: pathlib.Path,
    project_path: str,
    event_handler: protocol_module.EventHandler | None,
    checkpoint_handler: protocol_module.CheckpointHandler | None,
    timing_handler: Callable[[_statistics.TimingRecord], None] | None,
) -> tuple[protocol_module.TerminalFrame | None, Exception | None, int]:
    buffer.extend(value)
    while (newline := buffer.find(b"\n")) >= 0:
        raw = bytes(buffer[:newline])
        del buffer[: newline + 1]
        if protocol_error is not None:
            continue
        try:
            if terminal is not None:
                raise ValueError("child wrote a frame after its result")
            frame = protocol_module.decode_reply(
                raw,
                run_id=run_id,
                transitions_remaining=transitions_remaining,
                call_path=call_path,
                project_path=project_path,
            )
        except (TypeError, ValueError) as error:
            protocol_error = error
            continue
        terminal, transitions_remaining = _record_reply(
            frame,
            transitions_remaining,
            event_handler,
            checkpoint_handler,
            timing_handler,
        )
    return terminal, protocol_error, transitions_remaining


def _consume_response(
    reader: _FramedResponseReader,
    *,
    run_id: ids.RunId,
    transitions_remaining: int,
    call_path: pathlib.Path,
    project_path: str,
    event_handler: protocol_module.EventHandler | None,
    checkpoint_handler: protocol_module.CheckpointHandler | None = None,
    timing_handler: Callable[[_statistics.TimingRecord], None] | None,
    check_cancelled: _CancellationCheck | None = None,
    deadline: float | None = None,
) -> tuple[protocol_module.TerminalFrame | None, Exception | None]:
    """Drain frames through the result, preserving the first protocol error."""
    terminal: protocol_module.TerminalFrame | None = None
    protocol_error: Exception | None = None
    buffer = bytearray()
    while terminal is None:
        item = reader.receive(check_cancelled, deadline)
        if isinstance(item, _ResponseEnd):
            break
        if isinstance(item, _ResponseFailure):
            raise item.error
        try:
            terminal, protocol_error, transitions_remaining = (
                _decode_response_chunk(
                    item.value,
                    buffer,
                    terminal,
                    protocol_error,
                    transitions_remaining,
                    run_id=run_id,
                    call_path=call_path,
                    project_path=project_path,
                    event_handler=event_handler,
                    checkpoint_handler=checkpoint_handler,
                    timing_handler=timing_handler,
                )
            )
        except BaseException:
            reader.stop()
            raise
        finally:
            item.consumed.set()
        if terminal is not None:
            reader.stop()
    if buffer and protocol_error is None:
        protocol_error = ValueError("child wrote an incomplete protocol frame")
    return terminal, protocol_error


def _read_response(  # pyright: ignore[reportUnusedFunction]
    stream: BinaryIO,
    *,
    run_id: ids.RunId,
    transitions_remaining: int,
    call_path: pathlib.Path,
    project_path: str,
    event_handler: protocol_module.EventHandler | None,
    checkpoint_handler: protocol_module.CheckpointHandler | None = None,
    timing_handler: Callable[[_statistics.TimingRecord], None] | None,
    check_cancelled: _CancellationCheck | None = None,
    deadline: float | None = None,
) -> tuple[protocol_module.TerminalFrame | None, Exception | None]:
    reader = _FramedResponseReader(stream)
    try:
        return _consume_response(
            reader,
            run_id=run_id,
            transitions_remaining=transitions_remaining,
            call_path=call_path,
            project_path=project_path,
            event_handler=event_handler,
            checkpoint_handler=checkpoint_handler,
            timing_handler=timing_handler,
            check_cancelled=check_cancelled,
            deadline=deadline,
        )
    finally:
        reader.close()


def _wait_for_process(
    process: subprocess.Popen[bytes],
    check_cancelled: _CancellationCheck | None,
    deadline: float | None,
    /,
) -> None:
    while True:
        wait = _cooperate(check_cancelled, deadline)
        if _process.wait_until_reaped(process, wait):
            return


def invoke(
    *,
    project_root: pathlib.Path,
    project_path: str,
    definition_id: ids.GraphId,
    definition_module: str,
    input: object,
    run_id: ids.RunId,
    transitions_remaining: int,
    output_dir: pathlib.Path,
    call_path: pathlib.Path,
    owner_project_path: str,
    params_override: object = _OMITTED,
    event_handler: protocol_module.EventHandler | None = None,
    checkpoint_handler: protocol_module.CheckpointHandler | None = None,
    started_at: float | None = None,
    timing_handler: Callable[[_statistics.TimingRecord], None] | None = None,
    checkpointing: _run_model.CheckpointPolicy = (
        _run_model.CheckpointPolicy.OFF
    ),
    resume_continuation: bytes | None = None,
    retry_incomplete: bool = False,
    check_cancelled: _CancellationCheck | None = None,
    deadline: float | None = None,
) -> tuple[object, int]:
    """Run one child, streaming events, checkpoints, timings, and its budget."""
    _cooperate(check_cancelled, deadline)
    target = _process.resolve_project_root(project_root, project_path)
    owner_path = _process.normalize_project_path(project_path)
    child_environment = _environment(target, definition_id, definition_module)
    interpreter = _interpreter(child_environment)
    if interpreter is None:
        raise RuntimeError(
            f"{child_environment} has no Python interpreter; "
            "run `verdog sync` there "
            "[child_environment_missing]"
        )
    child_project_path = _process.compose_project_path(
        owner_project_path, owner_path
    )
    _register_project_by_value(project_root)
    request = _request(
        definition_id,
        definition_module,
        input,
        run_id,
        transitions_remaining,
        output_dir,
        call_path,
        child_project_path,
        params_override,
        started_at=started_at,
        checkpointing=checkpointing,
        resume_continuation=resume_continuation,
        retry_incomplete=retry_incomplete,
    )

    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["VIRTUAL_ENV"] = str(child_environment)
    path = environment.get("PATH")
    environment["PATH"] = str(interpreter.parent) + (
        os.pathsep + path if path else ""
    )
    group_options, owns_group = _process.process_options(environment)
    process: subprocess.Popen[bytes] = subprocess.Popen(
        (str(interpreter), "-m", "verdog_runtime.child"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=target,
        env=environment,
        **group_options,
    )
    if (
        process.stdin is None or process.stdout is None
    ):  # pragma: no cover - Popen contract
        _process.cleanup_after_interruption(process, owns_group=owns_group)
        raise RuntimeError("child process pipes were not created")
    input_stream = cast(BinaryIO, process.stdin)
    output_stream = cast(BinaryIO, process.stdout)
    reader: _FramedResponseReader | None = None
    writer: _RequestWriter | None = None
    try:
        reader = _FramedResponseReader(output_stream)
        writer = _RequestWriter(input_stream, request)
        writer.wait(check_cancelled, deadline)
        terminal, protocol_error = _consume_response(
            reader,
            run_id=run_id,
            transitions_remaining=transitions_remaining,
            call_path=call_path,
            project_path=child_project_path,
            event_handler=event_handler,
            checkpoint_handler=checkpoint_handler,
            timing_handler=timing_handler,
            check_cancelled=check_cancelled,
            deadline=deadline,
        )
        _wait_for_process(process, check_cancelled, deadline)
        returncode = process.returncode
        if returncode != 0:
            raise RuntimeError(f"{_detail(returncode)} [child_process_failed]")
        if protocol_error is not None or terminal is None:
            error = protocol_error or ValueError("child wrote no result")
            raise RuntimeError(
                f"invalid child response: {error}; {_detail(returncode)} "
                "[child_process_failed]"
            ) from error
    except BaseException:
        _process.cleanup_after_interruption(process, owns_group=owns_group)
        raise
    finally:
        if writer is not None:
            writer.close()
        if reader is not None:
            reader.close()
        if not input_stream.closed:
            input_stream.close()
        output_stream.close()
    if isinstance(terminal, protocol_module.ErrorFrame):
        raise declarations.RemoteWorkflowError(
            terminal.error.exception_type,
            terminal.error.message,
            terminal.error.traceback,
        )
    return terminal.output, terminal.transitions_remaining


def _error_frame(
    error: Exception, transitions_remaining: int, /
) -> protocol_module.ErrorFrame:
    return protocol_module.ErrorFrame(
        error=protocol_module.ErrorDetail(
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error),
            traceback="".join(traceback.format_exception(error)),
        ),
        transitions_remaining=transitions_remaining,
    )


def _bootstrap_error(code: str, message: str) -> protocol_module.ErrorFrame:
    return _error_frame(ValueError(f"{message} [{code}]"), 0)


def _call_paths(
    output_dir: str, call_path: str, /
) -> tuple[pathlib.Path, pathlib.Path]:
    output_root = pathlib.Path(output_dir)
    if not output_root.is_absolute():
        raise ValueError("output directory must be absolute")
    output_root = output_root.resolve()
    if not any(
        (output_root / name).is_file() for name in ("trace.log", "trace")
    ):
        raise ValueError("output directory has no trace")
    call = pathlib.PurePosixPath(call_path)
    if call.is_absolute() or ".." in call.parts or call.as_posix() != call_path:
        raise ValueError("call path must be output-directory-relative")
    native_call_path = pathlib.Path(*call.parts)
    resolved_call_path = (output_root / native_call_path).resolve()
    if not resolved_call_path.is_relative_to(output_root):
        raise ValueError("call path escapes the output directory")
    if not resolved_call_path.is_dir():
        raise ValueError("call path is not an existing directory")
    return output_root, native_call_path


def _event_frame(
    event: ExecutionEvent, remaining: int, /
) -> protocol_module.EventFrame:
    from verdog_runtime.interpreter.execution import NodeExecution

    if isinstance(event, NodeExecution):
        kind = "node"
        entity_id = str(event.node_id)
    else:
        kind = "edge"
        entity_id = str(event.edge_id)
    return protocol_module.EventFrame(
        kind=kind,
        run_id=event.run_id,
        graph_id=event.graph_id,
        entity_id=entity_id,
        status=event.status.value,
        project_path=event.project_path,
        transitions_remaining=remaining,
    )


def _checkpoint_frame(
    run_id: ids.RunId,
    remaining: int,
    emission: _CheckpointEmission,
    compatibility: Mapping[str, str],
    /,
) -> protocol_module.CheckpointFrame:
    if emission.restore_available:
        if emission.payload is None:  # pragma: no cover - interpreter invariant
            raise RuntimeError(
                "restorable child checkpoint has no continuation"
            )
        continuation = protocol_module.encode_binary_payload(
            _child_checkpoint.encode_child_checkpoint(
                _child_checkpoint.ChildCheckpointBundle(
                    compatibility=compatibility,
                    runtime=emission.payload,
                    shards=emission.shards,
                )
            )
        )
    else:
        continuation = None
    return protocol_module.CheckpointFrame(
        run_id=run_id,
        transitions_remaining=remaining,
        kind=emission.kind,
        completed=emission.completed,
        next=emission.next,
        restore_available=emission.restore_available,
        session_branch_available=emission.branch_available,
        continuation=continuation,
        artifact_references=emission.artifact_references,
        unavailable_code=emission.unavailable_code,
        unavailable_reason=emission.unavailable_reason,
    )


def _run_or_resume_child(
    dispatcher: Dispatcher,
    typed_definition: declarations.WorkflowDefinition[
        object, object, object, object
    ],
    graph: GraphDefinition[object, object, object, object],
    scope: CallScope,
    input: object,
    frame: protocol_module.CallFrame,
    budget: Budget,
    output_root: pathlib.Path,
    native_call_path: pathlib.Path,
    params_registry: _ParameterRegistry,
    encoded_output: list[str],
    resume_bundle: _child_checkpoint.ChildCheckpointBundle | None,
    /,
) -> None:
    from verdog_runtime.interpreter._continuation import (
        DefinitionReference,
        decode_continuation,
        fork_continuation,
    )
    from verdog_runtime.interpreter.policies import SessionPolicy

    def check_output_transport(value: object, /) -> bool:
        encoded_output.append(protocol_module.encode_payload(value))
        return True

    if resume_bundle is None:
        dispatcher._run_graph(  # pyright: ignore[reportPrivateUsage]
            graph,
            scope,
            input,
            frame.run_id,
            budget,
            output_root,
            native_call_path,
            frame.project_path,
            params_registry,
            dispatcher._root_resource_arguments,  # pyright: ignore[reportPrivateUsage]
            definition_reference=DefinitionReference(
                kind="subroutine",
                id=graph.id,
                module=typed_definition.entry.definition_module,
                project_path=frame.project_path,
            ),
            check_output_transport=check_output_transport,
        )
        return
    snapshot = decode_continuation(resume_bundle.runtime)
    for pending in resume_bundle.pending_forks:
        snapshot = fork_continuation(
            snapshot,
            run_id=ids.RunId(pending.run_id),
            source_output=pathlib.Path(pending.source_output),
            target_output=pathlib.Path(pending.target_output),
            sessions=SessionPolicy(pending.sessions),
        )
    if snapshot.run_id != frame.run_id:
        raise ValueError("child checkpoint belongs to a different run")
    result, budget.remaining = dispatcher._resume_continuation(  # pyright: ignore[reportPrivateUsage]
        typed_definition,
        snapshot,
        output_root,
        checkpoint_shards=resume_bundle.shards,
        retry_incomplete=frame.retry_incomplete,
        budget=budget,
        root_project_path=frame.project_path,
    )
    encoded_output.append(protocol_module.encode_payload(result.output))


def _child_checkpoint_state(
    frame: protocol_module.CallFrame,
    root: pathlib.Path,
    /,
) -> tuple[Mapping[str, str], _child_checkpoint.ChildCheckpointBundle | None]:
    """Prepare checkpoint identity before loading definitions."""
    if frame.checkpointing == "off":
        if frame.resume_continuation is not None:
            raise ValueError("a child continuation requires checkpointing")
        return {}, None

    # In the child interpreter, sys.prefix's environment marker names this
    # environment's in-process source roots. Compute the fingerprint only once.
    compatibility = _checkpoint_compatibility.checkpoint_compatibility(root)
    if frame.resume_continuation is None:
        return compatibility, None
    bundle = _child_checkpoint.decode_child_checkpoint(
        protocol_module.decode_binary_payload(frame.resume_continuation)
    )
    drift = _checkpoint_compatibility.compatibility_drift(
        bundle.compatibility, compatibility
    )
    if drift is not None:
        raise ValueError(
            "child checkpoint compatibility fingerprints do not match "
            f"the current workflow environment: {drift}"
        )
    return compatibility, bundle


def _serve(
    request: object,
    root: pathlib.Path,
    emit: _EmitFrame,
) -> protocol_module.SuccessFrame[str] | protocol_module.ErrorFrame:
    try:
        frame = protocol_module.decode_call(request)
    except ValueError as error:
        return _bootstrap_error("child_request_invalid", str(error))
    remaining_after_execution = frame.transitions_remaining
    try:
        output_root, native_call_path = _call_paths(
            frame.output_dir, frame.call_path
        )
        child_compatibility, resume_bundle = _child_checkpoint_state(
            frame, root
        )
        definition_type: type[
            declarations.WorkflowDefinition[object, object, object, object]
        ] = declarations.WorkflowDefinition
        typed_definition, _ = _definitions.load_definition(
            root,
            frame.definition_module,
            frame.definition_id,
            definition_type,
            lambda definition: definition.id,
        )
        from verdog_runtime.interpreter._calls import (
            Budget,
            workflow_subroutine,
        )

        root_definition, scope = workflow_subroutine(
            root,
            typed_definition,
        )
        _register_project_by_value(root)
        graph = root_definition.graph
        input = protocol_module.decode_payload(frame.input)
        root_address = (".", graph.id)
        params: Mapping[declarations.ParameterAddress, object] = (
            {
                root_address: protocol_module.decode_payload(
                    frame.params_override
                )
            }
            if frame.params_override is not None
            else {}
        )
        from verdog_runtime.interpreter.execution import (
            Dispatcher,
            _ParameterRegistry,  # pyright: ignore[reportPrivateUsage]
        )

        budget = Budget(frame.transitions_remaining)

        def record_timing(record: _statistics.TimingRecord) -> None:
            emit(protocol_module.TimingFrame(record=record))

        statistics = _statistics.RunStatistics(
            output_root,
            started_at=frame.started_at,
            record_handler=record_timing,
        )

        def record(event: ExecutionEvent) -> None:
            nonlocal remaining_after_execution
            remaining_after_execution = budget.remaining
            emit(_event_frame(event, budget.remaining))

        dispatcher = Dispatcher(
            project_root=root,
            execution_handler=record,
            _statistics=statistics,
        )
        dispatcher._configure(  # pyright: ignore[reportPrivateUsage]
            typed_definition, graph
        )
        dispatcher._checkpointing = frame.checkpointing  # pyright: ignore[reportPrivateUsage]
        dispatcher._checkpoint_by_reference = True  # pyright: ignore[reportPrivateUsage]
        dispatcher._retry_incomplete = frame.retry_incomplete  # pyright: ignore[reportPrivateUsage]
        dispatcher._configure_invocation_journal(  # pyright: ignore[reportPrivateUsage]
            output_root,
            retry_incomplete=frame.retry_incomplete,
        )

        def record_checkpoint(emission: _CheckpointEmission) -> None:
            emit(
                _checkpoint_frame(
                    frame.run_id,
                    budget.remaining,
                    emission,
                    child_compatibility,
                )
            )

        if frame.checkpointing != "off":
            dispatcher._checkpoint_handler = record_checkpoint  # pyright: ignore[reportPrivateUsage]
        params_registry = _ParameterRegistry.create(
            typed_definition.params_types,
            params,
            base=frame.project_path,
        )

        encoded_output: list[str] = []
        _run_or_resume_child(
            dispatcher,
            typed_definition,
            graph,
            scope,
            input,
            frame,
            budget,
            output_root,
            native_call_path,
            params_registry,
            encoded_output,
            resume_bundle,
        )
        remaining_after_execution = budget.remaining
        if (
            not encoded_output
        ):  # pragma: no cover - every successful graph has an exit
            raise RuntimeError("child output was not encoded")
        return protocol_module.SuccessFrame(
            output=encoded_output[0], transitions_remaining=budget.remaining
        )
    except Exception as error:
        error.add_note(f"Verdog workflow process: {frame.definition_id}")
        return _error_frame(error, remaining_after_execution)


def _protocol_output() -> BinaryIO:
    descriptor = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return os.fdopen(descriptor, "wb", buffering=0)


def _write_frame(
    protocol: BinaryIO, frame: protocol_module.OutgoingFrame, /
) -> None:
    protocol.write(protocol_module.encode_frame(frame))


def _exit(_protocol: BinaryIO, returncode: int, /) -> NoReturn:
    os._exit(returncode)


def _child_termination_signal() -> int | None:
    if os.name == "nt":
        value = getattr(signal, "SIGBREAK", None)
        return value if isinstance(value, int) else None
    return signal.SIGTERM


def _serve_until_stopped(
    request: object,
    root: pathlib.Path,
    emit: _EmitFrame,
) -> protocol_module.SuccessFrame[str] | protocol_module.ErrorFrame:
    """Let parent termination unwind the synchronous child call stack."""
    termination_signal = _child_termination_signal()
    # Supported Windows versions expose SIGBREAK.
    if termination_signal is None:  # pragma: no cover
        return _serve(request, root, emit)
    previous = signal.getsignal(termination_signal)

    def stop(_signum: int, _frame: object) -> None:
        raise _ChildStopped

    signal.signal(termination_signal, stop)
    try:
        return _serve(request, root, emit)
    finally:
        signal.signal(termination_signal, previous)


def main() -> NoReturn:
    """Serve one isolated workflow request using the framed child protocol."""
    protocol = _protocol_output()
    try:
        raw = sys.stdin.buffer.readline()
        request = cast(object, json.loads(raw))
        response = _serve_until_stopped(
            request,
            pathlib.Path.cwd().resolve(),
            lambda frame: _write_frame(protocol, frame),
        )
    except _ChildStopped:
        _exit(protocol, 130)
    except Exception as error:
        response = _bootstrap_error("child_request_invalid", str(error))
    _write_frame(protocol, response)
    _exit(protocol, 0)


if __name__ == "__main__":
    main()
