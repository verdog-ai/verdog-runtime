"""Typed frames and JSON encoding for the child-process protocol."""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import pathlib
from collections.abc import Callable
from typing import Generic, Literal, TypeAlias, TypeVar, cast, get_args

import cloudpickle

from verdog_runtime import (
    _artifact_references,
    _encoding,
    _run_model,
    _statistics,
)
from verdog_runtime.declarations import ids

VERSION = 13
OutputT = TypeVar("OutputT", covariant=True)
EventStatus: TypeAlias = Literal[
    "pending", "running", "waiting", "succeeded", "failed"
]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CallFrame:
    """Call metadata with payloads decoded only after loading its definition."""

    type: Literal["call"] = dataclasses.field(default="call", init=False)
    definition_id: ids.GraphId
    definition_module: str
    input: str
    run_id: ids.RunId
    transitions_remaining: int
    output_dir: str
    call_path: str
    project_path: str
    started_at: float
    checkpointing: _run_model.CheckpointPolicy = _run_model.CheckpointPolicy.OFF
    retry_incomplete: bool = False
    params_override: str | None = (
        None  # None omits the key; encoded None is a string.
    )
    resume_continuation: str | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EventFrame:
    type: Literal["event"] = dataclasses.field(default="event", init=False)
    kind: Literal["node", "edge"]
    run_id: ids.RunId
    graph_id: ids.GraphId
    entity_id: str
    status: EventStatus
    project_path: str
    transitions_remaining: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TimingFrame:
    type: Literal["timing"] = dataclasses.field(default="timing", init=False)
    record: _statistics.TimingRecord


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointFrame:
    """An opaque child continuation plus its user-facing boundary metadata."""

    type: Literal["checkpoint"] = dataclasses.field(
        default="checkpoint", init=False
    )
    run_id: ids.RunId
    transitions_remaining: int
    kind: _run_model.CheckpointKind
    completed: _run_model.Boundary | None
    next: _run_model.Boundary | None
    restore_available: bool
    session_branch_available: bool
    continuation: str | None
    artifact_references: dict[str, object] | None
    unavailable_code: str | None = None
    unavailable_reason: str | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SuccessFrame(Generic[OutputT]):
    """Carries an encoded string when sent and decoded output when received."""

    type: Literal["result"] = dataclasses.field(default="result", init=False)
    outcome: Literal["success"] = dataclasses.field(
        default="success", init=False
    )
    output: OutputT
    transitions_remaining: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ErrorDetail:
    exception_type: str
    message: str
    traceback: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ErrorFrame:
    type: Literal["result"] = dataclasses.field(default="result", init=False)
    outcome: Literal["error"] = dataclasses.field(default="error", init=False)
    error: ErrorDetail
    transitions_remaining: int


ReplyFrame: TypeAlias = (
    EventFrame
    | TimingFrame
    | CheckpointFrame
    | SuccessFrame[object]
    | ErrorFrame
)
TerminalFrame: TypeAlias = SuccessFrame[object] | ErrorFrame
OutgoingFrame: TypeAlias = (
    CallFrame
    | EventFrame
    | TimingFrame
    | CheckpointFrame
    | SuccessFrame[str]
    | ErrorFrame
)
EventHandler: TypeAlias = Callable[[EventFrame], None]
CheckpointHandler: TypeAlias = Callable[[CheckpointFrame], None]


def encode_payload(value: object, /) -> str:
    return base64.b64encode(cloudpickle.dumps(value)).decode("ascii")


def decode_payload(value: object, /) -> object:
    if not isinstance(value, str):
        raise ValueError("invalid child payload")
    try:
        return cloudpickle.loads(base64.b64decode(value, validate=True))
    except Exception as error:
        raise ValueError("invalid child payload") from error


def encode_binary_payload(value: bytes, /) -> str:
    return _encoding.encode_base64(value)


def decode_binary_payload(value: object, /) -> bytes:
    if not isinstance(value, str):
        raise ValueError("invalid child binary payload")
    try:
        return _encoding.decode_base64(value)
    except ValueError as error:
        raise ValueError("invalid child binary payload") from error


def seconds(value: object, /) -> float:
    if type(value) is not int and type(value) is not float:
        raise ValueError("invalid child time")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("invalid child time") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError("invalid child time")
    return result


def wire_project_path(value: object, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid child event project path")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("invalid child event project path")
    return value


def encode_frame(frame: OutgoingFrame, /) -> bytes:
    data: dict[str, object] = {"version": VERSION, **dataclasses.asdict(frame)}
    if isinstance(frame, CallFrame) and frame.params_override is None:
        del data["params_override"]
    if isinstance(frame, CallFrame) and frame.resume_continuation is None:
        del data["resume_continuation"]
    return (
        json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def decode_call(request: object, /) -> CallFrame:
    if not isinstance(request, dict):
        raise ValueError("request is not an object")
    frame = cast(dict[str, object], request)
    if frame.get("version") != VERSION or frame.get("type") != "call":
        raise ValueError("unsupported protocol frame")
    definition_id = frame.get("definition_id")
    definition_module = frame.get("definition_module")
    run_id = frame.get("run_id")
    remaining = frame.get("transitions_remaining")
    output_dir = frame.get("output_dir")
    call_path = frame.get("call_path")
    project_path = frame.get("project_path")
    checkpointing = frame.get("checkpointing")
    retry_incomplete = frame.get("retry_incomplete")
    if (
        not isinstance(definition_id, str)
        or not definition_id
        or not isinstance(definition_module, str)
        or not definition_module
        or any(
            not ids.is_valid_entity_id(part)
            for part in definition_module.split(".")
        )
        or not isinstance(run_id, str)
        or not run_id
        or type(remaining) is not int
        or remaining < 0
        or not isinstance(output_dir, str)
        or not isinstance(call_path, str)
        or not isinstance(project_path, str)
        or not isinstance(checkpointing, str)
        or not isinstance(retry_incomplete, bool)
    ):
        raise ValueError("request fields are invalid")
    try:
        checkpoint_policy = _run_model.CheckpointPolicy(checkpointing)
    except ValueError as error:
        raise ValueError("request fields are invalid") from error
    input_payload = frame.get("input")
    if not isinstance(input_payload, str):
        raise ValueError("invalid child payload")
    params_override = None
    if "params_override" in frame:
        params_override = frame["params_override"]
        if not isinstance(params_override, str):
            raise ValueError("invalid child payload")
    resume_continuation = None
    if "resume_continuation" in frame:
        resume_continuation = frame["resume_continuation"]
        decode_binary_payload(resume_continuation)
    return CallFrame(
        definition_id=ids.GraphId(definition_id),
        definition_module=definition_module,
        input=input_payload,
        run_id=ids.RunId(run_id),
        transitions_remaining=remaining,
        output_dir=output_dir,
        call_path=call_path,
        project_path=wire_project_path(project_path),
        started_at=seconds(frame.get("started_at")),
        checkpointing=checkpoint_policy,
        retry_incomplete=retry_incomplete,
        params_override=params_override,
        resume_continuation=cast(str | None, resume_continuation),
    )


def _decoded_frame(raw: bytes, /) -> dict[str, object]:
    decoded = cast(object, json.loads(raw))
    if not isinstance(decoded, dict):
        raise ValueError("child response is not an object")
    frame = cast(dict[str, object], decoded)
    if frame.get("version") != VERSION:
        raise ValueError("child protocol version is invalid")
    return frame


def _remaining(frame: dict[str, object], ceiling: int, /) -> int:
    remaining = frame.get("transitions_remaining")
    if type(remaining) is not int or not 0 <= remaining <= ceiling:
        raise ValueError("child returned an invalid transition budget")
    return remaining


def _event(
    frame: dict[str, object],
    expected_run_id: ids.RunId,
    transitions_remaining: int,
    /,
) -> EventFrame:
    if set(frame) != {item.name for item in dataclasses.fields(EventFrame)} | {
        "version"
    }:
        raise ValueError("invalid child event")
    kind = frame["kind"]
    run_id = frame["run_id"]
    graph_id = frame["graph_id"]
    entity_id = frame["entity_id"]
    status = frame["status"]
    if (
        not isinstance(kind, str)
        or kind not in {"node", "edge"}
        or not isinstance(run_id, str)
        or run_id != str(expected_run_id)
        or not isinstance(graph_id, str)
        or not graph_id
        or not isinstance(entity_id, str)
        or not entity_id
        or not isinstance(status, str)
        or status not in get_args(EventStatus)
    ):
        raise ValueError("invalid child event")
    return EventFrame(
        kind=cast(Literal["node", "edge"], kind),
        run_id=ids.RunId(run_id),
        graph_id=ids.GraphId(graph_id),
        entity_id=entity_id,
        status=cast(EventStatus, status),
        project_path=wire_project_path(frame["project_path"]),
        transitions_remaining=_remaining(frame, transitions_remaining),
    )


def _timing(
    frame: dict[str, object], call_path: pathlib.Path, project_path: str, /
) -> TimingFrame:
    if set(frame) != {item.name for item in dataclasses.fields(TimingFrame)} | {
        "version"
    }:
        raise ValueError("invalid child timing frame")
    value = frame["record"]
    if not isinstance(value, dict):
        raise ValueError("invalid child timing record")
    data = cast(dict[str, object], value)
    if set(data) != {
        item.name for item in dataclasses.fields(_statistics.TimingRecord)
    }:
        raise ValueError("invalid child timing record")
    for name in ("path", "project_path", "graph_id", "node_id", "node_type"):
        if not isinstance(data[name], str) or not data[name]:
            raise ValueError("invalid child timing identity")
    if data["status"] not in get_args(_statistics.TimingStatus):
        raise ValueError("invalid child timing status")
    record = _statistics.TimingRecord(
        path=wire_project_path(data["path"]),
        project_path=wire_project_path(data["project_path"]),
        graph_id=cast(str, data["graph_id"]),
        node_id=cast(str, data["node_id"]),
        node_type=cast(str, data["node_type"]),
        status=cast(_statistics.TimingStatus, data["status"]),
        duration_seconds=seconds(data["duration_seconds"]),
    )
    if not pathlib.PurePosixPath(record.path).is_relative_to(
        pathlib.PurePosixPath(call_path.as_posix())
    ):
        raise ValueError("child timing path escapes its call")
    if not pathlib.PurePosixPath(record.project_path).is_relative_to(
        pathlib.PurePosixPath(project_path)
    ):
        raise ValueError("child timing project path escapes its owner")
    return TimingFrame(record=record)


def _error_detail(value: object, /) -> ErrorDetail:
    if not isinstance(value, dict):
        raise ValueError("invalid child error")
    error = cast(dict[str, object], value)
    if set(error) != {item.name for item in dataclasses.fields(ErrorDetail)}:
        raise ValueError("invalid child error")
    exception_type = error["exception_type"]
    message = error["message"]
    remote_traceback = error["traceback"]
    if (
        not isinstance(exception_type, str)
        or not isinstance(message, str)
        or not isinstance(remote_traceback, str)
    ):
        raise ValueError("invalid child error")
    return ErrorDetail(
        exception_type=exception_type,
        message=message,
        traceback=remote_traceback,
    )


def _success(
    frame: dict[str, object], remaining: int, /
) -> SuccessFrame[object]:
    if set(frame) != {
        item.name for item in dataclasses.fields(SuccessFrame)
    } | {"version"}:
        raise ValueError("invalid child result")
    return SuccessFrame(
        output=decode_payload(frame["output"]), transitions_remaining=remaining
    )


def _error(frame: dict[str, object], remaining: int, /) -> ErrorFrame:
    if set(frame) != {item.name for item in dataclasses.fields(ErrorFrame)} | {
        "version"
    }:
        raise ValueError("invalid child result")
    return ErrorFrame(
        error=_error_detail(frame["error"]), transitions_remaining=remaining
    )


def _checkpoint_boundary(
    value: object, project_path: str, call_path: pathlib.Path, /
) -> _run_model.Boundary | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid child checkpoint boundary")
    boundary = cast(dict[str, object], value)
    if set(boundary) != {
        item.name for item in dataclasses.fields(_run_model.Boundary)
    }:
        raise ValueError("invalid child checkpoint boundary")
    boundary_project = wire_project_path(boundary["project_path"])
    boundary_call = wire_project_path(boundary["call_path"])
    if not pathlib.PurePosixPath(boundary_project).is_relative_to(
        pathlib.PurePosixPath(project_path)
    ):
        raise ValueError("child checkpoint project path escapes its owner")
    if not pathlib.PurePosixPath(boundary_call).is_relative_to(
        pathlib.PurePosixPath(call_path.as_posix())
    ):
        raise ValueError("child checkpoint path escapes its call")
    graph = boundary["graph"]
    node = boundary["node"]
    visit = boundary["visit"]
    if (
        not isinstance(graph, str)
        or not graph
        or not isinstance(node, str)
        or not node
        or type(visit) is not int
        or visit <= 0
    ):
        raise ValueError("invalid child checkpoint boundary")
    return _run_model.Boundary(
        project_path=boundary_project,
        graph=graph,
        node=node,
        visit=visit,
        call_path=boundary_call,
    )


def _checkpoint(
    frame: dict[str, object],
    expected_run_id: ids.RunId,
    transitions_remaining: int,
    call_path: pathlib.Path,
    project_path: str,
    /,
) -> CheckpointFrame:
    if set(frame) != {
        item.name for item in dataclasses.fields(CheckpointFrame)
    } | {"version"}:
        raise ValueError("invalid child checkpoint")
    run_id = frame["run_id"]
    kind = frame["kind"]
    restore_available = frame["restore_available"]
    branch_available = frame["session_branch_available"]
    continuation = frame["continuation"]
    artifact_references = frame["artifact_references"]
    unavailable_code = frame["unavailable_code"]
    unavailable_reason = frame["unavailable_reason"]
    if (
        not isinstance(run_id, str)
        or run_id != str(expected_run_id)
        or not isinstance(kind, str)
        or not isinstance(restore_available, bool)
        or not isinstance(branch_available, bool)
        or (
            unavailable_code is not None
            and not isinstance(unavailable_code, str)
        )
        or (
            unavailable_reason is not None
            and not isinstance(unavailable_reason, str)
        )
    ):
        raise ValueError("invalid child checkpoint")
    try:
        checkpoint_kind = _run_model.CheckpointKind(kind)
    except ValueError as error:
        raise ValueError("invalid child checkpoint") from error
    if restore_available:
        decode_binary_payload(continuation)
        if (
            _artifact_references.decode_artifact_references(
                artifact_references, pathlib.Path("child checkpoint")
            )
            is None
        ):
            raise ValueError(
                "restorable child checkpoint has no artifact references"
            )
        if unavailable_code is not None or unavailable_reason is not None:
            raise ValueError("invalid child checkpoint")
    elif continuation is not None or artifact_references is not None:
        raise ValueError("invalid child checkpoint")
    return CheckpointFrame(
        run_id=ids.RunId(run_id),
        transitions_remaining=_remaining(frame, transitions_remaining),
        kind=checkpoint_kind,
        completed=_checkpoint_boundary(
            frame["completed"], project_path, call_path
        ),
        next=_checkpoint_boundary(frame["next"], project_path, call_path),
        restore_available=restore_available,
        session_branch_available=branch_available,
        continuation=cast(str | None, continuation),
        artifact_references=cast(dict[str, object] | None, artifact_references),
        unavailable_code=unavailable_code,
        unavailable_reason=unavailable_reason,
    )


def decode_reply(
    raw: bytes,
    /,
    *,
    run_id: ids.RunId,
    transitions_remaining: int,
    call_path: pathlib.Path,
    project_path: str,
) -> ReplyFrame:
    frame = _decoded_frame(raw)
    if frame.get("type") == "event":
        return _event(frame, run_id, transitions_remaining)
    if frame.get("type") == "timing":
        return _timing(frame, call_path, project_path)
    if frame.get("type") == "checkpoint":
        return _checkpoint(
            frame, run_id, transitions_remaining, call_path, project_path
        )
    if frame.get("type") == "result":
        remaining = _remaining(frame, transitions_remaining)
        outcome = frame.get("outcome")
        if outcome == "success":
            return _success(frame, remaining)
        if outcome == "error":
            return _error(frame, remaining)
        raise ValueError("child result has no outcome")
    raise ValueError("child frame type is invalid")
