"""Versioned internal transport for run lifecycle operations."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Literal, cast

LIFECYCLE_COMMAND_VERSION = 1
Operation = Literal["resume", "restart", "fork"]
SessionMode = Literal["restore", "branch", "fresh"]
ArgumentMode = Literal["checkpoint", "reused", "overridden"]

_OPERATIONS = frozenset(("resume", "restart", "fork"))
_SESSION_MODES = frozenset(("restore", "branch", "fresh"))
_ARGUMENT_MODES = frozenset(("checkpoint", "reused", "overridden"))
_FIELDS = frozenset(
    (
        "version",
        "operation",
        "root",
        "definition_id",
        "definition_module",
        "source_output",
        "sessions",
        "checkpoint",
        "arguments_mode",
        "retry_incomplete",
        "as_json",
        "arguments",
    )
)


@dataclasses.dataclass(frozen=True, slots=True)
class LifecycleCommand:
    operation: Operation
    root: pathlib.Path
    definition_id: str
    definition_module: str
    source_output: pathlib.Path
    sessions: SessionMode
    checkpoint: int | None
    arguments_mode: ArgumentMode
    retry_incomplete: bool
    as_json: bool
    arguments: tuple[str, ...]


def _string(value: object, label: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid internal lifecycle {label}")
    return value


def _choice(
    value: object,
    choices: frozenset[str],
    label: str,
    /,
) -> str:
    selected = _string(value, label)
    if selected not in choices:
        raise ValueError(f"invalid internal lifecycle {label}")
    return selected


def _path(value: object, label: str, /) -> pathlib.Path:
    raw = _string(value, label)
    path = pathlib.Path(raw)
    if not path.is_absolute():
        raise ValueError(f"invalid internal lifecycle {label}")
    return path.resolve()


def _checkpoint(value: object, /) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError("invalid internal lifecycle checkpoint")
    return value


def _arguments(value: object, /) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("invalid internal lifecycle arguments")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise ValueError("invalid internal lifecycle arguments")
    return tuple(cast(list[str], items))


def _command(value: object, /) -> LifecycleCommand:
    if not isinstance(value, dict):
        raise ValueError("invalid internal lifecycle request")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("invalid internal lifecycle request")
    body = cast(dict[str, object], raw)
    version = body.get("version")
    if (
        frozenset(body) != _FIELDS
        or type(version) is not int
        or version != LIFECYCLE_COMMAND_VERSION
    ):
        raise ValueError("invalid internal lifecycle request")
    retry_incomplete = body["retry_incomplete"]
    as_json = body["as_json"]
    if not isinstance(retry_incomplete, bool) or not isinstance(as_json, bool):
        raise ValueError("invalid internal lifecycle flags")
    return LifecycleCommand(
        operation=cast(
            Operation,
            _choice(body["operation"], _OPERATIONS, "operation"),
        ),
        root=_path(body["root"], "project root"),
        definition_id=_string(body["definition_id"], "definition id"),
        definition_module=_string(
            body["definition_module"], "definition module"
        ),
        source_output=_path(body["source_output"], "source output"),
        sessions=cast(
            SessionMode,
            _choice(body["sessions"], _SESSION_MODES, "session mode"),
        ),
        checkpoint=_checkpoint(body["checkpoint"]),
        arguments_mode=cast(
            ArgumentMode,
            _choice(body["arguments_mode"], _ARGUMENT_MODES, "argument mode"),
        ),
        retry_incomplete=retry_incomplete,
        as_json=as_json,
        arguments=_arguments(body["arguments"]),
    )


def encode_lifecycle_command(command: LifecycleCommand, /) -> str:
    value = {
        "version": LIFECYCLE_COMMAND_VERSION,
        "operation": command.operation,
        "root": str(command.root),
        "definition_id": command.definition_id,
        "definition_module": command.definition_module,
        "source_output": str(command.source_output),
        "sessions": command.sessions,
        "checkpoint": command.checkpoint,
        "arguments_mode": command.arguments_mode,
        "retry_incomplete": command.retry_incomplete,
        "as_json": command.as_json,
        "arguments": list(command.arguments),
    }
    _command(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode_lifecycle_command(payload: str, /) -> LifecycleCommand:
    try:
        value: object = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid internal lifecycle request") from error
    return _command(value)


__all__ = [
    "ArgumentMode",
    "LIFECYCLE_COMMAND_VERSION",
    "LifecycleCommand",
    "Operation",
    "SessionMode",
    "decode_lifecycle_command",
    "encode_lifecycle_command",
]
