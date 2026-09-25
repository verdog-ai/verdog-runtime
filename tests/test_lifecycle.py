from __future__ import annotations

import json
from pathlib import Path

import pytest

from verdog_runtime._lifecycle import (
    LIFECYCLE_COMMAND_VERSION,
    LifecycleCommand,
    decode_lifecycle_command,
    encode_lifecycle_command,
)


def _value(tmp_path: Path) -> dict[str, object]:
    return {
        "version": LIFECYCLE_COMMAND_VERSION,
        "operation": "restart",
        "root": str(tmp_path.resolve()),
        "definition_id": "example.main",
        "definition_module": "example.workflows.main",
        "source_output": str((tmp_path / "source").resolve()),
        "sessions": "fresh",
        "checkpoint": None,
        "arguments_mode": "overridden",
        "retry_incomplete": False,
        "as_json": True,
        "arguments": ["--input.value", "--verdog-lifecycle", "雪"],
    }


def test_lifecycle_command_round_trips_workflow_arguments_exactly(
    tmp_path: Path,
) -> None:
    command = LifecycleCommand(
        operation="restart",
        root=tmp_path.resolve(),
        definition_id="example.main",
        definition_module="example.workflows.main",
        source_output=(tmp_path / "source").resolve(),
        sessions="fresh",
        checkpoint=None,
        arguments_mode="overridden",
        retry_incomplete=False,
        as_json=True,
        arguments=("--input.value", "--verdog-lifecycle", "雪"),
    )

    assert (
        decode_lifecycle_command(encode_lifecycle_command(command)) == command
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("version", 2),
        ("operation", "run"),
        ("root", "relative"),
        ("source_output", ""),
        ("sessions", "unknown"),
        ("checkpoint", 0),
        ("retry_incomplete", 1),
        ("arguments", ["valid", 3]),
    ],
)
def test_lifecycle_command_rejects_invalid_wire_members(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = _value(tmp_path)
    document[field] = value

    with pytest.raises(ValueError, match="invalid internal lifecycle"):
        decode_lifecycle_command(json.dumps(document))


def test_lifecycle_command_rejects_unknown_members(tmp_path: Path) -> None:
    document = _value(tmp_path)
    document["future"] = True

    with pytest.raises(ValueError, match="invalid internal lifecycle request"):
        decode_lifecycle_command(json.dumps(document))
