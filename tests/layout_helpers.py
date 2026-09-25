"""Reproduce historical output layout when creating compatibility fixtures."""

from dataclasses import fields
from pathlib import Path

from verdog_runtime._statistics import RunStatistics
from verdog_runtime.declarations.ids import GraphId
from verdog_runtime.interpreter._continuation import GraphFrameSnapshot
from verdog_runtime.interpreter.execution import (
    _GraphOutput,  # pyright: ignore[reportPrivateUsage]
    _encoded_id,  # pyright: ignore[reportPrivateUsage]
)


def legacy_graph_create(
    cls: type[_GraphOutput],
    root: Path,
    parent: Path,
    graph_id: GraphId,
    project_path: str,
    statistics: RunStatistics,
    /,
) -> _GraphOutput:
    relative = parent / f"graph-{_encoded_id(str(graph_id))}"
    (root / relative).mkdir(parents=True, exist_ok=False)
    return cls(root, relative, graph_id, project_path, statistics, parent)


def legacy_frame_state(frame: GraphFrameSnapshot, /) -> list[object]:
    return [
        getattr(frame, field.name)
        for field in fields(frame)
        if field.name != "report_path"
    ]
