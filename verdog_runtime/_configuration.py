"""Human-readable values supplied to a runtime visit."""

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields, is_dataclass
from os.path import relpath
from pathlib import Path
from typing import cast
from urllib.parse import quote

from ._markdown import format_cell, format_table
from ._statistics import TimingRecord


@dataclass(frozen=True, slots=True)
class ConfigurationValue:
    name: str
    value: object


def _rows(
    name: str, value: object, ancestors: frozenset[int]
) -> Iterator[tuple[str, object]]:
    if not is_dataclass(value) or isinstance(value, type):
        yield name or "params", value
    elif id(value) in ancestors:
        yield name or "params", "[cycle]"
    else:
        ancestors = ancestors | {id(value)}
        for item in fields(value):
            field_name = f"{name}.{item.name}" if name else item.name
            if not item.repr:
                yield field_name, "[redacted]"
                continue
            try:
                field_value: object = getattr(value, item.name)
            except Exception:
                yield field_name, "[unavailable]"
                continue
            yield from _rows(field_name, field_value, ancestors)


def write_configuration(report_dir: Path, values: Iterable[ConfigurationValue]) -> None:
    rows = (
        row for entry in values for row in _rows(entry.name, entry.value, frozenset())
    )
    table = format_table(rows, ("Parameter", "Value"))
    (report_dir / "config.md").write_text(
        f"# Configuration\n\n{table}\n", encoding="utf-8", errors="backslashreplace"
    )


@dataclass(frozen=True, slots=True)
class _InvocationCall:
    label: str
    directory: Path


_INVOCATION_PARENT = ".verdog-invocation.json"


def _relative_output(root: Path, value: Path, /) -> Path:
    if value.is_absolute():
        raise ValueError("invocation report path must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("invocation report path escapes the run directory")
    return resolved


def register_invocation(
    root: Path,
    child_output: Path,
    parent_graph: Path,
    call_visit: Path,
    graph_id: str | None,
    /,
    *,
    parent_report: Path,
) -> None:
    """Persist a child graph's parentage independently of directory nesting."""

    child_directory = _relative_output(root, child_output)
    _relative_output(root, parent_graph)
    _relative_output(root, parent_report)
    _relative_output(root, call_visit)
    if not call_visit.is_relative_to(parent_graph):
        raise ValueError("invocation call visit is outside its parent graph")
    payload = {
        "parent_graph": parent_graph.as_posix(),
        "call_visit": call_visit.as_posix(),
        "graph_id": graph_id,
        "parent_report": parent_report.as_posix(),
    }
    (child_directory / _INVOCATION_PARENT).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _registered_call(
    root: Path, graph: Path, graph_id: str, /
) -> tuple[Path, _InvocationCall] | None:
    child_directory = _relative_output(root, graph)
    metadata = child_directory / _INVOCATION_PARENT
    if not metadata.is_file():
        # Before node-only output, the graph directory wrapped the node visits
        # and reports/parent metadata lived beside that graph directory.
        child_directory = _relative_output(root, graph.parent)
        metadata = child_directory / _INVOCATION_PARENT
        if not metadata.is_file():
            return None
    raw_value: object = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("invalid invocation report parent metadata")
    raw = cast(dict[object, object], raw_value)
    expected = {"parent_graph", "call_visit", "graph_id"}
    if set(raw) not in (expected, expected | {"parent_report"}):
        raise ValueError("invalid invocation report parent metadata")
    parent_value = raw["parent_graph"]
    call_value = raw["call_visit"]
    registered_graph_id = raw["graph_id"]
    if not isinstance(parent_value, str) or not parent_value:
        raise ValueError("invalid invocation report parent metadata")
    if not isinstance(call_value, str) or not call_value:
        raise ValueError("invalid invocation report parent metadata")
    if registered_graph_id is not None and (
        not isinstance(registered_graph_id, str) or not registered_graph_id
    ):
        raise ValueError("invalid invocation report parent metadata")
    if registered_graph_id is not None and registered_graph_id != graph_id:
        raise ValueError("invocation report graph id does not match its metadata")
    parent_graph = Path(parent_value)
    call_visit = Path(call_value)
    report_value = raw.get("parent_report", parent_graph.parent.as_posix())
    if not isinstance(report_value, str) or not report_value:
        raise ValueError("invalid invocation report parent metadata")
    parent = _relative_output(root, Path(report_value))
    _relative_output(root, call_visit)
    try:
        call_path = call_visit.relative_to(parent_graph).as_posix()
    except ValueError as error:
        raise ValueError(
            "invocation report call visit is outside its parent graph"
        ) from error
    return parent, _InvocationCall(f"{call_path} — {graph_id}", child_directory)


def _call_link(call: _InvocationCall, parent: Path, filename: str) -> str:
    label = re.sub(r"([\\`*_\[\]])", r"\\\1", format_cell(call.label))
    target = Path(relpath(call.directory / filename, start=parent)).as_posix()
    return f"- [{label}]({quote(target, safe='/')})\n"


def _append(report: Path, text: str) -> None:
    with report.open("a", encoding="utf-8", errors="backslashreplace") as stream:
        stream.write(text)


def _calls_heading(report: Path, /) -> str:
    if report.is_file() and "\n## Calls\n\n" in report.read_text(
        encoding="utf-8", errors="replace"
    ):
        return ""
    return "\n## Calls\n\n"


class InvocationReports:
    """Link observed invocations from the root process's timing stream."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._directories: dict[Path, Path] = {}
        self._calls: dict[Path, list[_InvocationCall]] = {}

    def __call__(self, record: TimingRecord) -> None:
        graph = Path(record.path).parent.parent
        if graph in self._directories:
            return
        registered = _registered_call(self.root, graph, record.graph_id)
        directory = (
            registered[1].directory if registered is not None else self.root / graph
        )
        if not (directory / "config.md").is_file():
            directory = self.root / graph.parent
        if not (directory / "config.md").is_file():
            return
        self._directories[graph] = directory
        if registered is None:
            parent_graph = (
                graph.parent.parent.parent
                if directory == self.root / graph.parent
                else graph.parent.parent
            )
            if parent_graph == graph:
                return
            parent = self._directories.get(parent_graph)
            if parent is None:
                return
            call_path = directory.relative_to(parent).as_posix()
            call = _InvocationCall(f"{call_path} — {record.graph_id}", directory)
        else:
            parent, call = registered
        if not (parent / "config.md").is_file():
            return
        calls = self._calls.setdefault(parent, [])
        calls.append(call)
        report = parent / "config.md"
        heading = _calls_heading(report) if len(calls) == 1 else ""
        link = _call_link(call, parent, "config.md")
        if link not in report.read_text(encoding="utf-8", errors="replace"):
            _append(report, heading + link)

    def finish(self) -> None:
        for directory, calls in self._calls.items():
            report = directory / "stats.md"
            if not report.is_file():
                continue
            existing = report.read_text(encoding="utf-8", errors="replace")
            links = "".join(
                link
                for call in calls
                if (call.directory / "stats.md").is_file()
                for link in (_call_link(call, directory, "stats.md"),)
                if link not in existing
            )
            if links:
                _append(report, _calls_heading(report) + links)
