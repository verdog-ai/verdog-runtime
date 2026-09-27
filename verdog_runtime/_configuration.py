"""Human-readable values supplied to a runtime visit."""

import dataclasses
import html
import itertools
import json
import os.path
import pathlib
import re
import urllib.parse
from collections.abc import Iterable, Iterator
from typing import cast

from verdog_runtime import _markdown, _statistics


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigurationValue:
    name: str
    value: object


def _rows(
    name: str, value: object, ancestors: frozenset[int]
) -> Iterator[tuple[str, object]]:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        yield name or "params", value
    elif id(value) in ancestors:
        yield name or "params", "[cycle]"
    else:
        ancestors = ancestors | {id(value)}
        for item in dataclasses.fields(value):
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


def write_configuration(
    report_dir: pathlib.Path, values: Iterable[ConfigurationValue]
) -> None:
    rows = itertools.chain.from_iterable(
        _rows(entry.name, entry.value, frozenset()) for entry in values
    )
    table = _markdown.format_table(rows, ("Parameter", "Value"))
    (report_dir / "config.md").write_text(
        f"# Configuration\n\n{table}\n",
        encoding="utf-8",
        errors="backslashreplace",
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _InvocationCall:
    label: str
    directory: pathlib.Path
    parent_visit_index: int | None = None


_INVOCATION_PARENT = ".verdog-invocation.json"
_CALLS_MARKER = "\n## Calls\n\n"
_NAVIGATION = re.compile(
    r"\A<!-- verdog-navigation -->\n.*?\n<!-- /verdog-navigation -->\n\n",
    re.DOTALL,
)


def _relative_output(
    root: pathlib.Path, value: pathlib.Path, /
) -> pathlib.Path:
    if value.is_absolute():
        raise ValueError("invocation report path must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("invocation report path escapes the run directory")
    return resolved


def register_invocation(
    root: pathlib.Path,
    child_output: pathlib.Path,
    parent_graph: pathlib.Path,
    call_visit: pathlib.Path,
    graph_id: str | None,
    /,
    *,
    parent_report: pathlib.Path,
    parent_visit_index: int | None = None,
) -> None:
    """Persist a child graph's parentage independently of directory nesting."""
    child_directory = _relative_output(root, child_output)
    _relative_output(root, parent_graph)
    _relative_output(root, parent_report)
    _relative_output(root, call_visit)
    if not call_visit.is_relative_to(parent_graph):
        raise ValueError("invocation call visit is outside its parent graph")
    if parent_visit_index is not None and (
        type(parent_visit_index) is not int or parent_visit_index <= 0
    ):
        raise ValueError("invalid invocation parent visit index")
    metadata = child_directory / _INVOCATION_PARENT
    if metadata.exists() or metadata.is_symlink():
        parent, _ = _read_registered_call(root, child_directory, graph_id)
        existing = json.loads(metadata.read_text(encoding="utf-8"))
        if (
            parent != _relative_output(root, parent_report)
            or existing["parent_graph"] != parent_graph.as_posix()
            or existing["call_visit"] != call_visit.as_posix()
        ):
            raise ValueError("invocation report parent metadata changed")
        # Metadata is a checkpointed artifact. Resume must not rewrite it.
        return
    payload: dict[str, object] = {
        "parent_graph": parent_graph.as_posix(),
        "call_visit": call_visit.as_posix(),
        "graph_id": graph_id,
        "parent_report": parent_report.as_posix(),
    }
    if parent_visit_index is not None:
        payload["parent_visit_index"] = parent_visit_index
    metadata.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _registered_call(
    root: pathlib.Path, graph: pathlib.Path, graph_id: str, /
) -> tuple[pathlib.Path, _InvocationCall] | None:
    child_directory = _relative_output(root, graph)
    metadata = child_directory / _INVOCATION_PARENT
    if not metadata.is_file():
        # Before node-only output, the graph directory wrapped the node visits
        # and reports/parent metadata lived beside that graph directory.
        child_directory = _relative_output(root, graph.parent)
        metadata = child_directory / _INVOCATION_PARENT
        if not metadata.is_file():
            return None
    return _read_registered_call(root, child_directory, graph_id)


def _read_registered_call(
    root: pathlib.Path,
    child_directory: pathlib.Path,
    graph_id: str | None = None,
    /,
) -> tuple[pathlib.Path, _InvocationCall]:
    metadata = child_directory / _INVOCATION_PARENT
    if metadata.is_symlink():
        raise ValueError("invocation report metadata must not be a symlink")
    raw_value: object = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("invalid invocation report parent metadata")
    raw = cast(dict[object, object], raw_value)
    expected = {"parent_graph", "call_visit", "graph_id"}
    if (
        not expected
        <= set(raw)
        <= expected | {"parent_report", "parent_visit_index"}
    ):
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
    if (
        registered_graph_id is not None
        and graph_id is not None
        and registered_graph_id != graph_id
    ):
        raise ValueError(
            "invocation report graph id does not match its metadata"
        )
    parent_graph = pathlib.Path(parent_value)
    _relative_output(root, parent_graph)
    call_visit = pathlib.Path(call_value)
    report_value = raw.get("parent_report", parent_graph.parent.as_posix())
    if not isinstance(report_value, str) or not report_value:
        raise ValueError("invalid invocation report parent metadata")
    parent = _relative_output(root, pathlib.Path(report_value))
    if parent == child_directory:
        raise ValueError("invocation report cannot be its own parent")
    _relative_output(root, call_visit)
    try:
        call_path = call_visit.relative_to(parent_graph).as_posix()
    except ValueError as error:
        raise ValueError(
            "invocation report call visit is outside its parent graph"
        ) from error
    index = raw.get("parent_visit_index")
    if index is not None and (type(index) is not int or index <= 0):
        raise ValueError("invalid invocation parent visit index")
    selected_id = graph_id or registered_graph_id
    label = call_path if selected_id is None else f"{call_path} — {selected_id}"
    return parent, _InvocationCall(label, child_directory, index)


def _call_link(
    call: _InvocationCall, parent: pathlib.Path, filename: str
) -> str:
    return f"- {_report_link(call.label, call.directory / filename, parent)}\n"


def _report_link(label: str, report: pathlib.Path, parent: pathlib.Path) -> str:
    label = re.sub(r"([\\`*_\[\]])", r"\\\1", _markdown.format_cell(label))
    target = pathlib.Path(os.path.relpath(report, start=parent)).as_posix()
    return f"[{label}]({urllib.parse.quote(target, safe='/')})"


def _stored_calls(
    root: pathlib.Path, report: pathlib.Path
) -> list[_InvocationCall]:
    if report.is_symlink():
        raise ValueError("invocation report must not be a symlink")
    if not report.is_file():
        return []
    text = report.read_text(encoding="utf-8", errors="replace")
    if _CALLS_MARKER not in text:
        return []
    section = text.split(_CALLS_MARKER, 1)[1].split("\n## ", 1)[0]
    calls: list[_InvocationCall] = []
    for match in re.finditer(r"^- \[(.*)\]\((.*)\)$", section, re.MULTILINE):
        label, target = match.group(1, 2)
        url = urllib.parse.urlsplit(target)
        if url.scheme or url.netloc or url.query or url.fragment:
            raise ValueError("invocation report link must be a relative file")
        relative = pathlib.Path(urllib.parse.unquote(url.path))
        if relative.is_absolute() or relative.name not in (
            "config.md",
            "stats.md",
        ):
            raise ValueError("invalid invocation report link")
        directory = _relative_output(
            root, report.parent.relative_to(root) / relative.parent
        )
        if directory == report.parent:
            raise ValueError("invocation report cannot be its own parent")
        label = html.unescape(label.replace("<br>", "\n"))
        label = re.sub(r"\\([\\`*_\[\]])", r"\1", label)
        calls.append(_InvocationCall(label, directory))
    return calls


def _append(report: pathlib.Path, text: str) -> None:
    with report.open(
        "a", encoding="utf-8", errors="backslashreplace"
    ) as stream:
        stream.write(text)


def _calls_heading(report: pathlib.Path, /) -> str:
    if report.is_file() and "\n## Calls\n\n" in report.read_text(
        encoding="utf-8", errors="replace"
    ):
        return ""
    return "\n## Calls\n\n"


class InvocationReports:
    """Link observed invocations from the root process's timing stream."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root.resolve()
        self._directories: dict[pathlib.Path, pathlib.Path] = {}
        self._calls: dict[pathlib.Path, list[_InvocationCall]] = {}
        self._parents: dict[pathlib.Path, pathlib.Path] = {}
        self._restore()

    def _remember(self, parent: pathlib.Path, call: _InvocationCall) -> None:
        if call.directory == self.root:
            raise ValueError("invocation report root cannot have a parent")
        previous_parent = self._parents.get(call.directory)
        if previous_parent is not None and previous_parent != parent:
            raise ValueError("invocation report has conflicting parents")
        ancestor: pathlib.Path | None = parent
        ancestors = {call.directory}
        while ancestor is not None:
            if ancestor in ancestors:
                raise ValueError("invocation report parentage contains a cycle")
            ancestors.add(ancestor)
            ancestor = self._parents.get(ancestor)
        self._parents[call.directory] = parent
        calls = self._calls.setdefault(parent, [])
        for position, existing in enumerate(calls):
            if existing.directory == call.directory:
                calls[position] = call
                return
        calls.append(call)

    def _restore(self) -> None:
        registered: dict[
            pathlib.Path, tuple[pathlib.Path, _InvocationCall]
        ] = {}
        for directory, children, filenames in self.root.walk():
            children[:] = sorted(name for name in children if name != ".verdog")
            if _INVOCATION_PARENT in filenames:
                parent, call = _read_registered_call(self.root, directory)
                registered[directory] = parent, call
        pending = [self.root, *registered]
        seen: set[pathlib.Path] = set()
        while pending:
            directory = pending.pop()
            if directory in seen:
                continue
            seen.add(directory)
            for filename in ("config.md", "stats.md"):
                for stored in _stored_calls(self.root, directory / filename):
                    registration = registered.get(stored.directory)
                    if registration is not None:
                        parent, call = registration
                        if parent != directory:
                            raise ValueError(
                                "invocation report has conflicting parents"
                            )
                        stored = dataclasses.replace(call, label=stored.label)
                    self._remember(directory, stored)
                    pending.append(stored.directory)
        for parent, call in registered.values():
            if call.directory not in self._parents:
                self._remember(parent, call)

    def _ordered_calls(self, parent: pathlib.Path) -> list[_InvocationCall]:
        calls = self._calls.get(parent, [])
        # Legacy Calls lists retain observed order; unlisted metadata was
        # loaded in path order. New registrations carry execution order.
        positions = {
            call.directory: position for position, call in enumerate(calls)
        }
        return sorted(
            calls,
            key=lambda call: (
                call.parent_visit_index is not None,
                positions[call.directory]
                if call.parent_visit_index is None
                else call.parent_visit_index,
                call.directory.as_posix(),
            ),
        )

    def __call__(self, record: _statistics.TimingRecord) -> None:
        graph = pathlib.Path(record.path).parent.parent
        if graph in self._directories:
            return
        registered = _registered_call(self.root, graph, record.graph_id)
        directory = (
            registered[1].directory
            if registered is not None
            else self.root / graph
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
            call = _InvocationCall(
                f"{call_path} — {record.graph_id}", directory
            )
        else:
            parent, call = registered
        if not (parent / "config.md").is_file():
            return
        self._remember(parent, call)
        report = parent / "config.md"
        link = _call_link(call, parent, "config.md")
        if call.directory not in {
            stored.directory for stored in _stored_calls(self.root, report)
        }:
            _append(report, _calls_heading(report) + link)

    def finish(self) -> None:
        self._restore()
        directories = {self.root, *self._parents, *self._calls}
        available = {
            directory
            for directory in directories
            if (directory / "stats.md").is_file()
        }
        siblings = {
            parent: [
                call
                for call in self._ordered_calls(parent)
                if call.directory in available
            ]
            for parent in self._calls
        }
        for directory in sorted(available):
            report = directory / "stats.md"
            existing = report.read_text(encoding="utf-8", errors="replace")
            text = _NAVIGATION.sub("", existing, count=1)
            if _CALLS_MARKER in text:
                before, after = text.split(_CALLS_MARKER, 1)
                following = after.partition("\n## ")
                text = before + (
                    following[1] + following[2] if following[1] else "\n"
                )
            calls = siblings.get(directory, [])
            if calls:
                text = (
                    text.rstrip()
                    + "\n"
                    + _CALLS_MARKER
                    + "".join(
                        _call_link(call, directory, "stats.md")
                        for call in calls
                    )
                )
            navigation: list[str] = []
            parent = self._parents.get(directory)
            if parent is not None:
                calls = siblings[parent]
                position = next(
                    i
                    for i, call in enumerate(calls)
                    if call.directory == directory
                )
                if position:
                    navigation.append(
                        _report_link(
                            "← Previous call",
                            calls[position - 1].directory / "stats.md",
                            directory,
                        )
                    )
                if parent in available:
                    navigation.append(
                        _report_link("↑ Parent", parent / "stats.md", directory)
                    )
                if position + 1 < len(calls):
                    navigation.append(
                        _report_link(
                            "Next call →",
                            calls[position + 1].directory / "stats.md",
                            directory,
                        )
                    )
            if navigation:
                text = (
                    "<!-- verdog-navigation -->\n"
                    + " · ".join(navigation)
                    + "\n<!-- /verdog-navigation -->\n\n"
                    + text
                )
            if text != existing:
                report.write_text(
                    text, encoding="utf-8", errors="backslashreplace"
                )
