"""Attribute durable provider attempts to the existing statistics hierarchy."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import stat
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import TypeAlias, cast

from verdog_runtime import _run_metadata, _run_model

NodeIdentity: TypeAlias = tuple[str, str, str, str]
_COLUMNS = ("input_tokens", "cached_input_tokens", "output_tokens", "cost_usd")
HEADERS = (
    "Agent calls",
    "Input tokens",
    "Cached input",
    "Output tokens",
    "Est. USD",
)


@dataclasses.dataclass(slots=True)
class _Amount:
    total: Decimal = Decimal(0)
    known: int = 0
    unknown: int = 0

    def include(self, value: object, *, cost: bool) -> None:
        valid = isinstance(value, str) if cost else type(value) is int
        try:
            amount = Decimal(str(value)) if valid else None
        except InvalidOperation:
            amount = None
        if amount is None or not amount.is_finite() or amount < 0:
            self.unknown += 1
        else:
            self.total += amount
            self.known += 1

    def merge(self, other: _Amount) -> None:
        self.total += other.total
        self.known += other.known
        self.unknown += other.unknown

    def render(self, *, cost: bool) -> str:
        if self.unknown and not self.known:
            return "unknown"
        if cost:
            value = (
                "<0.000001"
                if 0 < self.total < Decimal("0.000001")
                else f"{self.total:.6f}"
            )
        else:
            value = str(int(self.total))
        return f"{value} + ?" if self.unknown else value


@dataclasses.dataclass(slots=True)
class UsageTotals:
    """Known sums and missing measurements, independent of timing rollback."""

    calls: int = 0
    amounts: dict[str, _Amount] = dataclasses.field(
        default_factory=lambda: {column: _Amount() for column in _COLUMNS}
    )

    def include(self, usage: Mapping[str, object] | None, /) -> None:
        """Include one actual attempt, including failed or unmeasured calls."""
        self.calls += 1
        for column, amount in self.amounts.items():
            amount.include(
                None if usage is None else usage.get(column),
                cost=column == "cost_usd",
            )

    def merge(self, other: UsageTotals, /) -> None:
        """Accumulate disjoint node rows without losing unknown values."""
        self.calls += other.calls
        for column, amount in self.amounts.items():
            amount.merge(other.amounts[column])

    def cells(self, node_type: str | None = None) -> tuple[str, ...]:
        """Distinguish inapplicable, missing, and partially measured usage."""
        applicable = node_type is None or node_type in (
            "agent",
            "subroutine_call",
            "workflow_call",
        )
        return (
            str(self.calls),
            *(
                self.amounts[column].render(cost=column == "cost_usd")
                if applicable
                else "—"
                for column in _COLUMNS
            ),
        )


def _object(path: pathlib.Path) -> dict[str, object] | None:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return None
        value: object = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _relative(value: object) -> pathlib.Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = pathlib.Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        return None
    return path


@dataclasses.dataclass(frozen=True, slots=True)
class _Parent:
    report: pathlib.Path
    visit: pathlib.Path


def _inventory(
    root: pathlib.Path,
) -> tuple[dict[pathlib.Path, _Parent], list[pathlib.Path]]:
    parents: dict[pathlib.Path, _Parent] = {}
    attempts: list[pathlib.Path] = []
    for directory, children, files in os.walk(root, followlinks=False):
        base = pathlib.Path(directory)
        children[:] = sorted(
            name
            for name in children
            if name != ".verdog" and not (base / name).is_symlink()
        )
        if ".verdog-invocation.json" in files:
            raw = _object(base / ".verdog-invocation.json")
            if raw is not None:
                graph = _relative(raw.get("parent_graph"))
                visit = _relative(raw.get("call_visit"))
                report = _relative(
                    raw.get(
                        "parent_report",
                        None if graph is None else graph.parent.as_posix(),
                    )
                )
                if (
                    graph is not None
                    and visit is not None
                    and report is not None
                    and (graph == report or graph.parent == report)
                    and visit.parent.parent == graph
                    and visit.name.isdecimal()
                ):
                    parents[base.relative_to(root)] = _Parent(report, visit)
        if (
            base.name == "invocations"
            and base.parent.name.isdecimal()
            and base != root
            and base.parent != root
            and base.parent.relative_to(root) not in parents
        ):
            found = {
                name
                for name in children
                if name.isdecimal()
                and (base / name / "prompt.txt").is_file()
                and not (base / name / "prompt.txt").is_symlink()
            }
            attempts.extend(base / name for name in children if name in found)
            # A nested graph may itself have a node named "invocations".
            # Only actual attempt directories hide authored provider artifacts.
            children[:] = [name for name in children if name not in found]
    return parents, attempts


def _owner(
    visit: pathlib.Path,
    report: pathlib.Path,
    graph: pathlib.Path,
    parents: Mapping[pathlib.Path, _Parent],
) -> pathlib.Path | None:
    if visit.parent.parent == graph:
        return report
    # Metadata locates flat legacy activations as well as nested call outputs.
    for candidate in sorted(
        parents, key=lambda path: len(path.parts), reverse=True
    ):
        try:
            parts = visit.relative_to(candidate).parts
        except ValueError:
            continue
        if len(parts) in (2, 3):
            return candidate
    return None


def _identity(
    owner: pathlib.Path,
    visit: pathlib.Path,
    report: pathlib.Path,
    parents: Mapping[pathlib.Path, _Parent],
    node_paths: Mapping[str, NodeIdentity],
) -> NodeIdentity | None:
    seen: set[pathlib.Path] = set()
    while owner != report:
        if owner in seen or owner not in parents:
            return None
        seen.add(owner)
        parent = parents[owner]
        owner, visit = parent.report, parent.visit
    return node_paths.get(visit.parent.as_posix())


def _inherited_attempts(root: pathlib.Path, run_id: str) -> set[pathlib.Path]:
    """Recognize copied legacy attempts even when they lack usage records."""
    try:
        header = _run_metadata.load_run_header(root)
        if (
            header.id != run_id
            or header.parent is None
            or header.parent.operation != "fork"
        ):
            return set()
        directory = (
            root
            / _run_model.CONTROL_DIRECTORY
            / _run_model.CHECKPOINT_DIRECTORY
        )
        first = directory / "000001"
        if directory.is_symlink() or first.is_symlink():
            return set()
        references = _run_metadata.load_checkpoint(
            first / "manifest.json"
        ).artifacts
    except _run_model.RunStoreError:
        return set()
    if references is None:
        return set()
    return {
        item.relative.parent
        for item in references.all_records()[1]
        if item.relative.name == "prompt.txt"
        and item.relative.parent.parent.name == "invocations"
    }


def collect(
    root: pathlib.Path,
    /,
    *,
    run_id: str,
    report_path: pathlib.Path,
    graph_path: pathlib.Path,
    node_paths: Mapping[str, NodeIdentity],
    known_node_paths: Mapping[str, NodeIdentity] | None = None,
) -> dict[NodeIdentity, UsageTotals]:
    """Read actual attempts and roll each into one direct row per report."""
    # ponytail: scan per report; index attempts if large runs make this slow.
    parents, attempts = _inventory(root)
    inherited = _inherited_attempts(root, run_id)
    result: dict[NodeIdentity, UsageTotals] = {}
    seen: set[str] = set()
    origins = node_paths if known_node_paths is None else known_node_paths
    for attempt in attempts:
        if attempt.relative_to(root) in inherited:
            continue
        visit = attempt.parent.parent.relative_to(root)
        origin_node = origins.get(visit.parent.as_posix())
        if origin_node is not None and origin_node[3] != "agent":
            continue
        owner = _owner(visit, report_path, graph_path, parents)
        if owner is None:
            continue
        identity = _identity(owner, visit, report_path, parents, node_paths)
        if identity is None:
            continue
        if owner == report_path and identity[3] != "agent":
            continue
        record = _object(attempt / "usage.json")
        usage: Mapping[str, object] | None = None
        if record is not None:
            if record.get("replayed") is True:
                continue
            origin = record.get("run_id")
            if isinstance(origin, str) and origin != run_id:
                continue
            recorded_visit = _relative(record.get("visit_path"))
            recorded_report = _relative(record.get("report_path"))
            valid = (
                record.get("schema_version") == 1
                and origin == run_id
                and recorded_visit == visit
                and recorded_report == owner
            )
            attempt_id = record.get("attempt_id")
            if valid and isinstance(attempt_id, str) and attempt_id:
                if attempt_id in seen:
                    continue
                seen.add(attempt_id)
                raw_usage = record.get("usage")
                if isinstance(raw_usage, dict):
                    usage = cast(dict[str, object], raw_usage)
        result.setdefault(identity, UsageTotals()).include(usage)
    return result
