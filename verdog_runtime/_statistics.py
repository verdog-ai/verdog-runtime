"""Node visit timings, accumulated by node and node type."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import pathlib
import time
from collections.abc import Callable, Generator
from typing import Literal, cast

from verdog_runtime import _markdown, cancellation

TimingStatus = Literal["succeeded", "failed", "cancelled"]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TimingRecord:
    path: str
    project_path: str
    graph_id: str
    node_id: str
    node_type: str
    status: TimingStatus
    duration_seconds: float


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TimingAggregate:
    project_path: str
    graph_id: str
    node_id: str
    node_type: str
    visits: int
    duration_seconds: float


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StatisticsSnapshot:
    elapsed_seconds: float
    nodes: tuple[TimingAggregate, ...]


EMPTY_STATISTICS = StatisticsSnapshot(elapsed_seconds=0.0, nodes=())


def validate_statistics_snapshot(snapshot: object, /) -> None:
    if not isinstance(snapshot, StatisticsSnapshot):
        raise ValueError("checkpoint graph statistics are invalid")
    elapsed = cast(object, snapshot.elapsed_seconds)
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("checkpoint graph elapsed timing is invalid")
    raw_nodes = cast(object, snapshot.nodes)
    if not isinstance(raw_nodes, tuple):
        raise ValueError("checkpoint graph timing aggregates are invalid")
    identities: set[tuple[str, str, str, str]] = set()
    for raw_node in cast(tuple[object, ...], raw_nodes):
        if not isinstance(raw_node, TimingAggregate):
            raise ValueError("checkpoint graph timing aggregate is invalid")
        node = raw_node
        identity_values = (
            cast(object, node.project_path),
            cast(object, node.graph_id),
            cast(object, node.node_id),
            cast(object, node.node_type),
        )
        if not all(
            isinstance(value, str) and value for value in identity_values
        ):
            raise ValueError("checkpoint graph timing identity is invalid")
        identity = cast(tuple[str, str, str, str], identity_values)
        if identity in identities:
            raise ValueError(
                "checkpoint graph timing identities are not unique"
            )
        identities.add(identity)
        visits = cast(object, node.visits)
        if type(visits) is not int or visits <= 0:
            raise ValueError("checkpoint graph timing visit count is invalid")
        duration = cast(object, node.duration_seconds)
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("checkpoint graph timing duration is invalid")


@dataclasses.dataclass(slots=True, kw_only=True)
class TimingSpan:
    """An explicitly managed node timing that may cross activation steps."""

    statistics: RunStatistics
    path: str
    project_path: str
    graph_id: str
    node_id: str
    node_type: str
    started_at: float
    default_status: TimingStatus = "succeeded"
    _finished: bool = False

    def finish(self, status: TimingStatus | None = None, /) -> None:
        if self._finished:
            return
        self._finished = True
        resolved = self.default_status if status is None else status
        elapsed = time.monotonic() - self.started_at
        self.statistics.accept(
            TimingRecord(
                path=self.path,
                project_path=self.project_path,
                graph_id=self.graph_id,
                node_id=self.node_id,
                node_type=self.node_type,
                status=resolved,
                duration_seconds=elapsed,
            )
        )
        self.statistics._trace(  # pyright: ignore[reportPrivateUsage]
            f"END {self.path} status={resolved} duration={elapsed:.6f}s"
        )


class RunStatistics:
    def __init__(
        self,
        root: pathlib.Path,
        *,
        started_at: float | None = None,
        record_handler: Callable[[TimingRecord], None] | None = None,
    ) -> None:
        self.root = root
        self.started_at = time.monotonic() if started_at is None else started_at
        self._output_dir = root
        self._scope_started_at = self.started_at
        self._record_handler = record_handler
        self._nodes: dict[tuple[str, str, str, str], tuple[int, float]] = {}

    def scoped(self, output_dir: pathlib.Path) -> RunStatistics:
        scope = RunStatistics(
            self.root,
            started_at=self.started_at,
            record_handler=self._record_handler,
        )
        scope._output_dir = output_dir
        scope._scope_started_at = time.monotonic()
        return scope

    def snapshot(self) -> StatisticsSnapshot:
        return StatisticsSnapshot(
            elapsed_seconds=max(0.0, time.monotonic() - self._scope_started_at),
            nodes=tuple(
                TimingAggregate(
                    project_path=project_path,
                    graph_id=graph_id,
                    node_id=node_id,
                    node_type=node_type,
                    visits=visits,
                    duration_seconds=seconds,
                )
                for (
                    project_path,
                    graph_id,
                    node_id,
                    node_type,
                ), (visits, seconds) in sorted(self._nodes.items())
            ),
        )

    def restore(self, snapshot: StatisticsSnapshot, /) -> None:
        validate_statistics_snapshot(snapshot)
        self._scope_started_at = time.monotonic() - snapshot.elapsed_seconds
        self._nodes = {
            (
                node.project_path,
                node.graph_id,
                node.node_id,
                node.node_type,
            ): (node.visits, node.duration_seconds)
            for node in snapshot.nodes
        }

    def accept(self, record: TimingRecord) -> None:
        key = (
            record.project_path,
            record.graph_id,
            record.node_id,
            record.node_type,
        )
        count, seconds = self._nodes.get(key, (0, 0.0))
        self._nodes[key] = count + 1, seconds + record.duration_seconds
        self.forward(record)

    def forward(self, record: TimingRecord) -> None:
        if self._record_handler is not None:
            self._record_handler(record)

    def start(
        self,
        *,
        path: str,
        project_path: str,
        graph_id: str,
        node_id: str,
        node_type: str,
        status: TimingStatus = "succeeded",
    ) -> TimingSpan:
        started = time.monotonic()
        self._trace(f"START {path}")
        return TimingSpan(
            statistics=self,
            path=path,
            project_path=project_path,
            graph_id=graph_id,
            node_id=node_id,
            node_type=node_type,
            started_at=started,
            default_status=status,
        )

    def _trace(self, message: str) -> None:
        elapsed = time.monotonic() - self.started_at
        # Retain an existing legacy log when resuming; dotted names cannot
        # collide with a node ID in the direct node/visit layout.
        path = self.root / "trace"
        if not path.is_file():
            path = self.root / "trace.log"
        with path.open("a", encoding="utf-8") as trace:
            trace.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"+{elapsed:10.3f}s] {message}\n"
            )

    @contextlib.contextmanager
    def measure(
        self,
        *,
        path: str,
        project_path: str,
        graph_id: str,
        node_id: str,
        node_type: str,
        status: TimingStatus = "succeeded",
    ) -> Generator[None]:
        span = self.start(
            path=path,
            project_path=project_path,
            graph_id=graph_id,
            node_id=node_id,
            node_type=node_type,
            status=status,
        )
        try:
            yield
        except BaseException as error:
            status = (
                "cancelled"
                if isinstance(error, cancellation.ExecutionCancelled)
                else "failed"
            )
            raise
        finally:
            span.finish(status)

    def write(self) -> None:
        total = time.monotonic() - self._scope_started_at
        nodes = _markdown.format_table(
            (
                (*identity, count, f"{seconds:.6f}")
                for identity, (count, seconds) in sorted(self._nodes.items())
            ),
            ("Project", "Graph", "Node", "Type", "Visits", "Seconds"),
        )
        kinds: dict[str, tuple[int, float]] = {}
        for (_, _, _, kind), (count, seconds) in self._nodes.items():
            previous_count, previous_seconds = kinds.get(kind, (0, 0.0))
            kinds[kind] = previous_count + count, previous_seconds + seconds
        summary = _markdown.format_table(
            (
                ("Total", "", f"{total:.6f}"),
                *(
                    (kind, count, f"{seconds:.6f}")
                    for kind, (count, seconds) in sorted(kinds.items())
                ),
            ),
            ("Type", "Visits", "Seconds"),
        )
        report = self._output_dir / "stats.md"
        calls = ""
        if report.is_file():
            existing = report.read_text(encoding="utf-8", errors="replace")
            marker = "\n## Calls\n\n"
            if marker in existing:
                calls = marker + existing.split(marker, 1)[1]
        report.write_text(
            f"## Summary\n\n{summary}\n\n## Nodes\n\n{nodes}\n{calls}",
            encoding="utf-8",
            errors="backslashreplace",
        )
