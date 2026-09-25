from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_agents import (
    EmptyState,
    _agent_graph,  # pyright: ignore[reportPrivateUsage]
    _RecordingInvoker,  # pyright: ignore[reportPrivateUsage]
    _workflow,  # pyright: ignore[reportPrivateUsage]
)
from verdog_runtime import ExecutionCancelled, _statistics
from verdog_runtime._statistics import RunStatistics, TimingRecord
from verdog_runtime.declarations import AgentNodeContext, Success, WorkflowConfiguration
from verdog_runtime.declarations.ids import AgentProfileId
from verdog_runtime.interpreter import Dispatcher


def _rows(output_dir: Path, section: str) -> list[list[str]]:
    report = (output_dir / "stats.md").read_text("utf-8")
    table = report.split(f"## {section}\n\n", 1)[1].split("\n\n", 1)[0]
    return [
        [cell.strip() for cell in line.split("|")[1:-1]] for line in table.splitlines()
    ]


@pytest.mark.parametrize(
    "node_type",
    [
        "python",
        "agent",
        "feature",
        "subroutine_call",
        "workflow_call",
        "enter",
        "exit",
        "failure",
        "unknown",
    ],
)
def test_repeated_visits_produce_compact_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_type: str
) -> None:
    now = 100.0
    monkeypatch.setattr(_statistics, "monotonic", lambda: now)
    records: list[TimingRecord] = []
    statistics = RunStatistics(tmp_path, record_handler=records.append)
    for visit in range(1, 26):
        now = 101.0 + 4 * (visit - 1)
        with statistics.measure(
            path=f"work/{visit:06d}",
            project_path=".",
            graph_id="main",
            node_id="work",
            node_type=node_type,
        ):
            assert len(records) == visit - 1
            now += 3.0
    now = 204.0
    statistics.write()

    assert {path.name for path in tmp_path.iterdir()} == {"trace.log", "stats.md"}
    kind_rows = _rows(tmp_path, "Summary")
    node_rows = _rows(tmp_path, "Nodes")
    assert kind_rows[0] == ["Type", "Visits", "Seconds"]
    assert node_rows[0] == ["Project", "Graph", "Node", "Type", "Visits", "Seconds"]
    assert kind_rows[2] == ["Total", "", "104.000000"]
    assert kind_rows[-1] == [node_type, "25", "75.000000"]
    assert node_rows[-1] == [".", "main", "work", node_type, "25", "75.000000"]
    assert len(kind_rows) == 4
    assert len(node_rows) == 3
    assert len(records) == 25
    assert all(
        record.status == "succeeded" and record.duration_seconds == 3.0
        for record in records
    )
    assert records[0] == TimingRecord(
        path="work/000001",
        project_path=".",
        graph_id="main",
        node_id="work",
        node_type=node_type,
        status="succeeded",
        duration_seconds=3.0,
    )
    lines = (tmp_path / "trace.log").read_text("utf-8").splitlines()
    assert len(lines) == 50
    assert re.fullmatch(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+\s*1\.000s\] START work/000001",
        lines[0],
    )
    assert lines[1].endswith(
        "END work/000001 status=succeeded duration=3.000000s"
    )


def test_identity_cells_are_escaped_once_without_parsing_numeric_text(
    tmp_path: Path,
) -> None:
    statistics = RunStatistics(tmp_path)
    statistics.accept(
        TimingRecord(
            path="work/000001",
            project_path="external|&<project>\r\nnext",
            graph_id="001",
            node_id="work\rpart\nend",
            node_type="kind|&<type>",
            status="succeeded",
            duration_seconds=3.25,
        )
    )
    statistics.write()

    assert _rows(tmp_path, "Nodes")[2:] == [
        [
            "external&#124;&amp;&lt;project&gt;<br>next",
            "001",
            "work<br>part<br>end",
            "kind&#124;&amp;&lt;type&gt;",
            "1",
            "3.250000",
        ]
    ]
    assert _rows(tmp_path, "Summary")[3:] == [
        ["kind&#124;&amp;&lt;type&gt;", "1", "3.250000"]
    ]


def test_scopes_keep_local_totals_and_forward_records_on_the_shared_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 20.0
    monkeypatch.setattr(_statistics, "monotonic", lambda: now)
    records: list[TimingRecord] = []
    statistics = RunStatistics(
        tmp_path, started_at=0.0, record_handler=records.append
    ).scoped(tmp_path)
    now = 21.0
    with statistics.measure(
        path="call/000001",
        project_path=".",
        graph_id="main",
        node_id="call",
        node_type="subroutine_call",
    ):
        now = 23.0
        child_output = tmp_path / "call/000001"
        child_output.mkdir(parents=True)
        child = statistics.scoped(child_output)
        with child.measure(
            path="call/000001/work/000001",
            project_path="external/child",
            graph_id="child",
            node_id="work",
            node_type="python",
        ):
            now = 31.0
        child.write()
        remote = TimingRecord(
            path="call/000001/call/000001/work/000001",
            project_path="external/child/external/remote",
            graph_id="remote",
            node_id="remote_work",
            node_type="python",
            status="succeeded",
            duration_seconds=2.0,
        )
        statistics.forward(remote)
        now = 33.0
    now = 35.0
    statistics.write()

    assert [(record.node_id, record.duration_seconds) for record in records] == [
        ("work", 8.0),
        ("remote_work", 2.0),
        ("call", 12.0),
    ]
    assert _rows(tmp_path, "Summary")[2:] == [
        ["Total", "", "15.000000"],
        ["subroutine_call", "1", "12.000000"],
    ]
    assert _rows(tmp_path, "Nodes")[2:] == [
        [".", "main", "call", "subroutine_call", "1", "12.000000"],
    ]
    assert _rows(child_output, "Summary")[2:] == [
        ["Total", "", "8.000000"],
        ["python", "1", "8.000000"],
    ]
    assert _rows(child_output, "Nodes")[2:] == [
        ["external/child", "child", "work", "python", "1", "8.000000"],
    ]
    assert not (child_output / "trace.log").exists()
    trace = (tmp_path / "trace.log").read_text("utf-8").splitlines()
    assert [float(line.split(" +", 1)[1].split("s]", 1)[0]) for line in trace] == [
        21.0,
        23.0,
        31.0,
        33.0,
    ]


def test_empty_run_has_total_with_blank_visits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_statistics, "monotonic", lambda: 3.0)
    RunStatistics(tmp_path, started_at=1.0).write()

    header, separator, total = _rows(tmp_path, "Summary")
    assert header == ["Type", "Visits", "Seconds"]
    assert len(separator) == 3
    assert total == ["Total", "", "2.000000"]
    nodes = _rows(tmp_path, "Nodes")
    assert len(nodes) == 2
    assert nodes[0] == ["Project", "Graph", "Node", "Type", "Visits", "Seconds"]


def test_snapshot_restores_aggregates_elapsed_time_and_existing_call_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(_statistics, "monotonic", lambda: now)
    first = RunStatistics(tmp_path).scoped(tmp_path)
    first.accept(
        TimingRecord(
            path="enter/000001",
            project_path=".",
            graph_id="main",
            node_id="enter",
            node_type="enter",
            status="succeeded",
            duration_seconds=2.0,
        )
    )
    now = 15.0
    snapshot = first.snapshot()
    (tmp_path / "stats.md").write_text(
        "old report\n\n## Calls\n\n- [child](activations/child/stats.md)\n",
        encoding="utf-8",
    )

    now = 100.0
    resumed = RunStatistics(tmp_path).scoped(tmp_path)
    resumed.restore(snapshot)
    resumed.accept(
        TimingRecord(
            path="work/000001",
            project_path=".",
            graph_id="main",
            node_id="work",
            node_type="python",
            status="succeeded",
            duration_seconds=3.0,
        )
    )
    now = 103.0
    resumed.write()

    assert _rows(tmp_path, "Summary")[2:] == [
        ["Total", "", "8.000000"],
        ["enter", "1", "2.000000"],
        ["python", "1", "3.000000"],
    ]
    assert [row[2] for row in _rows(tmp_path, "Nodes")[2:]] == ["enter", "work"]
    report = (tmp_path / "stats.md").read_text("utf-8")
    assert report.count("## Calls") == 1
    assert "[child](activations/child/stats.md)" in report


@pytest.mark.parametrize("error", [ValueError("failed"), ExecutionCancelled()])
def test_failed_and_cancelled_measurements_are_finalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    now = 0.0
    monkeypatch.setattr(_statistics, "monotonic", lambda: now)
    records: list[TimingRecord] = []
    statistics = RunStatistics(tmp_path, record_handler=records.append)
    path = "work/000001"
    with (
        pytest.raises(type(error)),
        statistics.measure(
            path=path,
            project_path=".",
            graph_id="main",
            node_id="work",
            node_type="python",
        ),
    ):
        now = 3.0
        raise error
    status = "cancelled" if isinstance(error, ExecutionCancelled) else "failed"
    statistics.write()

    assert len(records) == 1
    assert records[0].status == status
    assert records[0].duration_seconds == 3.0
    rows = _rows(tmp_path, "Summary")
    assert rows[-1] == ["python", "1", "3.000000"]
    assert rows[2] == ["Total", "", "3.000000"]
    assert f"END {path} status={status} duration=3.000000s" in (
        tmp_path / "trace.log"
    ).read_text("utf-8")


@pytest.mark.parametrize("calls", [0, 2])
def test_provider_calls_do_not_change_node_visit_count(
    tmp_path: Path, calls: int
) -> None:
    invoker = _RecordingInvoker()

    def implementation(
        input: object, state: EmptyState, context: AgentNodeContext, /
    ) -> Success[object, EmptyState]:
        for _ in range(calls):
            context.invoke("request", workspace=tmp_path)
        return Success(output=input, state=state)

    output = tmp_path / "output"
    (
        Dispatcher().run(
            _workflow(
                _agent_graph(implementation),
                WorkflowConfiguration(
                    profile_arguments={AgentProfileId("default"): invoker},
                ),
            ),
            None,
            output_dir=output,
        )
    )
    assert len(invoker.requests) == calls
    rows = _rows(output, "Summary")
    agents = [row for row in rows if row[:1] == ["agent"]]
    assert len(agents) == 1
    assert agents[0][1] == "1"
    assert float(agents[0][2]) >= 0.0
