from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import override
from urllib.parse import unquote

import pytest

from verdog_runtime._configuration import (
    ConfigurationValue,
    InvocationReports,
    register_invocation,
    write_configuration,
)
from verdog_runtime._statistics import TimingRecord


def _table_rows(tmp_path: Path) -> list[tuple[str, ...]]:
    report = (tmp_path / "config.md").read_text("utf-8")
    assert report.startswith("# Configuration\n\n| Parameter")
    return [
        tuple(cell.strip() for cell in line.strip("|").split("|"))
        for line in report.splitlines()[4:]
    ]


def test_configuration_flattens_fields_defaults_and_escapes_cells(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Search:
        width: int = 8

    @dataclass(frozen=True)
    class Params:
        backend: str = "lama"
        timeout: float = 120.0
        search: Search = field(default_factory=Search)
        secret: str = field(default="not-for-report", repr=False)

    params = Params()
    write_configuration(
        tmp_path,
        (
            ConfigurationValue("input", "problem|one\r\n<two>"),
            ConfigurationValue("", params),
            ConfigurationValue("./child::solver", params),
            ConfigurationValue("line|break\nname", "001"),
            ConfigurationValue("filename", Path("bad\udcff")),
        ),
    )

    assert _table_rows(tmp_path) == [
        ("input", "problem&#124;one<br>&lt;two&gt;"),
        ("backend", "lama"),
        ("timeout", "120.0"),
        ("search.width", "8"),
        ("secret", "[redacted]"),
        ("./child::solver.backend", "lama"),
        ("./child::solver.timeout", "120.0"),
        ("./child::solver.search.width", "8"),
        ("./child::solver.secret", "[redacted]"),
        ("line&#124;break<br>name", "001"),
        ("filename", "bad\\udcff"),
    ]
    assert "not-for-report" not in (tmp_path / "config.md").read_text("utf-8")


def test_configuration_handles_cycles_empty_params_and_unreadable_values(
    tmp_path: Path,
) -> None:
    @dataclass
    class Link:
        child: object = None

    @dataclass
    class Empty:
        pass

    class Opaque:
        @override
        def __repr__(self) -> str:
            raise ValueError("cannot display")

        def __deepcopy__(self, memo: object) -> object:
            raise AssertionError("configuration must not copy values")

    @dataclass
    class Values:
        opaque: object = field(default_factory=Opaque)
        hidden: object = field(default_factory=Opaque, repr=False)

    link = Link()
    link.child = link
    write_configuration(
        tmp_path,
        (
            ConfigurationValue("cycle", link),
            ConfigurationValue("again", link),
            ConfigurationValue("", Values()),
            ConfigurationValue("", Empty()),
            ConfigurationValue("", None),
            ConfigurationValue("optional", None),
        ),
    )

    assert _table_rows(tmp_path) == [
        ("cycle.child", "[cycle]"),
        ("again.child", "[cycle]"),
        ("opaque", "[unprintable Opaque]"),
        ("hidden", "[redacted]"),
        ("params", "None"),
        ("optional", "None"),
    ]


def _enter(graph: Path, *, graph_id: str = "child") -> TimingRecord:
    return TimingRecord(
        path=(graph / "custom-enter" / "000001").as_posix(),
        project_path=".",
        graph_id=graph_id,
        node_id="custom-enter",
        node_type="enter",
        status="succeeded",
        duration_seconds=0.0,
    )


def _links(report: Path) -> list[Path]:
    return [
        report.parent / unquote(target)
        for target in re.findall(r"\]\(([^\n]+)\)", report.read_text("utf-8"))
    ]


@pytest.mark.parametrize("legacy", [False, True])
def test_invocation_reports_link_direct_calls_in_observed_order(
    tmp_path: Path, legacy: bool
) -> None:
    reports = InvocationReports(tmp_path)
    root_graph = Path("graph-root") if legacy else Path()
    wrapper = "graph-child" if legacy else ""
    first = root_graph / "z" / "000001" / wrapper
    grandchild = first / "nested" / "000001" / wrapper
    second = root_graph / "a" / "000001" / wrapper
    repeated = root_graph / "z" / "000002" / wrapper
    graphs = (root_graph, first, grandchild, second, repeated)

    def scope(graph: Path) -> Path:
        return graph.parent if legacy else graph

    for graph in graphs:
        directory = tmp_path / scope(graph)
        directory.mkdir(parents=True, exist_ok=True)
        write_configuration(directory, (ConfigurationValue("params", None),))
        if not legacy and graph != root_graph:
            parent = graph.parent.parent
            register_invocation(
                tmp_path, graph, parent, graph, None, parent_report=parent
            )
        reports(_enter(graph))
    reports(_enter(first))

    expected = [tmp_path / scope(graph) / "config.md" for graph in (first, second, repeated)]
    assert _links(tmp_path / "config.md") == expected
    assert _links(tmp_path / scope(first) / "config.md") == [
        tmp_path / scope(grandchild) / "config.md"
    ]
    assert (tmp_path / "config.md").read_text("utf-8").count("## Calls") == 1
    assert "## Calls" not in (tmp_path / scope(repeated) / "config.md").read_text("utf-8")

    for graph in graphs:
        (tmp_path / scope(graph) / "stats.md").write_text("# Statistics\n", "utf-8")
    reports.finish()
    assert _links(tmp_path / "stats.md") == [
        path.with_name("stats.md") for path in expected
    ]
    assert _links(tmp_path / scope(first) / "stats.md") == [
        tmp_path / scope(grandchild) / "stats.md"
    ]


@pytest.mark.parametrize("legacy", [False, True])
def test_invocation_reports_escape_links_and_skip_missing_reports(
    tmp_path: Path, legacy: bool
) -> None:
    reports = InvocationReports(tmp_path)
    root_graph = Path("graph-root") if legacy else Path()
    wrapper = "graph-child" if legacy else ""
    write_configuration(tmp_path, ())
    reports(_enter(root_graph))
    missing = root_graph / "missing" / "000001" / wrapper
    reports(_enter(missing))
    child = root_graph / "call%2F[odd]" / "000001" / wrapper
    directory = tmp_path / (child.parent if legacy else child)
    directory.mkdir(parents=True)
    write_configuration(directory, ())
    if not legacy:
        register_invocation(
            tmp_path, child, root_graph, child, None, parent_report=root_graph
        )
    reports(_enter(child, graph_id="[child]*`\\|\nname"))
    report = (tmp_path / "config.md").read_text("utf-8")
    assert _links(tmp_path / "config.md") == [directory / "config.md"]
    assert "call%252F%5Bodd%5D/000001/config.md" in report
    assert r"\[child\]\*\`\\&#124;<br>name" in report

    child_without_stats = child / "unfinished" / "000001" / wrapper
    unfinished = tmp_path / (child_without_stats.parent if legacy else child_without_stats)
    unfinished.mkdir(parents=True)
    write_configuration(unfinished, ())
    if not legacy:
        register_invocation(
            tmp_path, child_without_stats, child, child_without_stats, None,
            parent_report=child,
        )
    reports(_enter(child_without_stats))
    (tmp_path / "stats.md").write_text("# Statistics\n", "utf-8")
    reports.finish()
    assert (tmp_path / "stats.md").read_text("utf-8") == "# Statistics\n"
    assert not (directory / "stats.md").exists()
    assert not (unfinished / "stats.md").exists()
