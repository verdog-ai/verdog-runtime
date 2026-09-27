import json
import re
from dataclasses import dataclass, field
from pathlib import Path
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


def test_configuration_flattens_fields_defaults_and_escapes_cells(
    tmp_path: Path,
) -> None:
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
    text = report.read_text("utf-8")
    if "\n## Calls\n\n" not in text:
        return []
    section = text.split("\n## Calls\n\n", 1)[1].split("\n## ", 1)[0]
    return [
        report.parent / unquote(target)
        for target in re.findall(r"\]\(([^\n]+)\)", section)
    ]


def _navigation(report: Path) -> dict[str, Path]:
    text = report.read_text("utf-8").split("<!-- /verdog-navigation -->", 1)[0]
    if not text.startswith("<!-- verdog-navigation -->\n"):
        return {}
    return {
        label: (report.parent / unquote(target)).resolve()
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
    }


def _reports(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_configuration(directory, ())
    (directory / "stats.md").write_text(
        "## Summary\n\n| Type | Visits | Seconds |\n"
        "| --- | --- | --- |\n| Total | | 1.000000 |\n"
        "\n## Nodes\n\n| Node | Visits |\n| --- | --- |\n",
        "utf-8",
    )


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

    expected = [
        tmp_path / scope(graph) / "config.md"
        for graph in (first, second, repeated)
    ]
    assert _links(tmp_path / "config.md") == expected
    assert _links(tmp_path / scope(first) / "config.md") == [
        tmp_path / scope(grandchild) / "config.md"
    ]
    assert (tmp_path / "config.md").read_text("utf-8").count("## Calls") == 1
    assert "## Calls" not in (
        tmp_path / scope(repeated) / "config.md"
    ).read_text("utf-8")

    for graph in graphs:
        (tmp_path / scope(graph) / "stats.md").write_text(
            "# Statistics\n", "utf-8"
        )
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
    unfinished = tmp_path / (
        child_without_stats.parent if legacy else child_without_stats
    )
    unfinished.mkdir(parents=True)
    write_configuration(unfinished, ())
    if not legacy:
        register_invocation(
            tmp_path,
            child_without_stats,
            child,
            child_without_stats,
            None,
            parent_report=child,
        )
    reports(_enter(child_without_stats))
    (tmp_path / "stats.md").write_text("# Statistics\n", "utf-8")
    reports.finish()
    assert (tmp_path / "stats.md").read_text("utf-8") == "# Statistics\n"
    assert not (directory / "stats.md").exists()
    assert not (unfinished / "stats.md").exists()


def test_navigation_orders_repeated_graphs_across_call_nodes_and_resume(
    tmp_path: Path,
) -> None:
    _reports(tmp_path)
    paths = (Path("z/000001"), Path("a/000001"), Path("z/000002"))
    for position, path in enumerate(paths, start=1):
        _reports(tmp_path / path)
        register_invocation(
            tmp_path,
            path,
            Path(),
            path,
            "same",
            parent_report=Path(),
            parent_visit_index=position * 3,
        )
    # The timing stream does not need to revisit completed siblings.
    reports = InvocationReports(tmp_path)
    reports.finish()
    expected = [tmp_path / path / "stats.md" for path in paths]
    assert _links(tmp_path / "stats.md") == expected
    assert _navigation(expected[0]) == {
        "↑ Parent": tmp_path / "stats.md",
        "Next call →": expected[1],
    }
    assert _navigation(expected[1]) == {
        "← Previous call": expected[0],
        "↑ Parent": tmp_path / "stats.md",
        "Next call →": expected[2],
    }
    before = {path: path.read_bytes() for path in expected}
    reports.finish()
    assert {path: path.read_bytes() for path in expected} == before

    new_path = Path("b/000001")
    _reports(tmp_path / new_path)
    register_invocation(
        tmp_path,
        new_path,
        Path(),
        new_path,
        "same",
        parent_report=Path(),
        parent_visit_index=15,
    )
    resumed = InvocationReports(tmp_path)
    resumed(_enter(new_path, graph_id="same"))
    resumed.finish()
    new_report = tmp_path / new_path / "stats.md"
    assert _navigation(expected[-1])["Next call →"] == new_report
    assert _navigation(new_report)["← Previous call"] == expected[-1]
    assert _links(tmp_path / "stats.md") == [*expected, new_report]
    assert (tmp_path / "stats.md").read_text("utf-8").count("## Calls") == 1
    assert (
        new_report.read_text("utf-8").count("<!-- verdog-navigation -->") == 1
    )


@pytest.mark.parametrize("metadata", [False, True])
def test_navigation_restores_legacy_call_order_and_flat_parentage(
    tmp_path: Path,
    metadata: bool,
) -> None:
    _reports(tmp_path)
    paths = (Path("activations/z"), Path("activations/a"))
    for index, path in enumerate(paths, start=1):
        _reports(tmp_path / path)
        if metadata:
            # Legacy metadata omitted both the report path and call order.
            (tmp_path / path / ".verdog-invocation.json").write_text(
                json.dumps(
                    {
                        "parent_graph": "graph-root",
                        "call_visit": f"graph-root/call/{index:06d}",
                        "graph_id": "same",
                    }
                ),
                "utf-8",
            )
    # Legacy reports themselves preserve chronological order without metadata.
    with (tmp_path / "stats.md").open("a", encoding="utf-8") as report:
        report.write(
            "\n## Calls\n\n"
            + "".join(
                f"- [call {index}](activations/{path.name}/stats.md)\n"
                for index, path in enumerate(paths, start=1)
            )
        )
    reports = InvocationReports(tmp_path)
    reports.finish()
    first, second = [tmp_path / path / "stats.md" for path in paths]
    assert _links(tmp_path / "stats.md") == [first, second]
    assert _navigation(first) == {
        "↑ Parent": tmp_path / "stats.md",
        "Next call →": second,
    }
    assert _navigation(second) == {
        "← Previous call": first,
        "↑ Parent": tmp_path / "stats.md",
    }


def test_navigation_escapes_relative_paths_and_skips_missing_reports(
    tmp_path: Path,
) -> None:
    paths = (
        Path("call %2F[odd](x)#é/000001"),
        Path("unfinished/000001"),
        Path("other call/000001"),
    )
    _reports(tmp_path)
    for index, path in enumerate(paths, start=1):
        _reports(tmp_path / path)
        register_invocation(
            tmp_path,
            path,
            Path(),
            path,
            "[child]*`\\|\nname",
            parent_report=Path(),
            parent_visit_index=index,
        )
    (tmp_path / paths[1] / "stats.md").unlink()
    reports = InvocationReports(tmp_path)
    reports.finish()
    first, _, last = [tmp_path / path / "stats.md" for path in paths]
    assert _navigation(first)["Next call →"] == last
    assert _navigation(last)["← Previous call"] == first
    assert (
        "../../call%20%252F%5Bodd%5D%28x%29%23%C3%A9/000001/stats.md"
        in last.read_text("utf-8")
    )
    assert r"\[child\]\*\`\\&#124;<br>name" in (
        tmp_path / "stats.md"
    ).read_text("utf-8")
    (tmp_path / "stats.md").unlink()
    reports.finish()
    assert "↑ Parent" not in _navigation(first)
    assert not (tmp_path / "stats.md").exists()
    assert not (tmp_path / paths[1] / "stats.md").exists()


def test_registering_restored_invocation_does_not_modify_metadata(
    tmp_path: Path,
) -> None:
    child = Path("call/000001")
    _reports(tmp_path)
    _reports(tmp_path / child)
    register_invocation(
        tmp_path, child, Path(), child, "child", parent_report=Path()
    )
    metadata = tmp_path / child / ".verdog-invocation.json"
    original = metadata.read_bytes(), metadata.stat().st_mtime_ns
    register_invocation(
        tmp_path,
        child,
        Path(),
        child,
        "child",
        parent_report=Path(),
        parent_visit_index=9,
    )
    InvocationReports(tmp_path).finish()
    assert (metadata.read_bytes(), metadata.stat().st_mtime_ns) == original
    with pytest.raises(ValueError, match="parent metadata changed"):
        register_invocation(
            tmp_path,
            child,
            Path(),
            Path("other/000001"),
            "child",
            parent_report=Path(),
        )


@pytest.mark.parametrize("index", [True, 0, -1, "1"])
def test_invocation_reports_reject_invalid_persisted_order(
    tmp_path: Path,
    index: object,
) -> None:
    child = Path("call/000001")
    _reports(tmp_path / child)
    (tmp_path / child / ".verdog-invocation.json").write_text(
        json.dumps(
            {
                "parent_graph": ".",
                "call_visit": child.as_posix(),
                "graph_id": "child",
                "parent_report": ".",
                "parent_visit_index": index,
            }
        ),
        "utf-8",
    )
    with pytest.raises(
        ValueError, match="invalid invocation parent visit index"
    ):
        InvocationReports(tmp_path)


def test_invocation_reports_reject_links_outside_run(tmp_path: Path) -> None:
    _reports(tmp_path)
    with (tmp_path / "config.md").open("a", encoding="utf-8") as report:
        report.write("\n## Calls\n\n- [foreign](../outside/stats.md)\n")
    with pytest.raises(ValueError, match="escapes the run directory"):
        InvocationReports(tmp_path)


def test_navigation_rejects_cycles_in_registered_parentage(
    tmp_path: Path,
) -> None:
    for child, parent in (("a", "b"), ("b", "a")):
        _reports(tmp_path / child)
        register_invocation(
            tmp_path,
            Path(child),
            Path(parent),
            Path(parent) / "call/000001",
            child,
            parent_report=Path(parent),
        )
    with pytest.raises(ValueError, match="parentage contains a cycle"):
        InvocationReports(tmp_path)


def test_navigation_rejects_legacy_calls_that_make_root_a_child(
    tmp_path: Path,
) -> None:
    _reports(tmp_path)
    _reports(tmp_path / "child")
    with (tmp_path / "config.md").open("a", encoding="utf-8") as report:
        report.write("\n## Calls\n\n- [child](child/config.md)\n")
    with (tmp_path / "child/config.md").open("a", encoding="utf-8") as report:
        report.write("\n## Calls\n\n- [root](../config.md)\n")
    with pytest.raises(ValueError, match="root cannot have a parent"):
        InvocationReports(tmp_path)
