from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from verdog_runtime import _usage_reports
from verdog_runtime._configuration import register_invocation


def _attempt(
    root: Path,
    report: str,
    node: str,
    *,
    slot: str = "000001",
    visit: str = "000001",
    tokens: int = 10,
    cost: str = "0.1",
    run_id: str = "current",
    status: str = "succeeded",
    attempt_id: str | None = None,
) -> Path:
    visit_path = Path(report) / node / visit
    directory = root / visit_path / "invocations" / slot
    directory.mkdir(parents=True)
    (directory / "prompt.txt").write_text("request", "utf-8")
    (directory / "usage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt_id": attempt_id or str(directory.relative_to(root)),
                "run_id": run_id,
                "project_path": ".",
                "graph_id": "graph",
                "node_id": node,
                "report_path": Path(report).as_posix(),
                "visit_path": visit_path.as_posix(),
                "status": status,
                "usage": {
                    "input_tokens": tokens,
                    "cached_input_tokens": 0,
                    "output_tokens": tokens // 2,
                    "cost_usd": cost,
                },
            }
        ),
        "utf-8",
    )
    return directory


def _collect(root: Path, report: str, **nodes: str) -> dict[str, list[str]]:
    graph_path = Path(report)
    rows = _usage_reports.collect(
        root,
        run_id="current",
        report_path=graph_path,
        graph_path=graph_path,
        node_paths={
            (graph_path / node).as_posix(): (".", "graph", node, kind)
            for node, kind in nodes.items()
        },
    )
    return {
        identity[2]: list(totals.cells()) for identity, totals in rows.items()
    }


def test_nested_cost_is_attributed_once_to_each_direct_caller(
    tmp_path: Path,
) -> None:
    _attempt(tmp_path, ".", "local")
    child = "call/000001"
    _attempt(tmp_path, child, "child_agent", tokens=20, cost="0.2")
    register_invocation(
        tmp_path,
        Path(child),
        Path(),
        Path(child),
        "child",
        parent_report=Path(),
    )
    grandchild = "activations/legacy-flat"
    _attempt(tmp_path, grandchild, "grand_agent", tokens=30, cost="0.3")
    register_invocation(
        tmp_path,
        Path(grandchild),
        Path(child),
        Path(child) / "nested/000001",
        "grandchild",
        parent_report=Path(child),
    )

    assert _collect(tmp_path, ".", local="agent", call="workflow_call") == {
        "local": ["1", "10", "0", "5", "0.100000"],
        "call": ["2", "50", "0", "25", "0.500000"],
    }
    assert _collect(
        tmp_path, child, child_agent="agent", nested="subroutine_call"
    ) == {
        "child_agent": ["1", "20", "0", "10", "0.200000"],
        "nested": ["1", "30", "0", "15", "0.300000"],
    }
    assert _collect(tmp_path, grandchild, grand_agent="agent") == {
        "grand_agent": ["1", "30", "0", "15", "0.300000"]
    }


def test_failures_retries_replays_inherited_and_legacy_attempts(
    tmp_path: Path,
) -> None:
    _attempt(tmp_path, ".", "agent", status="failed", attempt_id="first")
    _attempt(tmp_path, ".", "agent", slot="000002", tokens=20, cost="0.2")
    duplicate = _attempt(
        tmp_path, ".", "agent", slot="000003", attempt_id="first"
    )
    replay = _attempt(tmp_path, ".", "agent", slot="000004")
    (replay / "usage.json").write_text(
        '{"schema_version": 1, "replayed": true}'
    )
    _attempt(tmp_path, ".", "agent", slot="000005", run_id="source")
    missing = _attempt(tmp_path, ".", "agent", slot="000006")
    (missing / "usage.json").unlink()
    assert duplicate.is_dir()
    assert _collect(tmp_path, ".", agent="agent") == {
        "agent": ["3", "30 + ?", "0 + ?", "15 + ?", "0.300000 + ?"]
    }


def test_unknown_invalid_and_tiny_costs_are_not_reported_as_zero(
    tmp_path: Path,
) -> None:
    directory = _attempt(tmp_path, ".", "agent", cost="0.000000001")
    assert _collect(tmp_path, ".", agent="agent")["agent"][-1] == "<0.000001"
    record = cast(
        dict[str, object], json.loads((directory / "usage.json").read_text())
    )
    record["usage"] = {
        "input_tokens": True,
        "cached_input_tokens": -1,
        "output_tokens": None,
        "cost_usd": "NaN",
    }
    (directory / "usage.json").write_text(json.dumps(record))
    assert _collect(tmp_path, ".", agent="agent") == {
        "agent": ["1", "unknown", "unknown", "unknown", "unknown"]
    }


def test_unrelated_files_symlinks_and_escaping_metadata_are_not_followed(
    tmp_path: Path,
) -> None:
    real = _attempt(tmp_path, ".", "agent")
    _attempt(tmp_path, "agent/000001/workspace", "authored")
    (tmp_path / "agent/000001/invocations/000002").symlink_to(
        real, target_is_directory=True
    )
    (tmp_path / "linked").symlink_to(
        tmp_path / "agent", target_is_directory=True
    )
    outside = _attempt(tmp_path, "foreign", "agent")
    (outside.parents[3] / ".verdog-invocation.json").write_text(
        json.dumps(
            {
                "parent_graph": "..",
                "call_visit": "../call/000001",
                "parent_report": ".",
            }
        )
    )
    assert _collect(tmp_path, ".", agent="agent") == {
        "agent": ["1", "10", "0", "5", "0.100000"]
    }


def test_legacy_graph_wrapper_uses_recorded_report_parent(
    tmp_path: Path,
) -> None:
    child = Path("activations/child")
    directory = _attempt(tmp_path, str(child / "child_graph"), "agent")
    record = cast(
        dict[str, object], json.loads((directory / "usage.json").read_text())
    )
    record["report_path"] = child.as_posix()
    (directory / "usage.json").write_text(json.dumps(record))
    register_invocation(
        tmp_path,
        child,
        Path("root_graph"),
        Path("root_graph/call/000001"),
        "child_graph",
        parent_report=Path(),
    )
    rows = _usage_reports.collect(
        tmp_path,
        run_id="current",
        report_path=Path(),
        graph_path=Path("root_graph"),
        node_paths={
            "root_graph/call": (".", "root", "call", "subroutine_call")
        },
    )
    assert list(rows.values())[0].cells() == ("1", "10", "0", "5", "0.100000")


def test_nested_agent_named_invocations_is_not_pruned(tmp_path: Path) -> None:
    child = Path("call/000001")
    attempt = _attempt(tmp_path, child.as_posix(), "invocations")
    register_invocation(
        tmp_path, child, Path(), child, "child", parent_report=Path()
    )
    _attempt(tmp_path, attempt.relative_to(tmp_path).as_posix(), "authored")
    expected = ["1", "10", "0", "5", "0.100000"]
    assert _collect(tmp_path, ".", call="workflow_call") == {"call": expected}
    assert _collect(tmp_path, child.as_posix(), invocations="agent") == {
        "invocations": expected
    }
    _, attempts = _usage_reports._inventory(tmp_path)  # pyright: ignore[reportPrivateUsage]
    assert attempts == [attempt]


def test_agent_named_invocations_can_write_its_own_prompt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "123456"
    child = Path("call/000001")
    expected_attempts: set[Path] = set()
    for report in (Path(), child):
        attempt = _attempt(root, report.as_posix(), "invocations")
        expected_attempts.add(attempt)
        (attempt.parent.parent / "prompt.txt").write_text("authored prompt")
    register_invocation(
        root, child, Path(), child, "child", parent_report=Path()
    )
    expected = ["1", "10", "0", "5", "0.100000"]
    assert _collect(root, ".", invocations="agent", call="workflow_call") == {
        "invocations": expected,
        "call": expected,
    }
    assert _collect(root, child.as_posix(), invocations="agent") == {
        "invocations": expected
    }
    _, attempts = _usage_reports._inventory(root)  # pyright: ignore[reportPrivateUsage]
    assert set(attempts) == expected_attempts


def test_bad_fork_provenance_remains_unknown_instead_of_hiding_attempts(
    tmp_path: Path,
) -> None:
    from test_run_store import (
        _checkpoint,  # pyright: ignore[reportPrivateUsage]
    )

    from verdog_runtime._run_store import ParentRun, RunStore

    root = tmp_path / "run"
    project = tmp_path / "project"
    root.mkdir()
    project.mkdir()
    inherited = _attempt(root, ".", "agent")
    (inherited / "usage.json").unlink()
    store = RunStore.create(
        root,
        project_root=project,
        workflow_id="main",
        definition_id="graph",
        module="test",
        run_id="current",
        parent=ParentRun(
            run_id="source",
            operation="fork",
            checkpoint=2,
            arguments="checkpoint",
        ),
    )
    store.commit_checkpoint(_checkpoint(1), capture_artifacts=True)
    _attempt(root, ".", "agent", visit="000002")
    assert _collect(root, ".", agent="agent")["agent"] == [
        "1",
        "10",
        "0",
        "5",
        "0.100000",
    ]

    manifest = store.checkpoint_directory(1) / "manifest.json"
    original = manifest.read_bytes()
    for malformed in (b"not JSON", b'{"schema_version": 999}'):
        manifest.write_bytes(malformed)
        assert _collect(root, ".", agent="agent")["agent"] == [
            "2",
            "10 + ?",
            "0 + ?",
            "5 + ?",
            "0.100000 + ?",
        ]
    manifest.unlink()
    assert _collect(root, ".", agent="agent")["agent"][0] == "2"
    external = tmp_path / "external.json"
    external.write_bytes(original)
    manifest.symlink_to(external)
    assert _collect(root, ".", agent="agent")["agent"][0] == "2"
