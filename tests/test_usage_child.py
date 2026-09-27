"""Provider usage crosses graph and process boundaries without duplication."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from report_helpers import call_reports, table_rows
from test_child_process import (
    _child_project,  # pyright: ignore[reportPrivateUsage]
    _definition,  # pyright: ignore[reportPrivateUsage]
    _parent,  # pyright: ignore[reportPrivateUsage]
)

from verdog_runtime.declarations import SubroutineCall
from verdog_runtime.declarations.ids import GraphId
from verdog_runtime.interpreter import Dispatcher


@pytest.mark.parametrize("in_process", [False, True])
def test_child_usage_rolls_into_parent_call_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    child = _child_project(tmp_path)
    module = child / "src/child_project/subroutines/main/__init__.py"
    with module.open("a", encoding="utf-8") as source:
        source.write(
            """
from dataclasses import replace
from verdog_runtime import _usage
from verdog_runtime.agents import AgentReply
from verdog_runtime.declarations import (
    Agent, AgentAccess, AgentProfileDefinition, AgentSessionDefinition,
)
from verdog_runtime.declarations.ids import AgentProfileId, AgentSessionId


class MeasuredInvoker:
    session_provider = "claude"

    def __call__(self, request):
        _usage.capture(request.artifact_dir, {
            "provider": "claude",
            "scope": "invocation",
            "input_tokens": 1200,
            "cached_input_tokens": 200,
            "cache_write_input_tokens": 0,
            "output_tokens": 60,
            "reasoning_output_tokens": 10,
            "cost_usd": "0.125",
        })
        return AgentReply(text="measured")


def measured_visit(input, state, context):
    assert context.invoke(
        "test", workspace=context.output_dir, access=AgentAccess.READ_ONLY,
    ) == "measured"
    return Success(output=(input + 1, os.getpid()), state=state)


GRAPH = replace(
    GRAPH,
    nodes=(replace(WORK, operation=Agent(
        profile=AgentProfileId("test"), session=AgentSessionId("test"),
    )),),
    profiles=(AgentProfileDefinition(
        id=AgentProfileId("test"), name="Test",
        implementation=lambda input, params: MeasuredInvoker(),
    ),),
    sessions=(AgentSessionDefinition(
        id=AgentSessionId("test"), name="Test", persistent=False,
    ),),
    edges=(
        replace(GRAPH.edges[0], visit=VisitDefinition(
            implementation=measured_visit,
        )),
        GRAPH.edges[1],
    ),
)
"""
        )
    graph = _parent(child.name, [])
    if in_process:
        for name in tuple(sys.modules):
            if name == "child_project" or name.startswith("child_project."):
                monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "path", [str(child / "src"), *sys.path])
        graph = replace(
            graph,
            nodes=(
                replace(
                    graph.nodes[0],
                    operation=SubroutineCall(
                        definition_id=GraphId("child_project.main"),
                        definition_module="child_project.subroutines.main",
                        project_path=child.name,
                        params_types={
                            (child.name, GraphId("child_project.main")): type(
                                None
                            )
                        },
                        profile_arguments={},
                        session_arguments={},
                    ),
                ),
            ),
        )
    output = tmp_path / "output"
    result = Dispatcher(project_root=tmp_path).run(
        _definition(graph), 4, output_dir=output
    )
    assert result.output[0] == 5
    assert (result.output[1] == os.getpid()) is in_process

    parent_report = output / "stats.md"
    child_reports = call_reports(parent_report)
    assert len(child_reports) == 1
    expected = ["1", "1200", "200", "60", "0.125000"]
    for report, node, kind in (
        (
            parent_report,
            "call",
            "subroutine_call" if in_process else "workflow_call",
        ),
        (child_reports[0], "work", "agent"),
    ):
        rows = table_rows(report, "Nodes")
        charged = [row for row in rows if row[6] != "0"]
        assert len(charged) == 1
        assert charged[0][2:4] == [node, kind]
        assert charged[0][6:] == expected
        summary = table_rows(report, "Summary")
        assert next(row for row in summary if row[0] == "Total")[3:] == expected
        assert next(row for row in summary if row[0] == kind)[3:] == expected

    records = list(output.rglob("usage.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text("utf-8"))
    assert (
        record["report_path"]
        == child_reports[0].parent.relative_to(output).as_posix()
    )
    assert record["status"] == "succeeded"
