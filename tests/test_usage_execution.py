from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from report_helpers import table_rows
from test_agents import (
    EmptyState,
    _agent_graph,  # pyright: ignore[reportPrivateUsage]
    _workflow,  # pyright: ignore[reportPrivateUsage]
)

from verdog_runtime import _usage
from verdog_runtime._run_store import RunStore
from verdog_runtime.agents import (
    AgentReply,
    AgentRequest,
    AgentSessionAction,
    AgentSessionCapabilities,
)
from verdog_runtime.declarations import (
    AgentNodeContext,
    EdgeDefinition,
    Success,
    VisitDefinition,
    WorkflowConfiguration,
)
from verdog_runtime.declarations.ids import (
    AgentProfileId,
    EdgeId,
    NodeId,
    ProviderSessionId,
)
from verdog_runtime.interpreter import (
    CheckpointPolicy,
    Dispatcher,
    SessionPolicy,
)
from verdog_runtime.interpreter._invocations import InvocationJournalError


@dataclass
class _MeteredInvoker:
    session_provider = "claude"
    session_capabilities = AgentSessionCapabilities(fork_latest=True)

    interrupt_first: bool = False
    failures: set[int] = field(default_factory=set[int])
    requests: list[AgentRequest] = field(default_factory=list[AgentRequest])
    turns: dict[str, int] = field(default_factory=dict[str, int])

    def __call__(self, request: AgentRequest, /) -> AgentReply:
        self.requests.append(request)
        source = request.provider_session_id
        number = self.turns.get(str(source), 0) + 1
        session_id = (
            str(source)
            if source is not None
            and request.provider_session_action is AgentSessionAction.CONTINUE
            else f"session-{len(self.requests)}"
        )
        self.turns[session_id] = number
        _usage.capture(
            request.artifact_dir,
            {
                "scope": "session",
                "provider": "claude",
                "provider_session_id": session_id,
                "model": "fake-model",
                "input_tokens": number * 100,
                "cached_input_tokens": number * 20,
                "cache_write_input_tokens": 0,
                "output_tokens": number * 40,
                "reasoning_output_tokens": None,
                "cost_usd": str(Decimal("0.25") * number),
            },
        )
        if self.interrupt_first and len(self.requests) == 1:
            raise KeyboardInterrupt
        if len(self.requests) in self.failures:
            raise RuntimeError("provider failed after reporting usage")
        return AgentReply(
            text=f"reply-{len(self.requests)}",
            provider_session_id=ProviderSessionId(session_id),
        )


def _total(output: Path) -> list[str]:
    return table_rows(output / "stats.md", "Summary")[0][3:]


def _records(output: Path) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(path.read_text()))
        for path in sorted(output.glob("*/*/invocations/*/usage.json"))
    ]


def _without_capture(*args: object) -> None:
    pass


def test_multiple_calls_and_authored_parse_failure_keep_paid_usage(
    tmp_path: Path,
) -> None:
    invoker = _MeteredInvoker()

    def implementation(
        input: object, state: EmptyState, context: AgentNodeContext, /
    ) -> Success[object, EmptyState]:
        context.invoke("first", workspace=tmp_path)
        response = context.invoke("second", workspace=tmp_path)
        return Success(output=json.loads(response), state=state)

    definition = _workflow(
        _agent_graph(implementation),
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    output = tmp_path / "run"
    with pytest.raises(json.JSONDecodeError):
        Dispatcher(project_root=tmp_path).run(
            definition,
            None,
            output_dir=output,
            checkpointing=CheckpointPolicy.OFF,
        )

    assert _total(output) == ["2", "200", "40", "80", "0.500000"]
    row = next(
        row
        for row in table_rows(output / "stats.md", "Nodes")
        if row[2] == "agent"
    )
    assert row[4] == "1"
    assert row[6:] == _total(output)
    assert all(record["status"] == "succeeded" for record in _records(output))


def test_accounting_failure_keeps_reply_and_invalidates_the_session_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoker = _MeteredInvoker()
    original_finish = _usage.finish
    finished = 0

    def finish(
        directory: Path, status: str, provider_session_id: str | None
    ) -> dict[str, object] | None:
        nonlocal finished
        finished += 1
        if finished == 2:
            raise OSError("accounting finalization failed after capture")
        return original_finish(directory, status, provider_session_id)

    monkeypatch.setattr(_usage, "finish", finish)

    def implementation(
        input: object, state: EmptyState, context: AgentNodeContext, /
    ) -> Success[object, EmptyState]:
        assert context.invoke("first", workspace=tmp_path) == "reply-1"
        assert context.invoke("second", workspace=tmp_path) == "reply-2"
        return Success(
            output=context.invoke("third", workspace=tmp_path), state=state
        )

    definition = _workflow(
        _agent_graph(implementation),
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    output = tmp_path / "run"
    result = Dispatcher(project_root=tmp_path).run(
        definition,
        None,
        output_dir=output,
        checkpointing=CheckpointPolicy.OFF,
    )

    assert result.output == "reply-3"
    assert len(invoker.requests) == 3
    assert invoker.requests[-1].provider_session_id == "session-1"
    records = _records(output)
    assert [
        cast(dict[str, object], record["usage"])["input_tokens"]
        for record in records
    ] == [100, 100, None]
    assert _total(output) == [
        "3",
        "200 + ?",
        "40 + ?",
        "80 + ?",
        "0.500000 + ?",
    ]


def test_replay_restores_baseline_without_counting_the_replayed_call(
    tmp_path: Path,
) -> None:
    invoker = _MeteredInvoker()
    visits = 0

    def implementation(
        input: object, state: EmptyState, context: AgentNodeContext, /
    ) -> Success[object, EmptyState]:
        nonlocal visits
        context.invoke("first", workspace=tmp_path)
        visits += 1
        if visits == 1:
            raise KeyboardInterrupt
        return Success(
            output=context.invoke("second", workspace=tmp_path), state=state
        )

    definition = _workflow(
        _agent_graph(implementation),
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Dispatcher(project_root=tmp_path).run(
            definition,
            None,
            output_dir=output,
            checkpointing=CheckpointPolicy.REQUIRED,
        )
    assert _total(output) == ["1", "100", "20", "40", "0.250000"]

    result = Dispatcher(project_root=tmp_path).resume(
        definition, output_dir=output
    )
    assert result.output == "reply-2"
    assert visits == 2
    assert len(invoker.requests) == 2
    assert invoker.requests[1].provider_session_id == "session-1"
    assert _total(output) == ["2", "200", "40", "80", "0.500000"]
    assert (
        sum(record.get("replayed") is True for record in _records(output)) == 1
    )


def test_failed_attempt_and_explicit_retry_count_once_each(
    tmp_path: Path,
) -> None:
    invoker = _MeteredInvoker(interrupt_first=True)

    def implementation(
        input: object, state: EmptyState, context: AgentNodeContext, /
    ) -> Success[object, EmptyState]:
        return Success(
            output=context.invoke("prompt", workspace=tmp_path), state=state
        )

    definition = _workflow(
        _agent_graph(implementation),
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        Dispatcher(project_root=tmp_path).run(
            definition,
            None,
            output_dir=output,
            checkpointing=CheckpointPolicy.REQUIRED,
        )
    with pytest.raises(InvocationJournalError, match="invocation.ambiguous"):
        Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)
    assert len(invoker.requests) == 1
    assert _total(output) == ["1", "100", "20", "40", "0.250000"]

    Dispatcher(project_root=tmp_path).resume(
        definition, output_dir=output, retry_incomplete=True
    )
    assert len(invoker.requests) == 2
    assert _total(output) == ["2", "200", "40", "80", "0.500000"]
    actual = [
        record for record in _records(output) if not record.get("replayed")
    ]
    assert [record["status"] for record in actual] == ["cancelled", "succeeded"]
    assert len({record["attempt_id"] for record in actual}) == 2


@pytest.mark.parametrize("policy", [SessionPolicy.BRANCH, SessionPolicy.FRESH])
@pytest.mark.parametrize("legacy", [False, True])
def test_fork_excludes_inherited_cost_and_applies_session_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: SessionPolicy,
    legacy: bool,
) -> None:
    invoker = _MeteredInvoker()
    stop_before_second = True

    def implementation(
        input: object, state: EmptyState, context: AgentNodeContext, /
    ) -> Success[object, EmptyState]:
        if context.node_id == "second" and stop_before_second:
            raise KeyboardInterrupt
        return Success(
            output=context.invoke("prompt", workspace=tmp_path), state=state
        )

    graph = _agent_graph(implementation)
    first = graph.nodes[0]
    second = replace(first, id=NodeId("second"), name="Second")
    graph = replace(
        graph,
        nodes=(first, second),
        edges=(
            graph.edges[0],
            EdgeDefinition(
                id=EdgeId("first-second"),
                source=first.id,
                target=second.id,
                visit=VisitDefinition(implementation=implementation),
            ),
            EdgeDefinition(
                id=EdgeId("second-exit"), source=second.id, target=graph.exit.id
            ),
        ),
    )
    definition = _workflow(
        graph,
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    source = tmp_path / "source"
    with monkeypatch.context() as capture, pytest.raises(KeyboardInterrupt):
        if legacy:
            for name in ("begin", "capture", "finish", "replay"):
                capture.setattr(_usage, name, _without_capture)
        Dispatcher(project_root=tmp_path).run(
            definition,
            None,
            output_dir=source,
            checkpointing=CheckpointPolicy.REQUIRED,
        )
    store = RunStore.open(source)
    checkpoint = next(
        item.sequence
        for item in store.checkpoints()
        if item.next and item.next.node == "second"
    )
    source_id = store.manifest().id
    assert _total(source) == (
        ["1", "unknown", "unknown", "unknown", "unknown"]
        if legacy
        else ["1", "100", "20", "40", "0.250000"]
    )
    stop_before_second = False

    target = tmp_path / "fork"
    Dispatcher(project_root=tmp_path).fork(
        definition,
        source_output_dir=source,
        checkpoint=checkpoint,
        output_dir=target,
        sessions=policy,
    )

    request = invoker.requests[1]
    assert request.provider_session_id == (
        "session-1" if policy is SessionPolicy.BRANCH else None
    )
    assert request.provider_session_action is (
        AgentSessionAction.FORK
        if policy is SessionPolicy.BRANCH
        else AgentSessionAction.CONTINUE
    )
    assert _total(target) == (
        ["1", "unknown", "unknown", "unknown", "unknown"]
        if legacy and policy is SessionPolicy.BRANCH
        else ["1", "100", "20", "40", "0.250000"]
    )
    target_id = RunStore.open(target).manifest().id
    assert {record["run_id"] for record in _records(target)} == (
        {target_id} if legacy else {source_id, target_id}
    )
    rows = {row[2]: row for row in table_rows(target / "stats.md", "Nodes")}
    assert rows["agent"][4] == "1"
    assert rows["agent"][6:] == ["0", "0", "0", "0", "0.000000"]
    assert rows["second"][6:] == _total(target)


@pytest.mark.parametrize("branch_failure", [False, True])
def test_caught_provider_failure_preserves_the_correct_session_baseline(
    tmp_path: Path, branch_failure: bool
) -> None:
    invoker = _MeteredInvoker(failures={2})

    def implementation(
        input: object, state: EmptyState, context: AgentNodeContext, /
    ) -> Success[object, EmptyState]:
        if context.node_id == "agent":
            context.invoke("first", workspace=tmp_path)
            if branch_failure:
                return Success(output=input, state=state)
        with pytest.raises(RuntimeError, match="provider failed"):
            context.invoke("failed", workspace=tmp_path)
        return Success(
            output=context.invoke("after-failure", workspace=tmp_path),
            state=state,
        )

    graph = _agent_graph(implementation)
    if branch_failure:
        second = replace(graph.nodes[0], id=NodeId("second"), name="Second")
        graph = replace(
            graph,
            nodes=(*graph.nodes, second),
            edges=(
                graph.edges[0],
                EdgeDefinition(
                    id=EdgeId("first-second"),
                    source=graph.nodes[0].id,
                    target=second.id,
                    visit=VisitDefinition(implementation=implementation),
                ),
                EdgeDefinition(
                    id=EdgeId("second-exit"),
                    source=second.id,
                    target=graph.exit.id,
                ),
            ),
        )
    definition = _workflow(
        graph,
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    output = tmp_path / "run"
    Dispatcher(project_root=tmp_path).run(
        definition,
        None,
        output_dir=output,
        checkpointing=CheckpointPolicy.REQUIRED
        if branch_failure
        else CheckpointPolicy.OFF,
    )

    assert _total(output) == ["3", "300", "60", "120", "0.750000"]
    assert [request.provider_session_id for request in invoker.requests] == [
        None,
        "session-1",
        "session-1",
    ]
    expected_action = (
        AgentSessionAction.FORK
        if branch_failure
        else AgentSessionAction.CONTINUE
    )
    assert all(
        request.provider_session_action is expected_action
        for request in invoker.requests[1:]
    )
    records = _records(output)
    assert [record["status"] for record in records] == [
        "succeeded",
        "failed",
        "succeeded",
    ]
    assert [
        cast(dict[str, object], record["usage"])["cost_usd"]
        for record in records
    ] == ["0.25"] * 3
