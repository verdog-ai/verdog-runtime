from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from types import ModuleType
from typing import NoReturn, cast

import pytest

from verdog_runtime._run_store import RunStatus, RunStore, RunStoreError
from verdog_runtime.agents import (
    AgentReply,
    AgentRequest,
    AgentSessionAction,
    AgentSessionCapabilities,
)
from verdog_runtime.declarations import (
    Agent,
    AgentNodeContext,
    AgentProfileParameter,
    AgentSessionDefinition,
    AgentSessionParameter,
    EdgeDefinition,
    GraphDefinition,
    NodeContext,
    NodeDefinition,
    PortDefinition,
    Python,
    SubroutineCall,
    SubroutineDefinition,
    Success,
    VisitDefinition,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import (
    AgentProfileId,
    AgentSessionId,
    EdgeId,
    GraphId,
    NodeId,
    ProviderSessionId,
)
from verdog_runtime.interpreter import (
    CheckpointPolicy,
    Dispatcher,
    SessionPolicy,
)
from verdog_runtime.interpreter._continuation import (
    GraphFrameSnapshot,
    decode_continuation,
)

_module_ids = count()


@dataclass(frozen=True, slots=True)
class _LifecycleState:
    visits: int = 0
    artifact: Path | None = None


@dataclass(slots=True)
class _LifecycleControl:
    fail_last: bool = True
    marker: str = "source"
    events: list[tuple[str, int, int]] = field(
        default_factory=lambda: list[tuple[str, int, int]]()
    )


def _register_workflow(
    graph: GraphDefinition[int, int, None, object],
    /,
    *,
    configuration: WorkflowConfiguration | None = None,
    sessions: tuple[AgentSessionDefinition, ...] = (),
) -> WorkflowDefinition[int, int, None, object]:
    module_name = f"runtime_test_lifecycle_{next(_module_ids)}"
    module = ModuleType(module_name)
    module.__dict__["definition"] = lambda: SubroutineDefinition(graph=graph)
    sys.modules[module_name] = module
    return WorkflowDefinition(
        id=GraphId(f"{graph.id}_workflow"),
        input_type=int,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module_name,
            params_types={(".", graph.id): type(None)},
            profile_arguments={
                parameter.id: parameter.id
                for parameter in graph.profile_parameters
            },
            session_arguments={
                parameter.id: parameter.id
                for parameter in graph.session_parameters
            },
        ),
        configuration=(
            WorkflowConfiguration() if configuration is None else configuration
        ),
        sessions=sessions,
    )


def _lifecycle_workflow(
    control: _LifecycleControl,
) -> WorkflowDefinition[int, int, None, object]:
    def first(
        input: int,
        state: _LifecycleState,
        context: NodeContext[None],
        /,
    ) -> Success[int, _LifecycleState]:
        control.events.append(("first", state.visits, input))
        artifact = context.output_dir / "first.txt"
        artifact.write_text(control.marker, encoding="utf-8")
        return Success(
            output=input + 1,
            state=_LifecycleState(state.visits + 1, artifact),
        )

    def second(
        input: int,
        state: _LifecycleState,
        context: NodeContext[None],
        /,
    ) -> Success[int, _LifecycleState]:
        control.events.append(("second", state.visits, input))
        artifact = context.output_dir / "second.txt"
        artifact.write_text(control.marker, encoding="utf-8")
        return Success(
            output=input + 1,
            state=_LifecycleState(state.visits + 1, artifact),
        )

    def last(
        input: int,
        state: _LifecycleState,
        context: NodeContext[None],
        /,
    ) -> Success[int, _LifecycleState]:
        control.events.append(("last", state.visits, input))
        if control.fail_last:
            raise RuntimeError("stop after two committed boundaries")
        artifact = context.output_dir / "last.txt"
        artifact.write_text(control.marker, encoding="utf-8")
        return Success(
            output=input + 1,
            state=_LifecycleState(state.visits + 1, artifact),
        )

    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    first_node: NodeDefinition[_LifecycleState, object] = NodeDefinition(
        id=NodeId("first"),
        name="first",
        operation=Python(),
        state_type=_LifecycleState,
    )
    second_node: NodeDefinition[_LifecycleState, object] = NodeDefinition(
        id=NodeId("second"),
        name="second",
        operation=Python(),
        state_type=_LifecycleState,
    )
    last_node: NodeDefinition[_LifecycleState, object] = NodeDefinition(
        id=NodeId("last"),
        name="last",
        operation=Python(),
        state_type=_LifecycleState,
    )
    graph: GraphDefinition[int, int, None, object] = GraphDefinition(
        id=GraphId("lifecycle"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=PortDefinition(id=NodeId("failure")),
        nodes=(first_node, second_node, last_node),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter_first"),
                source=enter.id,
                target=first_node.id,
                visit=VisitDefinition(implementation=first),
            ),
            EdgeDefinition(
                id=EdgeId("first_second"),
                source=first_node.id,
                target=second_node.id,
                visit=VisitDefinition(implementation=second),
            ),
            EdgeDefinition(
                id=EdgeId("second_last"),
                source=second_node.id,
                target=last_node.id,
                visit=VisitDefinition(implementation=last),
            ),
            EdgeDefinition(
                id=EdgeId("last_exit"), source=last_node.id, target=exit_.id
            ),
        ),
    )
    return _register_workflow(graph)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    source = project / "src"
    source.mkdir(parents=True)
    (source / "workflow.py").write_text("VERSION = 1\n", encoding="utf-8")
    return project


def _interrupted_source(
    tmp_path: Path,
) -> tuple[
    Path,
    WorkflowDefinition[int, int, None, object],
    _LifecycleControl,
    Path,
]:
    project = _project(tmp_path)
    control = _LifecycleControl()
    definition = _lifecycle_workflow(control)
    output = tmp_path / "runs" / "source"
    with pytest.raises(RuntimeError, match="two committed boundaries"):
        (
            Dispatcher(project_root=project).run(
                definition,
                10,
                output_dir=output,
                checkpointing=CheckpointPolicy.REQUIRED,
                workflow_arguments=("--input.value", "10"),
            )
        )
    store = RunStore.open(output)
    assert store.manifest().status is RunStatus.FAILED
    # The entry boundary is independently restorable, followed by the two
    # completed ordinary nodes.
    assert [item.sequence for item in store.checkpoints()] == [1, 2, 3]
    assert not tuple((output / ".verdog/checkpoints").glob("*/artifacts"))
    return project, definition, control, output


def _state_artifacts(output: Path, checkpoint: int) -> tuple[Path, ...]:
    snapshot = decode_continuation(
        RunStore.open(output).checkpoint_shard(checkpoint, "runtime.pkl")
    )
    root = cast(GraphFrameSnapshot, snapshot.frames[0])
    return tuple(
        value.artifact
        for slot in root.state
        if isinstance((value := slot.value), _LifecycleState)
        and value.artifact is not None
    )


def test_fork_uses_selected_checkpoint_and_rebases_artifacts(
    tmp_path: Path,
) -> None:
    project, definition, control, source = _interrupted_source(tmp_path)
    source_store = RunStore.open(source)
    source_manifest = source_store.manifest()
    source_checkpoint_count = source_manifest.checkpoints.count
    (source / "later.txt").write_text(
        "outside the checkpoint", encoding="utf-8"
    )
    control.fail_last = False

    control.events.clear()
    control.marker = "forked-from-one"
    first_target = tmp_path / "runs" / "fork-one"
    first_result = Dispatcher(project_root=project).fork(
        definition,
        source_output_dir=source,
        checkpoint=2,
        output_dir=first_target,
        sessions=SessionPolicy.FRESH,
    )

    assert first_result.output == 13
    assert not (first_target / "later.txt").exists()
    assert not (first_target / "last" / "000001" / "stacktrace.txt").exists()
    assert [event[0] for event in control.events] == ["second", "last"]
    first_manifest = RunStore.open(first_target).manifest()
    assert first_manifest.id != source_manifest.id
    assert first_manifest.parent is not None
    assert first_manifest.parent.run_id == source_manifest.id
    assert first_manifest.parent.operation == "fork"
    assert first_manifest.parent.checkpoint == 2
    assert first_manifest.parent.arguments == "checkpoint"
    assert first_manifest.launch.workflow_arguments == ("--input.value", "10")
    assert all(
        path.is_relative_to(first_target)
        for path in _state_artifacts(first_target, 1)
    )
    assert (first_target / "first" / "000001" / "first.txt").read_text(
        "utf-8"
    ) == "source"
    assert (first_target / "second" / "000001" / "second.txt").read_text(
        "utf-8"
    ) == "forked-from-one"

    control.events.clear()
    control.marker = "must-not-run-second"
    second_target = tmp_path / "runs" / "fork-two"
    second_result = Dispatcher(project_root=project).fork(
        definition,
        source_output_dir=source,
        checkpoint=3,
        output_dir=second_target,
        sessions=SessionPolicy.FRESH,
    )

    assert second_result.output == 13
    assert [event[0] for event in control.events] == ["last"]
    second_manifest = RunStore.open(second_target).manifest()
    assert second_manifest.id not in {source_manifest.id, first_manifest.id}
    assert second_manifest.parent is not None
    assert second_manifest.parent.checkpoint == 3
    assert all(
        path.is_relative_to(second_target)
        for path in _state_artifacts(second_target, 1)
    )
    assert (second_target / "second" / "000001" / "second.txt").read_text(
        "utf-8"
    ) == "source"
    assert (
        RunStore.open(source).manifest().checkpoints.count
        == source_checkpoint_count
    )


@pytest.mark.parametrize("operation", ("resume", "fork"))
@pytest.mark.parametrize(
    ("change", "code"),
    (
        ("delete", "checkpoint.artifact_unavailable"),
        ("modify", "checkpoint.artifact_corrupt"),
    ),
)
def test_resume_and_fork_reject_changed_published_artifacts(
    tmp_path: Path, operation: str, change: str, code: str
) -> None:
    project, definition, control, source = _interrupted_source(tmp_path)
    artifact = source / "first" / "000001" / "first.txt"
    if change == "delete":
        artifact.unlink()
    else:
        # Same-size edits must be detected by content verification.
        artifact.write_text("edited", encoding="utf-8")
    control.fail_last = False
    control.events.clear()
    target = tmp_path / "runs" / "invalid-fork"

    with pytest.raises(RunStoreError) as captured:
        dispatcher = Dispatcher(project_root=project)
        if operation == "resume":
            dispatcher.resume(definition, output_dir=source)
        else:
            dispatcher.fork(
                definition,
                source_output_dir=source,
                checkpoint=2,
                output_dir=target,
                sessions=SessionPolicy.FRESH,
            )

    assert captured.value.code == code
    assert not control.events
    assert not target.exists()


def test_resume_reuses_outputs_and_preserves_uncommitted_files(
    tmp_path: Path,
) -> None:
    project, definition, control, source = _interrupted_source(tmp_path)
    first = source / "first" / "000001" / "first.txt"
    original_inode = first.stat().st_ino
    uncommitted = source / "last" / "000001" / "attempt.txt"
    uncommitted.write_text("interrupted attempt", encoding="utf-8")
    control.fail_last = False
    control.events.clear()

    result = Dispatcher(project_root=project).resume(
        definition, output_dir=source
    )

    assert result.output == 13
    assert [event[0] for event in control.events] == ["last"]
    assert first.stat().st_ino == original_inode
    assert first.read_text("utf-8") == "source"
    assert uncommitted.read_text("utf-8") == "interrupted attempt"
    assert (source / "last" / "000002" / "last.txt").read_text(
        "utf-8"
    ) == "source"
    assert not tuple((source / ".verdog/checkpoints").glob("*/artifacts"))


def test_materialized_fork_resumes_without_source_files(tmp_path: Path) -> None:
    project, definition, control, source = _interrupted_source(tmp_path)
    target = tmp_path / "runs" / "independent-fork"
    with pytest.raises(RuntimeError, match="two committed boundaries"):
        Dispatcher(project_root=project).fork(
            definition,
            source_output_dir=source,
            checkpoint=3,
            output_dir=target,
            sessions=SessionPolicy.FRESH,
        )
    shutil.rmtree(source)
    control.fail_last = False
    control.events.clear()

    result = Dispatcher(project_root=project).resume(
        definition, output_dir=target
    )

    assert result.output == 13
    assert [event[0] for event in control.events] == ["last"]
    assert (target / "first" / "000001" / "first.txt").read_text(
        "utf-8"
    ) == "source"
    assert (target / "second" / "000001" / "second.txt").read_text(
        "utf-8"
    ) == "source"
    assert not tuple((target / ".verdog/checkpoints").glob("*/artifacts"))


def test_restart_creates_lineage_but_resets_workflow_state(
    tmp_path: Path,
) -> None:
    project, definition, control, source = _interrupted_source(tmp_path)
    source_manifest = RunStore.open(source).manifest()
    control.fail_last = False

    control.events.clear()
    fresh_target = tmp_path / "runs" / "restart-fresh"
    fresh_result = Dispatcher(project_root=project).restart(
        definition,
        20,
        source_output_dir=source,
        output_dir=fresh_target,
        sessions=SessionPolicy.FRESH,
        workflow_arguments=("--input.value", "10"),
        arguments_mode="reused",
    )

    assert fresh_result.output == 23
    assert control.events == [
        ("first", 0, 20),
        ("second", 0, 21),
        ("last", 0, 22),
    ]
    fresh_manifest = RunStore.open(fresh_target).manifest()
    assert fresh_manifest.id != source_manifest.id
    assert fresh_manifest.parent is not None
    assert fresh_manifest.parent.run_id == source_manifest.id
    assert fresh_manifest.parent.operation == "restart"
    assert fresh_manifest.parent.checkpoint is None
    assert fresh_manifest.parent.arguments == "reused"

    # Simulate a newer serializable boundary whose nested/provider sessions
    # cannot be branched. The default must use the previous usable anchor.
    latest_manifest = (
        RunStore.open(source).checkpoint_directory(3) / "manifest.json"
    )
    latest = cast(
        dict[str, object], json.loads(latest_manifest.read_text("utf-8"))
    )
    latest["fork_with_branch_available"] = False
    latest_manifest.write_text(json.dumps(latest), encoding="utf-8")

    control.events.clear()
    branch_target = tmp_path / "runs" / "restart-branch"
    branch_result = Dispatcher(project_root=project).restart(
        definition,
        30,
        source_output_dir=source,
        output_dir=branch_target,
        sessions=SessionPolicy.BRANCH,
        workflow_arguments=("--input.value", "30"),
        arguments_mode="overridden",
    )

    assert branch_result.output == 33
    assert control.events == [
        ("first", 0, 30),
        ("second", 0, 31),
        ("last", 0, 32),
    ]
    branch_manifest = RunStore.open(branch_target).manifest()
    assert branch_manifest.parent is not None
    assert branch_manifest.parent.operation == "restart"
    assert branch_manifest.parent.checkpoint == 2
    assert branch_manifest.parent.arguments == "overridden"
    assert branch_manifest.launch.workflow_arguments == ("--input.value", "30")

    with pytest.raises(ValueError, match="checkpoint does not exist: 0"):
        (
            Dispatcher(project_root=project).restart(
                definition,
                40,
                source_output_dir=source,
                output_dir=tmp_path / "runs" / "restart-invalid-checkpoint",
                sessions=SessionPolicy.BRANCH,
                source_checkpoint=0,
            )
        )


@dataclass(slots=True)
class _BranchingInvoker:
    session_provider = "lifecycle-test"
    session_capabilities = AgentSessionCapabilities(fork_latest=True)

    requests: list[AgentRequest] = field(default_factory=list[AgentRequest])

    def __call__(self, request: AgentRequest, /) -> AgentReply:
        self.requests.append(request)
        return AgentReply(
            text=f"reply-{len(self.requests)}",
            provider_session_id=ProviderSessionId(
                f"provider-session-{len(self.requests)}"
            ),
        )


def _session_workflow(
    project: Path,
    invoker: _BranchingInvoker,
    control: _LifecycleControl,
) -> WorkflowDefinition[int, int, None, object]:
    def agent(
        input: int,
        state: _LifecycleState,
        context: AgentNodeContext[None],
        /,
    ) -> Success[int, _LifecycleState]:
        context.invoke(
            str(context.node_id),
            workspace=project,
        )
        return Success(output=input + 1, state=state)

    def gate(
        input: int,
        state: _LifecycleState,
        _context: NodeContext[None],
        /,
    ) -> Success[int, _LifecycleState]:
        if control.fail_last:
            raise RuntimeError("stop between agent calls")
        return Success(output=input + 1, state=state)

    profile_id = AgentProfileId("profile")
    session_id = AgentSessionId("conversation")
    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    first: NodeDefinition[_LifecycleState, object] = NodeDefinition(
        id=NodeId("first_agent"),
        name="first agent",
        operation=Agent(profile=profile_id, session=session_id),
        state_type=_LifecycleState,
    )
    gate_node: NodeDefinition[_LifecycleState, object] = NodeDefinition(
        id=NodeId("gate"),
        name="gate",
        operation=Python(),
        state_type=_LifecycleState,
    )
    second: NodeDefinition[_LifecycleState, object] = NodeDefinition(
        id=NodeId("second_agent"),
        name="second agent",
        operation=Agent(profile=profile_id, session=session_id),
        state_type=_LifecycleState,
    )
    graph: GraphDefinition[int, int, None, object] = GraphDefinition(
        id=GraphId("session_lifecycle"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=PortDefinition(id=NodeId("failure")),
        nodes=(first, gate_node, second),
        profile_parameters=(
            AgentProfileParameter(id=profile_id, name="profile"),
        ),
        session_parameters=(
            AgentSessionParameter(id=session_id, name="conversation"),
        ),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter_first"),
                source=enter.id,
                target=first.id,
                visit=VisitDefinition(implementation=agent),
            ),
            EdgeDefinition(
                id=EdgeId("first_gate"),
                source=first.id,
                target=gate_node.id,
                visit=VisitDefinition(implementation=gate),
            ),
            EdgeDefinition(
                id=EdgeId("gate_second"),
                source=gate_node.id,
                target=second.id,
                visit=VisitDefinition(implementation=agent),
            ),
            EdgeDefinition(
                id=EdgeId("second_exit"), source=second.id, target=exit_.id
            ),
        ),
    )
    return _register_workflow(
        graph,
        configuration=WorkflowConfiguration(
            profile_arguments={profile_id: invoker}
        ),
        sessions=(
            AgentSessionDefinition(
                id=session_id,
                name="conversation",
                persistent=True,
            ),
        ),
    )


def test_fork_applies_branch_and_fresh_conversation_policies(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    invoker = _BranchingInvoker()
    control = _LifecycleControl()
    definition = _session_workflow(project, invoker, control)
    source = tmp_path / "runs" / "session-source"

    with pytest.raises(RuntimeError, match="between agent calls"):
        (
            Dispatcher(project_root=project).run(
                definition,
                1,
                output_dir=source,
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )
    assert len(invoker.requests) == 1
    assert invoker.requests[0].provider_session_id is None
    assert (
        invoker.requests[0].provider_session_action
        is AgentSessionAction.CONTINUE
    )
    source_store = RunStore.open(source)
    assert source_store.checkpoints()[1].fork_with_branch_available
    control.fail_last = False

    branch_target = tmp_path / "runs" / "session-branch"
    branch_result = Dispatcher(project_root=project).fork(
        definition,
        source_output_dir=source,
        checkpoint=2,
        output_dir=branch_target,
        sessions=SessionPolicy.BRANCH,
    )

    assert branch_result.output == 4
    branch_request = invoker.requests[1]
    assert branch_request.provider_session_id == ProviderSessionId(
        "provider-session-1"
    )
    assert branch_request.provider_session_action is AgentSessionAction.FORK
    branch_snapshot = decode_continuation(
        RunStore.open(branch_target).checkpoint_shard(1, "runtime.pkl")
    )
    assert (
        branch_snapshot.sessions[0].provider_session_id == "provider-session-1"
    )
    assert branch_snapshot.sessions[0].copy_on_write

    fresh_target = tmp_path / "runs" / "session-fresh"
    fresh_result = Dispatcher(project_root=project).fork(
        definition,
        source_output_dir=source,
        checkpoint=2,
        output_dir=fresh_target,
        sessions=SessionPolicy.FRESH,
    )

    assert fresh_result.output == 4
    fresh_request = invoker.requests[2]
    assert fresh_request.provider_session_id is None
    assert fresh_request.provider_session_action is AgentSessionAction.CONTINUE
    fresh_snapshot = decode_continuation(
        RunStore.open(fresh_target).checkpoint_shard(1, "runtime.pkl")
    )
    assert fresh_snapshot.sessions[0].provider is None
    assert fresh_snapshot.sessions[0].provider_session_id is None
    assert not fresh_snapshot.sessions[0].copy_on_write


def test_resume_and_fork_reject_exact_compatibility_drift_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, definition, _control, source = _interrupted_source(tmp_path)
    (project / "src" / "workflow.py").write_text(
        "VERSION = 2\n", encoding="utf-8"
    )

    def unexpected_decode(_payload: bytes) -> NoReturn:
        raise AssertionError(
            "compatibility drift must be rejected before decoding"
        )

    monkeypatch.setattr(
        "verdog_runtime.interpreter._continuation.decode_continuation",
        unexpected_decode,
    )

    with pytest.raises(ValueError, match=r"changed source_sha256$"):
        (
            Dispatcher(project_root=project).resume(
                definition,
                output_dir=source,
            )
        )

    target = tmp_path / "runs" / "incompatible-fork"
    with pytest.raises(ValueError, match=r"changed source_sha256$"):
        (
            Dispatcher(project_root=project).fork(
                definition,
                source_output_dir=source,
                checkpoint=2,
                output_dir=target,
                sessions=SessionPolicy.FRESH,
            )
        )
    assert not target.exists()


def test_recorded_run_without_checkpointing_skips_compatibility_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    control = _LifecycleControl(fail_last=False)
    definition = _lifecycle_workflow(control)

    def unexpected_fingerprint(_project: Path) -> NoReturn:
        raise AssertionError(
            "a non-checkpointed run does not need a fingerprint"
        )

    monkeypatch.setattr(
        "verdog_runtime._checkpoint_compatibility.checkpoint_compatibility",
        unexpected_fingerprint,
    )
    output = tmp_path / "runs" / "recorded"
    result = Dispatcher(project_root=project).run(
        definition,
        10,
        output_dir=output,
        checkpointing=CheckpointPolicy.OFF,
        _record_run=True,
    )

    assert result.output == 13
    assert RunStore.open(output).manifest().compatibility == {}
