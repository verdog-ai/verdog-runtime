from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Never, cast, override

import pytest
import cloudpickle
from layout_helpers import legacy_frame_state, legacy_graph_create
from report_helpers import call_reports, table_rows
from verdog_runtime._run_store import (
    CheckpointKind,
    CheckpointPolicy,
    RunStatus,
    RunStore,
)
from verdog_runtime.declarations import (
    CallContext,
    CallVisitDefinition,
    EdgeDefinition,
    FeatureDefinition,
    FeatureKind,
    GraphDefinition,
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
from verdog_runtime.declarations.ids import EdgeId, FeatureId, GraphId, NodeId, RunId
from verdog_runtime.interpreter import Dispatcher, SessionPolicy, initial_workflow_state
from verdog_runtime.interpreter._continuation import (
    FORMAT_VERSION,
    CallFrameSnapshot,
    ContinuationSnapshot,
    GraphFrameSnapshot,
    StateSlot,
    continuation_digest,
    decode_continuation,
    encode_continuation,
    fork_continuation,
    restore_workflow_state,
    snapshot_workflow_state,
)


@dataclass(frozen=True, slots=True)
class Count:
    value: int = 0


class UnserializableChildError(RuntimeError):
    @override
    def __reduce__(self) -> Never:
        raise TypeError("child error cannot be serialized")


class UnserializableValue:
    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnserializableValue)

    @override
    def __reduce__(self) -> Never:
        raise TypeError("call request cannot be serialized")


@dataclass(frozen=True, slots=True)
class ForkInput:
    artifact: Path
    external: Path


@dataclass(frozen=True, slots=True)
class ForkParams:
    root: Path = Path(".")
    adapter_run_id: str = ""


def _graph() -> tuple[
    GraphDefinition[object, object, None, object],
    NodeDefinition[Count, object],
    FeatureDefinition[int, object],
]:
    node = NodeDefinition[Count, object](
        id=NodeId("work"),
        name="work",
        operation=Python(),
        state_type=Count,
    )
    feature = FeatureDefinition[int, object](
        id=FeatureId("remaining"),
        label="remaining",
        description="remaining work",
        kind=FeatureKind.INTEGER,
    )
    return (
        GraphDefinition(
            id=GraphId("continuation"),
            params_type=type(None),
            enter=PortDefinition(id=NodeId("enter")),
            exit=PortDefinition(id=NodeId("exit")),
            failure=PortDefinition(id=NodeId("failure")),
            nodes=(node,),
            edges=(),
            features=(feature,),
        ),
        node,
        feature,
    )


def test_state_snapshot_rebinds_to_fresh_definition_objects() -> None:
    first, first_node, first_feature = _graph()
    state = initial_workflow_state(first)
    state = state._replace(first_node, Count(3))  # pyright: ignore[reportPrivateUsage]
    state = state._replace(first_feature, 2)  # pyright: ignore[reportPrivateUsage]

    slots = snapshot_workflow_state(state)
    second, second_node, second_feature = _graph()
    restored = restore_workflow_state(second, slots)

    assert restored.get(second_node) == Count(3)
    assert restored.get(second_feature) == 2
    with pytest.raises(KeyError, match="unknown workflow state key"):
        restored.get(first_node)


def test_state_restore_rejects_address_and_type_drift() -> None:
    graph, _, _ = _graph()
    slots = snapshot_workflow_state(initial_workflow_state(graph))

    with pytest.raises(ValueError, match="addresses do not match"):
        restore_workflow_state(
            graph,
            (*slots, StateSlot(address=("node", "unknown"), value=Count())),
        )
    changed = tuple(
        StateSlot(address=slot.address, value="wrong")
        if slot.address == ("node", "work")
        else slot
        for slot in slots
    )
    with pytest.raises(TypeError, match="expected Count"):
        restore_workflow_state(graph, changed)


def test_continuation_codec_is_versioned_and_digested() -> None:
    snapshot = ContinuationSnapshot(
        format_version=FORMAT_VERSION,
        run_id=RunId("run"),
        transitions_remaining=7,
        frames=(),
        sessions=(),
    )

    payload = encode_continuation(snapshot)

    assert decode_continuation(payload) == snapshot
    assert len(continuation_digest(payload)) == 64

    with pytest.raises(ValueError, match="unsupported continuation format"):
        encode_continuation(replace(snapshot, format_version=FORMAT_VERSION - 1))

    from verdog_runtime.interpreter._continuation import ParameterSlot, SessionSnapshot

    duplicated_parameters = replace(
        snapshot,
        parameters=(
            ParameterSlot(address=(".", GraphId("graph")), value=None),
            ParameterSlot(address=(".", GraphId("graph")), value=None),
        ),
    )
    with pytest.raises(ValueError, match="parameter addresses must be unique"):
        encode_continuation(duplicated_parameters)

    invalid = replace(
        snapshot,
        sessions=(
            SessionSnapshot(
                resource_id="session-1",
                persistent=False,
                provider="codex",
                provider_session_id="provider-session",
                access="read-only",
            ),
        ),
    )
    with pytest.raises(ValueError, match="provider session anchor is invalid"):
        encode_continuation(invalid)

    malformed = replace(snapshot, sessions=cast(Any, (object(),)))
    with pytest.raises(ValueError, match="continuation sessions are invalid"):
        encode_continuation(malformed)


def test_continuation_rejects_unknown_call_phase() -> None:
    from verdog_runtime.interpreter._continuation import DefinitionReference

    call = CallFrameSnapshot(
        frame_id="call",
        parent_graph_frame_id="parent",
        node_id=NodeId("call"),
        incoming_edge_id=EdgeId("incoming"),
        visit_path="graph-parent/call/000001",
        adapter_run_id=RunId("adapter"),
        operation=DefinitionReference(
            kind="subroutine",
            id=GraphId("child"),
            module="example.child",
            project_path=".",
        ),
        input=None,
        prior_state=None,
        child_input=None,
        child_params=None,
        child_params_override=False,
        phase=cast(Any, "unknown"),
    )
    snapshot = ContinuationSnapshot(
        format_version=FORMAT_VERSION,
        run_id=RunId("run"),
        transitions_remaining=1,
        frames=(call,),
        sessions=(),
    )

    with pytest.raises(ValueError, match="call phase is invalid"):
        encode_continuation(snapshot)


@pytest.mark.parametrize(
    ("child_output", "child_error", "child_call_path", "message"),
    (
        (1, RuntimeError("failed"), None, "both output and error"),
        (None, cast(Any, "not-an-exception"), None, "invalid child error"),
        (None, None, "/absolute/attempt-000001", "invalid child path"),
        (None, None, "../attempt-000001", "invalid child path"),
        (None, None, "graph-parent//attempt-000001", "invalid child path"),
    ),
)
def test_continuation_rejects_invalid_returned_call_state(
    child_output: object,
    child_error: Exception | Any | None,
    child_call_path: str | None,
    message: str,
) -> None:
    from verdog_runtime.interpreter._continuation import DefinitionReference

    call = CallFrameSnapshot(
        frame_id="call",
        parent_graph_frame_id="parent",
        node_id=NodeId("call"),
        incoming_edge_id=EdgeId("incoming"),
        visit_path="graph-parent/call/000001",
        adapter_run_id=RunId("adapter"),
        operation=DefinitionReference(
            kind="subroutine",
            id=GraphId("child"),
            module="example.child",
            project_path=".",
        ),
        input=None,
        prior_state=None,
        child_input=None,
        child_params=None,
        child_params_override=False,
        phase="child_returned",
        child_activation_id="activation",
        child_graph_frame_id="activation",
        child_call_path=child_call_path,
        child_output=child_output,
        child_error=child_error,
    )
    snapshot = ContinuationSnapshot(
        format_version=FORMAT_VERSION,
        run_id=RunId("run"),
        transitions_remaining=1,
        frames=(call,),
        sessions=(),
    )

    with pytest.raises(ValueError, match=message):
        encode_continuation(snapshot)


@dataclass(frozen=True, slots=True)
class _PathState:
    artifact: Path
    external: Path


@pytest.mark.parametrize("legacy_slot", [False, True])
def test_fork_rebases_run_paths_and_applies_session_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_slot: bool,
) -> None:
    from verdog_runtime.interpreter._continuation import (
        DefinitionReference,
        GraphFrameSnapshot,
        Ready,
        SessionSnapshot,
    )

    source = (tmp_path / "source").resolve()
    target = (tmp_path / "target").resolve()
    external = (tmp_path / "external.txt").resolve()
    frame = GraphFrameSnapshot(
        frame_id="root",
        definition=DefinitionReference(
            kind="subroutine",
            id=GraphId("graph"),
            module="example.graph",
            project_path=".",
        ),
        scope_current=GraphId("graph"),
        scope_root=GraphId("graph"),
        scope_root_workflow_id=GraphId("workflow"),
        call_path="graph-graph",
        entry_input=_PathState(source / "input.txt", external),
        params=None,
        value=source / "value.txt",
        state=(),
        control=Ready(
            incoming_edge_id=EdgeId("enter_work"), target_node_id=NodeId("work")
        ),
        visits=(),
        session_bindings=(("conversation", "session-1"),),
    )
    snapshot = ContinuationSnapshot(
        format_version=FORMAT_VERSION,
        run_id=RunId("source-run"),
        transitions_remaining=7,
        frames=(frame,),
        sessions=(
            SessionSnapshot(
                resource_id="session-1",
                persistent=True,
                provider="codex",
                provider_session_id="provider-source",
                access="read-only",
                copy_on_write=False,
                branch_supported=True,
            ),
        ),
    )

    if legacy_slot:
        # Slotted dataclasses encode fields in order. Older pickles have no
        # state entry for the report_path field appended by the new runtime.
        with monkeypatch.context() as legacy:
            legacy.setattr(GraphFrameSnapshot, "__getstate__", legacy_frame_state)
            payload = encode_continuation(snapshot)
        unnormalized = cast(ContinuationSnapshot, cloudpickle.loads(payload))
        assert not hasattr(unnormalized.frames[0], "report_path")
        snapshot = decode_continuation(payload)
        assert cast(GraphFrameSnapshot, snapshot.frames[0]).report_path is None

    branched = fork_continuation(
        snapshot,
        run_id=RunId("branch-run"),
        source_output=source,
        target_output=target,
        sessions=SessionPolicy.BRANCH,
    )
    branched_frame = cast(GraphFrameSnapshot, branched.frames[0])
    assert branched.run_id == RunId("branch-run")
    assert branched_frame.entry_input == _PathState(target / "input.txt", external)
    assert branched_frame.value == target / "value.txt"
    assert branched.sessions[0].provider_session_id == "provider-source"
    assert branched.sessions[0].copy_on_write

    fresh = fork_continuation(
        snapshot,
        run_id=RunId("fresh-run"),
        source_output=source,
        target_output=target,
        sessions=SessionPolicy.FRESH,
    )
    assert fresh.sessions[0].provider is None
    assert fresh.sessions[0].provider_session_id is None
    assert fresh.sessions[0].access is None
    assert not fresh.sessions[0].copy_on_write


def test_fork_rejects_unbranchable_sessions_and_rebased_key_collisions(
    tmp_path: Path,
) -> None:
    from verdog_runtime.interpreter._continuation import (
        DefinitionReference,
        GraphFrameSnapshot,
        Ready,
        SessionSnapshot,
    )

    source = (tmp_path / "source").resolve()
    target = (tmp_path / "target").resolve()
    frame = GraphFrameSnapshot(
        frame_id="root",
        definition=DefinitionReference(
            kind="subroutine",
            id=GraphId("graph"),
            module="example.graph",
            project_path=".",
        ),
        scope_current=GraphId("graph"),
        scope_root=GraphId("graph"),
        scope_root_workflow_id=GraphId("workflow"),
        call_path="graph-graph",
        entry_input=None,
        params=None,
        value=None,
        state=(),
        control=Ready(
            incoming_edge_id=EdgeId("enter_work"), target_node_id=NodeId("work")
        ),
        visits=(),
        session_bindings=(("conversation", "session-1"),),
    )
    snapshot = ContinuationSnapshot(
        format_version=FORMAT_VERSION,
        run_id=RunId("source-run"),
        transitions_remaining=7,
        frames=(frame,),
        sessions=(
            SessionSnapshot(
                resource_id="session-1",
                persistent=True,
                provider="legacy",
                provider_session_id="provider-source",
                access="read-only",
                branch_supported=False,
            ),
        ),
    )

    with pytest.raises(ValueError, match="cannot be branched independently"):
        fork_continuation(
            snapshot,
            run_id=RunId("branch-run"),
            source_output=source,
            target_output=target,
            sessions=SessionPolicy.BRANCH,
        )

    colliding = replace(
        snapshot,
        frames=(
            replace(
                frame,
                value={source / "artifact.txt": 1, target / "artifact.txt": 2},
                session_bindings=(),
            ),
        ),
        sessions=(),
    )
    with pytest.raises(ValueError, match="mapping keys collide"):
        fork_continuation(
            colliding,
            run_id=RunId("fresh-run"),
            source_output=source,
            target_output=target,
            sessions=SessionPolicy.FRESH,
        )


def test_returned_call_fork_rebases_journal_and_restores_all_parameters(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "path-source").resolve()
    target = (tmp_path / "path-target").resolve()
    external = (tmp_path / "external.txt").resolve()
    definition, child_id, child_calls, control = _path_call_workflow(
        "rebase",
        source,
    )
    input = ForkInput(source / "input.txt", external)

    with pytest.raises(KeyboardInterrupt, match="path call after child return"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                input,
                output_dir=source,
                params={(".", child_id): ForkParams(source / "configured.txt")},
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )

    source_store = RunStore.open(source)
    checkpoint = source_store.manifest().checkpoints.latest_restorable
    assert checkpoint is not None
    snapshot = decode_continuation(
        source_store.checkpoint_shard(checkpoint, "runtime.pkl")
    )
    call_frame = snapshot.frames[-1]
    assert isinstance(call_frame, CallFrameSnapshot)
    assert call_frame.phase == "child_returned"
    assert call_frame.adapter_run_id == RunId(source_store.manifest().id)
    assert call_frame.input == input
    assert call_frame.prior_state.path == source / "state.txt"  # type: ignore[attr-defined]
    assert call_frame.child_input == input
    assert call_frame.child_params == ForkParams(
        source / "override.txt",
        adapter_run_id=source_store.manifest().id,
    )
    assert call_frame.child_output == source / "child-output.txt"
    configured = {slot.address: slot.value for slot in snapshot.parameters}[
        (".", child_id)
    ]
    assert configured == ForkParams(source / "configured.txt")
    assert child_calls == [input]

    control["interrupt"] = False
    forked = Dispatcher(project_root=tmp_path).fork(
        definition,
        source_output_dir=source,
        checkpoint=checkpoint,
        output_dir=target,
        sessions=SessionPolicy.FRESH,
    )
    resumed = Dispatcher(project_root=tmp_path).resume(
        definition,
        output_dir=source,
    )

    assert forked.output == target / "child-output.txt"
    assert resumed.output == source / "child-output.txt"
    assert child_calls == [input]
    assert RunStore.open(target).manifest().status is RunStatus.SUCCEEDED
    assert RunStore.open(source).manifest().status is RunStatus.SUCCEEDED


def _preserve(
    input: object, state: Count, _context: object, /
) -> Success[object, Count]:
    return Success(output=input, state=Count(state.value + 1))


def test_dispatcher_commits_each_completed_boundary(tmp_path: Path) -> None:
    graph, node, feature = _graph()
    graph: GraphDefinition[object, object, None, object] = GraphDefinition(
        id=graph.id,
        params_type=graph.params_type,
        enter=graph.enter,
        exit=graph.exit,
        failure=graph.failure,
        nodes=graph.nodes,
        features=(feature,),
        edges=(
            # The feature stays uninitialized; it is still part of the snapshot.
            EdgeDefinition(
                id=EdgeId("enter_work"),
                source=graph.enter.id,
                target=node.id,
                visit=VisitDefinition(implementation=_preserve),
            ),
            EdgeDefinition(
                id=EdgeId("work_exit"),
                source=node.id,
                target=graph.exit.id,
            ),
        ),
    )
    module_name = "runtime_test_checkpoint_entry"
    module = ModuleType(module_name)
    module.definition = lambda: SubroutineDefinition(graph=graph)  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    definition: WorkflowDefinition[object, object, None, object] = WorkflowDefinition(
        id=GraphId("checkpoint_workflow"),
        input_type=object,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module_name,
            params_types={(".", graph.id): type(None)},
            profile_arguments={},
            session_arguments={},
        ),
        configuration=WorkflowConfiguration(),
    )
    output = tmp_path / "run"

    result = Dispatcher(project_root=tmp_path, transition_limit=10).run(
        definition,
        "input",
        output_dir=output,
        checkpointing=CheckpointPolicy.AUTO,
    )

    assert result.output == "input"
    store = RunStore.open(output)
    manifest = store.manifest()
    assert manifest.status is RunStatus.SUCCEEDED
    checkpoints = store.checkpoints()
    assert [item.sequence for item in checkpoints] == [1, 2]
    assert all(item.restore_available for item in checkpoints)
    terminal = decode_continuation(store.checkpoint_shard(2, "runtime.pkl"))
    assert terminal.transitions_remaining == 8
    assert len(terminal.frames) == 1


def test_resume_retries_only_the_unfinished_node(tmp_path: Path) -> None:
    calls: list[str] = []
    fail = True

    def first(input: int, state: Count, _context: object, /) -> Success[int, Count]:
        calls.append("first")
        return Success(output=input + 1, state=Count(state.value + 1))

    def second(input: int, state: Count, _context: object, /) -> Success[int, Count]:
        nonlocal fail
        calls.append("second")
        if fail:
            raise RuntimeError("interrupted second node")
        return Success(output=input + 1, state=Count(state.value + 1))

    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    first_node: NodeDefinition[Count, object] = NodeDefinition(
        id=NodeId("first"), name="first", operation=Python(), state_type=Count
    )
    second_node: NodeDefinition[Count, object] = NodeDefinition(
        id=NodeId("second"), name="second", operation=Python(), state_type=Count
    )
    graph: GraphDefinition[int, int, None, object] = GraphDefinition(
        id=GraphId("resumable"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(first_node, second_node),
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
                id=EdgeId("second_exit"),
                source=second_node.id,
                target=exit_.id,
            ),
        ),
    )
    module_name = "runtime_test_resumable_entry"
    module = ModuleType(module_name)
    module.definition = lambda: SubroutineDefinition(graph=graph)  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    definition: WorkflowDefinition[int, int, None, object] = WorkflowDefinition(
        id=GraphId("resumable_workflow"),
        input_type=int,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module_name,
            params_types={(".", graph.id): type(None)},
            profile_arguments={},
            session_arguments={},
        ),
        configuration=WorkflowConfiguration(),
    )
    output = tmp_path / "resumable-run"

    with pytest.raises(RuntimeError, match="interrupted second"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=output,
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )
    assert calls == ["first", "second"]
    assert RunStore.open(output).manifest().status is RunStatus.FAILED

    fail = False
    result = Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)

    assert result.output == 3
    assert calls == ["first", "second", "second"]
    assert RunStore.open(output).manifest().status is RunStatus.SUCCEEDED
    second_visits = output / "second"
    assert sorted(path.name for path in second_visits.iterdir()) == [
        "000001",
        "000002",
    ]


@dataclass(slots=True, kw_only=True)
class _NestedCallScenario:
    adapter_mode: str
    calls: list[str]
    failures: dict[str, bool]
    evaluations: int = 0

    def child_first(
        self, input: int, state: Count, _context: object, /
    ) -> Success[int, Count]:
        self.calls.append("child_first")
        if self.adapter_mode == "mutable":
            mutable = cast(list[int], cast(object, input))
            mutable.append(99)
            return Success(output=len(mutable), state=Count(state.value + 1))
        return Success(output=input + 1, state=Count(state.value + 1))

    def child_second(
        self, input: int, state: Count, _context: object, /
    ) -> Success[int, Count]:
        self.calls.append("child_second")
        if self.failures["child"]:
            raise KeyboardInterrupt("nested child interrupted")
        if self.failures["unserializable_child_exception"]:
            raise UnserializableChildError("unserializable child failure")
        if self.failures["child_exception"]:
            raise ValueError("recorded child failure")
        return Success(output=input + 1, state=Count(state.value + 1))

    def _child_input(self, input: int, /) -> object:
        if self.adapter_mode == "mutable":
            return [input]
        if self.adapter_mode in {
            "diverge",
            "swallow_diverge",
            "swallow_diverge_base",
        }:
            return input + self.evaluations - 1
        return input

    def _child_params(self, context: CallContext[None, None, int, int], /) -> object:
        if self.adapter_mode == "nan":
            return float("nan")
        if self.adapter_mode == "unserializable_request":
            return UnserializableValue()
        return context.child_params

    @staticmethod
    def _invoke_child(
        context: CallContext[None, None, int, int],
        child_input: object,
        selected_params: object,
        /,
    ) -> int:
        if selected_params is context.child_params:
            return context.invoke(cast(Any, child_input))
        return context.invoke(
            cast(Any, child_input),
            params=cast(Any, selected_params),
        )

    def _recover_value_error(self, state: Count, /) -> Success[int, Count]:
        if not self.failures["catch_child_exception"]:
            raise
        self.calls.append("caught")
        return Success(output=97, state=Count(state.value + 1))

    def _recover_divergence(self, state: Count, /) -> Success[int, Count]:
        if self.adapter_mode != "swallow_diverge":
            raise
        self.calls.append("swallowed_divergence")
        return Success(output=98, state=Count(state.value + 1))

    def _recover_base_divergence(self, state: Count, /) -> Success[int, Count]:
        if self.adapter_mode != "swallow_diverge_base" or self.evaluations == 1:
            raise
        self.calls.append("swallowed_base_divergence")
        return Success(output=98, state=Count(state.value + 1))

    def _repeat_invoke(
        self,
        input: int,
        state: Count,
        context: CallContext[None, None, int, int],
        /,
    ) -> Success[int, Count] | None:
        if self.adapter_mode == "two":
            context.invoke(input)
        if self.adapter_mode != "swallow_two":
            return None
        try:
            context.invoke(input)
        except BaseException:
            self.calls.append("swallowed_second_invoke")
            return Success(output=99, state=Count(state.value + 1))
        return None

    def call(
        self,
        input: int,
        state: Count,
        context: CallContext[None, None, int, int],
        /,
    ) -> Success[int, Count]:
        self.evaluations += 1
        self.calls.append("call")
        if self.adapter_mode == "zero":
            return Success(output=input, state=state)
        try:
            child_output = self._invoke_child(
                context,
                self._child_input(input),
                self._child_params(context),
            )
        except ValueError:
            return self._recover_value_error(state)
        except Exception:
            return self._recover_divergence(state)
        except BaseException:
            return self._recover_base_divergence(state)
        repeated = self._repeat_invoke(input, state, context)
        if repeated is not None:
            return repeated
        if self.failures["complete"]:
            raise KeyboardInterrupt("nested completion interrupted")
        return Success(output=child_output + 10, state=Count(state.value + 1))

    async def async_call(
        self,
        input: int,
        state: Count,
        _context: CallContext[None, None, int, int],
        /,
    ) -> Success[int, Count]:
        return Success(output=input, state=state)


def _nested_workflow(
    module_suffix: str,
    *,
    fail_child: bool = False,
    fail_complete: bool = False,
    child_exception: bool = False,
    catch_child_exception: bool = False,
    adapter_mode: str = "normal",
    unserializable_child_exception: bool = False,
) -> tuple[WorkflowDefinition[int, int, None, object], list[str], dict[str, bool]]:
    scenario = _NestedCallScenario(
        adapter_mode=adapter_mode,
        calls=[],
        failures={
            "child": fail_child,
            "complete": fail_complete,
            "child_exception": child_exception,
            "catch_child_exception": catch_child_exception,
            "unserializable_child_exception": unserializable_child_exception,
        },
    )

    child_enter = PortDefinition(id=NodeId("child_enter"))
    child_exit = PortDefinition(id=NodeId("child_exit"))
    child_failure = PortDefinition(id=NodeId("child_failure"))
    first: NodeDefinition[Count, object] = NodeDefinition(
        id=NodeId("child_first"),
        name="child first",
        operation=Python(),
        state_type=Count,
    )
    second: NodeDefinition[Count, object] = NodeDefinition(
        id=NodeId("child_second"),
        name="child second",
        operation=Python(),
        state_type=Count,
    )
    child: GraphDefinition[int, int, None, object] = GraphDefinition(
        id=GraphId(f"nested_{module_suffix}__child"),
        params_type=type(None),
        enter=child_enter,
        exit=child_exit,
        failure=child_failure,
        nodes=(first, second),
        edges=(
            EdgeDefinition(
                id=EdgeId("child_enter_first"),
                source=child_enter.id,
                target=first.id,
                visit=VisitDefinition(implementation=scenario.child_first),
            ),
            EdgeDefinition(
                id=EdgeId("child_first_second"),
                source=first.id,
                target=second.id,
                visit=VisitDefinition(implementation=scenario.child_second),
            ),
            EdgeDefinition(
                id=EdgeId("child_second_exit"),
                source=second.id,
                target=child_exit.id,
            ),
        ),
    )
    child_module_name = f"runtime_test_nested_child_{module_suffix}"
    child_module = ModuleType(child_module_name)
    child_module.definition = lambda: SubroutineDefinition(graph=child)  # type: ignore[attr-defined]
    sys.modules[child_module_name] = child_module

    parent_enter = PortDefinition(id=NodeId("parent_enter"))
    parent_exit = PortDefinition(id=NodeId("parent_exit"))
    parent_failure = PortDefinition(id=NodeId("parent_failure"))
    call_node: NodeDefinition[Count, object] = NodeDefinition(
        id=NodeId("call_child"),
        name="call child",
        operation=SubroutineCall(
            definition_id=child.id,
            definition_module=child_module_name,
            params_types={(".", child.id): type(None)},
            profile_arguments={},
            session_arguments={},
        ),
        state_type=Count,
    )
    parent: GraphDefinition[int, int, None, object] = GraphDefinition(
        id=GraphId(f"nested_{module_suffix}"),
        params_type=type(None),
        enter=parent_enter,
        exit=parent_exit,
        failure=parent_failure,
        nodes=(call_node,),
        edges=(
            EdgeDefinition(
                id=EdgeId("parent_enter_call"),
                source=parent_enter.id,
                target=call_node.id,
                visit=CallVisitDefinition(
                    implementation=(
                        cast(Any, scenario.async_call)
                        if adapter_mode == "async"
                        else scenario.call
                    )
                ),
            ),
            EdgeDefinition(
                id=EdgeId("parent_call_exit"),
                source=call_node.id,
                target=parent_exit.id,
            ),
        ),
    )
    parent_module_name = f"runtime_test_nested_parent_{module_suffix}"
    parent_module = ModuleType(parent_module_name)
    parent_module.definition = lambda: SubroutineDefinition(graph=parent)  # type: ignore[attr-defined]
    sys.modules[parent_module_name] = parent_module
    return (
        WorkflowDefinition(
            id=GraphId(f"nested_workflow_{module_suffix}"),
            input_type=int,
            entry=SubroutineCall(
                definition_id=parent.id,
                definition_module=parent_module_name,
                params_types={
                    (".", parent.id): type(None),
                    (".", child.id): type(None),
                },
                profile_arguments={},
                session_arguments={},
            ),
            configuration=WorkflowConfiguration(),
        ),
        scenario.calls,
        scenario.failures,
    )


def _path_call_workflow(
    module_suffix: str,
    source: Path,
) -> tuple[
    WorkflowDefinition[ForkInput, Path, None, object],
    GraphId,
    list[ForkInput],
    dict[str, bool],
]:
    child_calls: list[ForkInput] = []
    control = {"interrupt": True}

    def child_impl(
        input: ForkInput,
        state: Count,
        _context: object,
        /,
    ) -> Success[Path, Count]:
        child_calls.append(input)
        return Success(
            output=input.artifact.parent / "child-output.txt",
            state=Count(state.value + 1),
        )

    child_id = GraphId(f"path_{module_suffix}__child")
    child_enter = PortDefinition(id=NodeId("child_enter"))
    child_exit = PortDefinition(id=NodeId("child_exit"))
    child_failure = PortDefinition(id=NodeId("child_failure"))
    child_node: NodeDefinition[Count, object] = NodeDefinition(
        id=NodeId("child"),
        name="child",
        operation=Python(),
        state_type=Count,
    )
    child = GraphDefinition[ForkInput, Path, ForkParams, object](
        id=child_id,
        params_type=ForkParams,
        enter=child_enter,
        exit=child_exit,
        failure=child_failure,
        nodes=(child_node,),
        edges=(
            EdgeDefinition(
                id=EdgeId("child_in"),
                source=child_enter.id,
                target=child_node.id,
                visit=VisitDefinition(implementation=child_impl),
            ),
            EdgeDefinition(
                id=EdgeId("child_out"),
                source=child_node.id,
                target=child_exit.id,
            ),
        ),
    )
    child_module_name = f"runtime_test_path_child_{module_suffix}"
    child_module = ModuleType(child_module_name)
    child_module.definition = lambda: SubroutineDefinition(graph=child)  # type: ignore[attr-defined]
    sys.modules[child_module_name] = child_module

    @dataclass(frozen=True, slots=True)
    class PathState:
        path: Path = source / "state.txt"

    parent_module_name = f"runtime_test_path_parent_{module_suffix}"
    PathState.__module__ = parent_module_name
    PathState.__qualname__ = "PathState"

    def call_impl(
        input: ForkInput,
        state: PathState,
        context: CallContext[None, ForkParams, ForkInput, Path],
        /,
    ) -> Success[Path, PathState]:
        assert state.path == input.artifact.parent / "state.txt"
        assert context.child_params == ForkParams(
            input.artifact.parent / "configured.txt"
        )
        output = context.invoke(
            input,
            params=ForkParams(
                input.artifact.parent / "override.txt",
                adapter_run_id=str(context.run_id),
            ),
        )
        if control["interrupt"]:
            raise KeyboardInterrupt("interrupt path call after child return")
        return Success(output=output, state=state)

    parent_id = GraphId(f"path_{module_suffix}")
    parent_enter = PortDefinition(id=NodeId("parent_enter"))
    parent_exit = PortDefinition(id=NodeId("parent_exit"))
    parent_failure = PortDefinition(id=NodeId("parent_failure"))
    call_node: NodeDefinition[PathState, object] = NodeDefinition(
        id=NodeId("call"),
        name="call",
        operation=SubroutineCall(
            definition_id=child.id,
            definition_module=child_module_name,
            params_types={(".", child.id): ForkParams},
            profile_arguments={},
            session_arguments={},
        ),
        state_type=PathState,
    )
    parent = GraphDefinition[ForkInput, Path, None, object](
        id=parent_id,
        params_type=type(None),
        enter=parent_enter,
        exit=parent_exit,
        failure=parent_failure,
        nodes=(call_node,),
        edges=(
            EdgeDefinition(
                id=EdgeId("parent_in"),
                source=parent_enter.id,
                target=call_node.id,
                visit=CallVisitDefinition(implementation=call_impl),
            ),
            EdgeDefinition(
                id=EdgeId("parent_out"),
                source=call_node.id,
                target=parent_exit.id,
            ),
        ),
    )
    parent_module = ModuleType(parent_module_name)
    parent_module.PathState = PathState  # type: ignore[attr-defined]
    parent_module.definition = lambda: SubroutineDefinition(graph=parent)  # type: ignore[attr-defined]
    sys.modules[parent_module_name] = parent_module
    return (
        WorkflowDefinition(
            id=GraphId(f"path_workflow_{module_suffix}"),
            input_type=ForkInput,
            entry=SubroutineCall(
                definition_id=parent.id,
                definition_module=parent_module_name,
                params_types={
                    (".", parent.id): type(None),
                    (".", child.id): ForkParams,
                },
                profile_arguments={},
                session_arguments={},
            ),
            configuration=WorkflowConfiguration(),
        ),
        child.id,
        child_calls,
        control,
    )


@pytest.mark.parametrize("layout", ["inline", "flat", "attempt"])
def test_resume_continues_inside_nested_durable_subroutine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    definition, calls, failures = _nested_workflow(
        "active",
        fail_child=True,
    )
    output = tmp_path / "nested-active"

    with monkeypatch.context() as legacy:
        if layout != "inline":
            original_push = Dispatcher._push_local_child  # pyright: ignore[reportPrivateUsage]

            def legacy_push(
                dispatcher: Dispatcher, parent: Any, call: Any, *args: Any
            ) -> None:
                assert call.child_activation_id is not None
                if layout == "flat":
                    path = Path("activations") / call.child_activation_id
                    call.child_call_path = None
                else:
                    path = Path(call.visit_path) / "attempt-000001"
                    call.child_call_path = path.as_posix()
                (parent.graph_output.root / path).mkdir(parents=True)
                original_push(dispatcher, parent, call, *args)

            legacy.setattr(
                "verdog_runtime.interpreter.execution._GraphOutput.create",
                classmethod(legacy_graph_create),
            )
            legacy.setattr(Dispatcher, "_push_local_child", legacy_push)

        with pytest.raises(KeyboardInterrupt, match="nested child interrupted"):
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=output,
                checkpointing=CheckpointPolicy.REQUIRED,
            )
    store = RunStore.open(output)
    assert store.manifest().status is RunStatus.INTERRUPTED
    latest = store.manifest().checkpoints.latest_restorable
    assert latest is not None
    snapshot = decode_continuation(store.checkpoint_shard(latest, "runtime.pkl"))
    assert [type(frame) for frame in snapshot.frames] == [
        GraphFrameSnapshot,
        CallFrameSnapshot,
        GraphFrameSnapshot,
    ]
    call_frame = snapshot.frames[1]
    assert isinstance(call_frame, CallFrameSnapshot)
    assert call_frame.phase == "child_active"
    assert calls == ["call", "child_first", "child_second"]
    if layout == "flat":
        assert call_frame.child_call_path is None
        assert call_frame.child_activation_id is not None
        child_output = output / "activations" / call_frame.child_activation_id
    else:
        child_output = (
            output / "graph-nested_active/call_child/000001/attempt-000001"
            if layout == "attempt"
            else output / "call_child/000001"
        )
        assert call_frame.child_call_path == child_output.relative_to(output).as_posix()
    child_frame = cast(GraphFrameSnapshot, snapshot.frames[-1])
    child_nodes = (
        child_output / "graph-nested_active__child" if layout != "inline" else child_output
    )
    assert output / child_frame.call_path == child_nodes

    failures["child"] = False
    result = Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)

    assert result.output == 13
    assert calls == [
        "call",
        "child_first",
        "child_second",
        "child_second",
        "call",
    ]
    assert tuple(child_output.parent.iterdir()) == (child_output,)
    second_visits = child_nodes / "child_second"
    assert sorted(path.name for path in second_visits.iterdir()) == [
        "000001",
        "000002",
    ]
    assert call_reports(output / "config.md") == [child_output / "config.md"]
    assert call_reports(output / "stats.md") == [child_output / "stats.md"]
    assert (output / "stats.md").read_text("utf-8").count("## Calls") == 1
    child_rows = table_rows(child_output / "stats.md", "Nodes")
    assert any(row[2:5] == ["child_enter", "enter", "1"] for row in child_rows)
    assert any(row[2:5] == ["child_first", "python", "1"] for row in child_rows)


def test_resume_rejects_a_child_attempt_belonging_to_another_visit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition, calls, _ = _nested_workflow("foreign_attempt", fail_child=True)
    output = tmp_path / "foreign-attempt"
    original_snapshot = Dispatcher._call_snapshot  # pyright: ignore[reportPrivateUsage]

    def wrong_visit(dispatcher: Dispatcher, call: Any, /) -> CallFrameSnapshot:
        stored = original_snapshot(dispatcher, call)
        if stored.phase != "child_active":
            return stored
        return replace(
            stored,
            child_call_path=(
                "call_child/000002"
            ),
        )

    with monkeypatch.context() as corrupt:
        corrupt.setattr(Dispatcher, "_call_snapshot", wrong_visit)
        with pytest.raises(KeyboardInterrupt, match="nested child interrupted"):
            Dispatcher(project_root=tmp_path).run(
                definition, 1, output_dir=output, checkpointing=CheckpointPolicy.REQUIRED
            )
    before_resume = list(calls)

    with pytest.raises(ValueError, match="outside its call visit"):
        Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)
    assert calls == before_resume


def test_resume_pending_local_call_supersedes_partial_child_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition, calls, _ = _nested_workflow("pending_collision")
    child_id = GraphId("nested_pending_collision__child")
    output = tmp_path / "pending-collision"
    original_checkpoint = Dispatcher._checkpoint  # pyright: ignore[reportPrivateUsage]
    armed = False

    def interrupt_before_child_entry(
        dispatcher: Dispatcher,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal armed
        kind = cast(CheckpointKind, args[2])
        completed = args[3]
        if kind is CheckpointKind.CHILD_START:
            original_checkpoint(dispatcher, *args, **kwargs)
            armed = True
            return
        if (
            armed
            and kind is CheckpointKind.ENTRY
            and completed is not None
            and completed.graph == str(child_id)
        ):
            armed = False
            raise KeyboardInterrupt("before child entry checkpoint")
        original_checkpoint(dispatcher, *args, **kwargs)

    monkeypatch.setattr(Dispatcher, "_checkpoint", interrupt_before_child_entry)

    with pytest.raises(KeyboardInterrupt, match="before child entry checkpoint"):
        Dispatcher(project_root=tmp_path).run(
            definition,
            1,
            output_dir=output,
            checkpointing=CheckpointPolicy.REQUIRED,
        )

    store = RunStore.open(output)
    latest = store.manifest().checkpoints.latest_restorable
    assert latest is not None
    assert store.checkpoints()[-1].kind is CheckpointKind.CHILD_START
    snapshot = decode_continuation(store.checkpoint_shard(latest, "runtime.pkl"))
    assert [type(frame) for frame in snapshot.frames] == [
        GraphFrameSnapshot,
        CallFrameSnapshot,
    ]
    call = cast(CallFrameSnapshot, snapshot.frames[-1])
    assert call.phase == "child_pending"
    assert call.child_activation_id is None
    call_directory = output / "call_child/000001"
    abandoned_enter = call_directory / "child_enter/000001"
    assert abandoned_enter.is_dir()
    marker = abandoned_enter / "preserved.txt"
    marker.write_text("interrupted entry", encoding="utf-8")
    assert calls == ["call"]

    result = Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)

    assert result.output == 13
    assert calls == ["call", "child_first", "child_second", "call"]
    assert sorted(path.name for path in (call_directory / "child_enter").iterdir()) == [
        "000001", "000002"
    ]
    assert (call_directory / "child_first/000001").is_dir()
    assert marker.read_text("utf-8") == "interrupted entry"
    assert not list(call_directory.glob("attempt-*"))
    assert not (output / "activations").exists()


def test_resume_pending_workflow_call_supersedes_partial_child_attempt(
    tmp_path: Path,
) -> None:
    from test_child_process import (
        _definition,  # pyright: ignore[reportPrivateUsage]
        _durable_parent,  # pyright: ignore[reportPrivateUsage]
        _resumable_child_project,  # pyright: ignore[reportPrivateUsage]
    )
    from verdog_runtime._protocol import EventFrame
    from verdog_runtime.interpreter._calls import Budget

    class InterruptBeforeRemoteEntryCheckpoint(Dispatcher):
        interrupted = False

        @override
        def _forward_child_event(self, event: EventFrame, budget: Budget) -> None:
            super()._forward_child_event(event, budget)
            if (
                not self.interrupted
                and event.kind == "node"
                and event.graph_id == GraphId("child_project.main")
                and event.entity_id == "enter"
                and event.status == "succeeded"
            ):
                self.interrupted = True
                raise KeyboardInterrupt("before remote child entry checkpoint")

    child = _resumable_child_project(tmp_path)
    child_source = child / "src/child_project/subroutines/main/__init__.py"
    source_text = child_source.read_text("utf-8")
    child_source.write_text(
        source_text.replace(
            "def first(input, state, context, /):\n",
            "def first(input, state, context, /):\n"
            "    import time\n"
            "    time.sleep(0.5)\n",
        ),
        encoding="utf-8",
    )
    definition = _definition(_durable_parent(child.name))
    output = tmp_path / "outputs" / "pending-remote-collision"

    with pytest.raises(
        KeyboardInterrupt,
        match="before remote child entry checkpoint",
    ):
        InterruptBeforeRemoteEntryCheckpoint(project_root=tmp_path).run(
            definition,
            4,
            output_dir=output,
            checkpointing=CheckpointPolicy.REQUIRED,
        )

    store = RunStore.open(output)
    latest = store.manifest().checkpoints.latest_restorable
    assert latest is not None
    assert store.checkpoints()[-1].kind is CheckpointKind.CHILD_START
    assert not any(
        name.startswith("children/") for name in store.checkpoint_shards(latest)
    )
    snapshot = decode_continuation(store.checkpoint_shard(latest, "runtime.pkl"))
    call = cast(CallFrameSnapshot, snapshot.frames[-1])
    assert call.phase == "child_pending"
    assert call.child_call_path is None
    call_directory = output / "call/000001"
    abandoned_enter = call_directory / "enter/000001"
    assert abandoned_enter.is_dir()
    marker = abandoned_enter / "preserved.txt"
    marker.write_text("interrupted entry", encoding="utf-8")

    (child / "allow-second").touch()
    result = Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)

    assert result.output == 6
    # The abandoned process may finish after its event callback is interrupted;
    # pending-call replay is deliberately at-least-once across this ambiguity.
    assert (child / "first-visits").read_text("utf-8").splitlines() == [
        "first",
        "first",
    ]
    assert sorted(path.name for path in (call_directory / "enter").iterdir()) == [
        "000001", "000002"
    ]
    assert marker.read_text("utf-8") == "interrupted entry"
    assert not list(call_directory.glob("attempt-*"))


def test_resume_after_return_interruption_does_not_reinvoke_child(
    tmp_path: Path,
) -> None:
    definition, calls, failures = _nested_workflow(
        "returned",
        fail_complete=True,
    )
    output = tmp_path / "nested-returned"

    with pytest.raises(KeyboardInterrupt, match="nested completion interrupted"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=output,
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )
    store = RunStore.open(output)
    assert store.manifest().status is RunStatus.INTERRUPTED
    latest = store.manifest().checkpoints.latest_restorable
    assert latest is not None
    snapshot = decode_continuation(store.checkpoint_shard(latest, "runtime.pkl"))
    assert [type(frame) for frame in snapshot.frames] == [
        GraphFrameSnapshot,
        CallFrameSnapshot,
    ]
    call_frame = snapshot.frames[1]
    assert isinstance(call_frame, CallFrameSnapshot)
    assert call_frame.phase == "child_returned"
    assert calls == ["call", "child_first", "child_second", "call"]

    failures["complete"] = False
    result = Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)

    assert result.output == 13
    assert calls == [
        "call",
        "child_first",
        "child_second",
        "call",
        "call",
    ]


def test_call_replay_rejects_a_different_child_request(tmp_path: Path) -> None:
    definition, calls, _ = _nested_workflow(
        "divergent",
        adapter_mode="diverge",
    )

    with pytest.raises(RuntimeError, match=r"\[call_replay_diverged\]"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=tmp_path / "divergent",
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )

    assert calls == ["call", "child_first", "child_second", "call"]


def test_call_replay_divergence_cannot_be_swallowed(tmp_path: Path) -> None:
    definition, calls, _ = _nested_workflow(
        "swallowed_divergence",
        adapter_mode="swallow_diverge",
    )

    with pytest.raises(RuntimeError, match=r"\[call_replay_diverged\]"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=tmp_path / "swallowed-divergence",
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )

    assert calls == ["call", "child_first", "child_second", "call"]


def test_call_replay_divergence_caught_as_base_exception_still_fails(
    tmp_path: Path,
) -> None:
    definition, calls, _ = _nested_workflow(
        "swallowed_base_divergence",
        adapter_mode="swallow_diverge_base",
    )

    with pytest.raises(RuntimeError, match=r"\[call_replay_diverged\]"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=tmp_path / "swallowed-base-divergence",
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )

    assert calls == [
        "call",
        "child_first",
        "child_second",
        "call",
        "swallowed_base_divergence",
    ]


def test_extra_call_replay_invocation_cannot_be_swallowed(tmp_path: Path) -> None:
    definition, calls, _ = _nested_workflow(
        "swallowed_second_invoke",
        adapter_mode="swallow_two",
    )

    with pytest.raises(RuntimeError, match=r"\[call_invocation_count\]"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=tmp_path / "swallowed-second-invoke",
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )

    assert calls == [
        "call",
        "child_first",
        "child_second",
        "call",
        "swallowed_second_invoke",
    ]


@pytest.mark.parametrize("adapter_mode", ("zero", "two"))
def test_call_visit_must_invoke_exactly_one_child(
    tmp_path: Path,
    adapter_mode: str,
) -> None:
    definition, _, _ = _nested_workflow(
        f"invocations_{adapter_mode}",
        adapter_mode=adapter_mode,
    )

    with pytest.raises(RuntimeError, match=r"\[call_invocation_count\]"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=tmp_path / f"invocations-{adapter_mode}",
                checkpointing=CheckpointPolicy.AUTO,
            )
        )


def test_async_call_visit_is_rejected_as_synchronous_api(tmp_path: Path) -> None:
    definition, _, _ = _nested_workflow("async_adapter", adapter_mode="async")

    with pytest.raises(RuntimeError, match=r"\[async_call_visit\]"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=tmp_path / "async-adapter",
                checkpointing=CheckpointPolicy.AUTO,
            )
        )


def test_deterministic_nan_child_params_do_not_diverge(tmp_path: Path) -> None:
    definition, _, _ = _nested_workflow("nan_request", adapter_mode="nan")

    result = Dispatcher(project_root=tmp_path).run(
        definition,
        1,
        output_dir=tmp_path / "nan-request",
        checkpointing=CheckpointPolicy.REQUIRED,
    )

    assert result.output == 13


def test_local_child_mutation_does_not_change_replay_journal(tmp_path: Path) -> None:
    definition, _, _ = _nested_workflow(
        "mutable_request",
        adapter_mode="mutable",
    )

    result = Dispatcher(project_root=tmp_path).run(
        definition,
        1,
        output_dir=tmp_path / "mutable-request",
        checkpointing=CheckpointPolicy.REQUIRED,
    )

    assert result.output == 13


@pytest.mark.parametrize("policy", (CheckpointPolicy.AUTO, CheckpointPolicy.REQUIRED))
def test_unserializable_call_request_obeys_checkpoint_policy(
    tmp_path: Path,
    policy: CheckpointPolicy,
) -> None:
    definition, calls, _ = _nested_workflow(
        f"unserializable_request_{policy.value}",
        adapter_mode="unserializable_request",
    )
    output = tmp_path / f"unserializable-request-{policy.value}"

    if policy is CheckpointPolicy.REQUIRED:
        with pytest.raises(RuntimeError, match=r"\[checkpoint_required\]"):
            (
                Dispatcher(project_root=tmp_path).run(
                    definition,
                    1,
                    output_dir=output,
                    checkpointing=policy,
                )
            )
        assert calls == ["call"]
        return

    result = Dispatcher(project_root=tmp_path).run(
        definition,
        1,
        output_dir=output,
        checkpointing=policy,
    )
    assert result.output == 13
    assert calls == ["call", "child_first", "child_second", "call"]
    assert any(
        checkpoint.unavailable_code == "checkpoint.serialization_failed"
        for checkpoint in RunStore.open(output).checkpoints()
    )


def test_call_visit_can_catch_recorded_child_exception(tmp_path: Path) -> None:
    definition, calls, _ = _nested_workflow(
        "caught_exception",
        child_exception=True,
        catch_child_exception=True,
    )
    output = tmp_path / "caught-exception"

    result = Dispatcher(project_root=tmp_path).run(
        definition,
        1,
        output_dir=output,
        checkpointing=CheckpointPolicy.REQUIRED,
    )

    assert result.output == 97
    assert calls == ["call", "child_first", "child_second", "call", "caught"]
    child_return = [
        checkpoint
        for checkpoint in RunStore.open(output).checkpoints()
        if checkpoint.kind is CheckpointKind.CHILD_RETURN
    ]
    assert child_return and child_return[-1].restore_available


def test_resume_replays_child_exception_without_rerunning_child(tmp_path: Path) -> None:
    definition, calls, failures = _nested_workflow(
        "replayed_exception",
        child_exception=True,
    )
    output = tmp_path / "replayed-exception"

    with pytest.raises(ValueError, match="recorded child failure"):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=output,
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )
    before = list(calls)
    failures["child_exception"] = False

    with pytest.raises(ValueError, match="recorded child failure"):
        (
            Dispatcher(project_root=tmp_path).resume(
                definition,
                output_dir=output,
            )
        )

    assert before == ["call", "child_first", "child_second", "call"]
    assert calls == [*before, "call"]


@pytest.mark.parametrize(
    ("policy", "message"),
    (
        (CheckpointPolicy.AUTO, "unserializable child failure"),
        (CheckpointPolicy.REQUIRED, "checkpoint_required"),
    ),
)
def test_unserializable_child_exception_has_honest_checkpoint_behavior(
    tmp_path: Path,
    policy: CheckpointPolicy,
    message: str,
) -> None:
    definition, _, _ = _nested_workflow(
        f"unserializable_{policy.value}",
        unserializable_child_exception=True,
    )
    output = tmp_path / f"unserializable-{policy.value}"

    with pytest.raises(RuntimeError, match=message):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                1,
                output_dir=output,
                checkpointing=policy,
            )
        )

    unavailable = [
        checkpoint
        for checkpoint in RunStore.open(output).checkpoints()
        if checkpoint.unavailable_code == "checkpoint.serialization_failed"
    ]
    assert unavailable
    assert not unavailable[-1].restore_available
