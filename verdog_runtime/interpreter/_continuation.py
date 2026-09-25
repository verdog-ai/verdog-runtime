"""Serializable execution continuations for durable workflow checkpoints.

The interpreter deliberately snapshots declaration addresses and values rather
than pickling :class:`WorkflowState`.  WorkflowState uses object identities as
its in-memory keys, and those identities necessarily change when a workflow
definition is loaded in a new process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import cloudpickle

from .._statistics import (
    EMPTY_STATISTICS,
    StatisticsSnapshot,
    validate_statistics_snapshot,
)
from ..declarations import (
    AgentAccess,
    FeatureDefinition,
    FeatureNodeDefinition,
    GraphDefinition,
    NodeDefinition,
    WorkflowState,
)
from ..declarations.ids import EdgeId, GraphId, NodeId, ParameterAddress, RunId
from ..declarations.keys import StateAddress, StateKey
from .features import validate_feature_value
from .policies import SessionPolicy
from .validation import require_immutable_state

FORMAT_VERSION = 4


@dataclass(frozen=True, slots=True, kw_only=True)
class DefinitionReference:
    kind: Literal["subroutine", "workflow"]
    id: GraphId
    module: str
    project_path: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StateSlot:
    address: StateAddress
    value: object


@dataclass(frozen=True, slots=True, kw_only=True)
class Ready:
    incoming_edge_id: EdgeId
    target_node_id: NodeId


@dataclass(frozen=True, slots=True, kw_only=True)
class WaitingForChild:
    call_frame_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildReturned:
    call_frame_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Terminal:
    outcome: Literal["success", "failure"]


GraphControl: TypeAlias = Ready | WaitingForChild | ChildReturned | Terminal


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphFrameSnapshot:
    frame_id: str
    definition: DefinitionReference
    scope_current: GraphId
    scope_root: GraphId
    scope_root_workflow_id: GraphId | None
    call_path: str
    entry_input: object
    params: object
    value: object
    state: tuple[StateSlot, ...]
    control: GraphControl
    visits: tuple[tuple[NodeId, int], ...]
    session_bindings: tuple[tuple[str, str], ...]
    statistics: StatisticsSnapshot = EMPTY_STATISTICS
    # None denotes the older graph-wrapped layout, whose reports are one level up.
    report_path: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CallFrameSnapshot:
    frame_id: str
    parent_graph_frame_id: str
    node_id: NodeId
    incoming_edge_id: EdgeId
    visit_path: str
    adapter_run_id: RunId
    operation: DefinitionReference
    input: object
    prior_state: object
    child_input: object
    child_params: object
    child_params_override: bool
    phase: Literal["child_pending", "child_active", "child_returned"]
    child_activation_id: str | None = None
    child_call_path: str | None = None
    child_graph_frame_id: str | None = None
    child_output: object | None = None
    child_error: Exception | None = None


FrameSnapshot: TypeAlias = GraphFrameSnapshot | CallFrameSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionSnapshot:
    resource_id: str
    persistent: bool
    provider: str | None
    provider_session_id: str | None
    access: str | None
    copy_on_write: bool = False
    branch_supported: bool | None = None
    tainted: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterSlot:
    address: ParameterAddress
    value: object


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationSnapshot:
    format_version: int
    run_id: RunId
    transitions_remaining: int
    frames: tuple[FrameSnapshot, ...]
    sessions: tuple[SessionSnapshot, ...]
    parameters: tuple[ParameterSlot, ...] = ()


def snapshot_workflow_state(state: WorkflowState[Any], /) -> tuple[StateSlot, ...]:
    """Return deterministic, definition-independent state slots."""

    raw = cast(
        dict[int, tuple[StateKey[Any, Any], object]],
        state._states,  # pyright: ignore[reportPrivateUsage]
    )
    slots = (
        StateSlot(address=owner.state_key, value=value) for owner, value in raw.values()
    )
    return tuple(sorted(slots, key=lambda slot: slot.address))


def restore_workflow_state(
    graph: GraphDefinition[Any, Any, Any, Any],
    slots: tuple[StateSlot, ...],
    /,
) -> WorkflowState[Any]:
    """Validate slots against a freshly loaded graph and create a new universe."""

    owners = _state_owners(graph)
    values = _state_values(slots)
    missing = owners.keys() - values.keys()
    extra = values.keys() - owners.keys()
    if missing or extra:
        raise ValueError(
            "checkpoint state addresses do not match the workflow definition: "
            f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )

    restored = [
        (owner, _validated_state_value(owner, values[address]))
        for address, owner in owners.items()
    ]
    return WorkflowState[Any]._initial(  # pyright: ignore[reportPrivateUsage]
        restored
    )


def _state_owners(
    graph: GraphDefinition[Any, Any, Any, Any], /
) -> dict[StateAddress, StateKey[Any, Any]]:
    owners: dict[StateAddress, StateKey[Any, Any]] = {}
    for node in graph.nodes:
        if isinstance(node, FeatureNodeDefinition):
            continue
        if node.state_key in owners:
            raise ValueError(f"duplicate workflow state address: {node.state_key!r}")
        owners[node.state_key] = node
    for feature in graph.features:
        if feature.state_key in owners:
            raise ValueError(f"duplicate workflow state address: {feature.state_key!r}")
        owners[feature.state_key] = feature
    return owners


def _state_values(slots: tuple[StateSlot, ...], /) -> dict[StateAddress, object]:
    values: dict[StateAddress, object] = {}
    for slot in slots:
        if slot.address in values:
            raise ValueError(f"duplicate checkpoint state address: {slot.address!r}")
        values[slot.address] = slot.value
    return values


def _validated_state_value(owner: StateKey[Any, Any], value: object, /) -> object:
    if isinstance(owner, NodeDefinition):
        if type(value) is not owner.state_type:
            raise TypeError(
                f"checkpoint state for node {owner.id} has type "
                f"{type(value).__name__}, expected {owner.state_type.__name__}"
            )
        require_immutable_state(value)
        return value
    feature = cast(FeatureDefinition[Any, Any], owner)
    if value is not None:
        validate_feature_value(feature, value)
    return value


def encode_continuation(snapshot: ContinuationSnapshot, /) -> bytes:
    _validate_continuation(snapshot)
    return cloudpickle.dumps(snapshot)


def decode_continuation(payload: bytes, /) -> ContinuationSnapshot:
    value: object = cloudpickle.loads(payload)
    if not isinstance(value, ContinuationSnapshot):
        raise ValueError("checkpoint payload is not a continuation")
    _validate_continuation(value)
    # Old slotted instances have no report_path slot value. Normalize it before
    # dataclass rebasing during a fork; their physical output paths stay intact.
    return replace(
        value,
        frames=tuple(
            replace(frame, report_path=getattr(frame, "report_path", None))
            if isinstance(frame, GraphFrameSnapshot)
            else frame
            for frame in value.frames
        ),
    )


def continuation_digest(payload: bytes, /) -> str:
    return sha256(payload).hexdigest()


def _validate_continuation(snapshot: ContinuationSnapshot, /) -> None:
    if (
        type(snapshot.format_version) is not int
        or snapshot.format_version != FORMAT_VERSION
    ):
        raise ValueError("unsupported continuation format")
    raw_run_id = cast(object, snapshot.run_id)
    if not isinstance(raw_run_id, str) or not raw_run_id:
        raise ValueError("checkpoint run id is invalid")
    if (
        type(snapshot.transitions_remaining) is not int
        or snapshot.transitions_remaining < 0
    ):
        raise ValueError("checkpoint transition budget is invalid")
    raw_frames = cast(object, snapshot.frames)
    if not isinstance(raw_frames, tuple) or not all(
        isinstance(frame, (GraphFrameSnapshot, CallFrameSnapshot))
        for frame in cast(tuple[object, ...], raw_frames)
    ):
        raise ValueError("checkpoint continuation frames are invalid")
    raw_sessions = cast(object, snapshot.sessions)
    if not isinstance(raw_sessions, tuple):
        raise ValueError("checkpoint continuation sessions are invalid")
    raw_parameters = cast(object, snapshot.parameters)
    if not isinstance(raw_parameters, tuple) or not all(
        isinstance(parameter, ParameterSlot)
        for parameter in cast(tuple[object, ...], raw_parameters)
    ):
        raise ValueError("checkpoint continuation parameters are invalid")
    addresses = [parameter.address for parameter in snapshot.parameters]
    if len(addresses) != len(set(addresses)):
        raise ValueError("checkpoint continuation parameter addresses must be unique")
    frame_ids = [frame.frame_id for frame in snapshot.frames]
    if any(not frame_id for frame_id in frame_ids) or len(frame_ids) != len(
        set(frame_ids)
    ):
        raise ValueError("checkpoint continuation frame ids must be unique")
    for frame in snapshot.frames:
        if isinstance(frame, GraphFrameSnapshot):
            validate_statistics_snapshot(frame.statistics)
            report_path: object = getattr(frame, "report_path", None)
            if report_path is not None and (
                not isinstance(report_path, str)
                or not report_path
                or Path(report_path).is_absolute()
                or ".." in Path(report_path).parts
                or Path(report_path).as_posix() != report_path
                or Path(report_path) not in {
                    Path(frame.call_path), Path(frame.call_path).parent
                }
            ):
                raise ValueError("checkpoint graph report path is invalid")
    calls = tuple(
        frame for frame in snapshot.frames if isinstance(frame, CallFrameSnapshot)
    )
    if any(
        not isinstance(cast(object, frame.adapter_run_id), str)
        or not frame.adapter_run_id
        for frame in calls
    ):
        raise ValueError("checkpoint call adapter run id is invalid")
    phases = {"child_pending", "child_active", "child_returned"}
    for frame in calls:
        raw_phase = cast(object, frame.phase)
        if not isinstance(raw_phase, str) or raw_phase not in phases:
            raise ValueError("checkpoint call phase is invalid")
        activation_id = cast(object, frame.child_activation_id)
        child_call_path = cast(object, frame.child_call_path)
        operation_kind = cast(object, frame.operation.kind)
        if operation_kind not in {"subroutine", "workflow"}:
            raise ValueError("checkpoint call operation kind is invalid")
        if frame.phase == "child_pending":
            if activation_id is not None or child_call_path is not None:
                raise ValueError("pending checkpoint call has a child attempt")
            if (
                frame.child_graph_frame_id is not None
                or frame.child_output is not None
                or frame.child_error is not None
            ):
                raise ValueError("pending checkpoint call has child results")
            continue
        if child_call_path is not None:
            child_path = (
                Path(child_call_path) if isinstance(child_call_path, str) else None
            )
            if (
                child_path is None
                or not child_call_path
                or child_path.is_absolute()
                or ".." in child_path.parts
                or child_path.as_posix() != child_call_path
            ):
                raise ValueError("started checkpoint call has an invalid child path")
        if operation_kind == "subroutine":
            if not isinstance(activation_id, str) or not activation_id:
                raise ValueError("started local checkpoint call has no activation id")
            if frame.child_graph_frame_id != frame.child_activation_id:
                raise ValueError(
                    "started local checkpoint call has no matching child graph"
                )
        else:
            if activation_id is not None or child_call_path is None:
                raise ValueError(
                    "started workflow checkpoint call has no valid child path"
                )
            if frame.child_graph_frame_id is not None:
                raise ValueError("workflow checkpoint call identifies a local graph")
        if frame.phase == "child_active":
            if frame.child_output is not None or frame.child_error is not None:
                raise ValueError("active checkpoint call has child results")
        elif frame.phase == "child_returned":
            raw_child_error = cast(object, frame.child_error)
            if raw_child_error is not None and not isinstance(
                raw_child_error, Exception
            ):
                raise ValueError("returned checkpoint call has an invalid child error")
            if frame.child_output is not None and raw_child_error is not None:
                raise ValueError("returned checkpoint call has both output and error")
    _validate_sessions(snapshot)


def _validate_session(session: SessionSnapshot, /) -> None:
    if type(session.persistent) is not bool:
        raise ValueError("checkpoint session persistence is invalid")
    if type(session.copy_on_write) is not bool or type(session.tainted) is not bool:
        raise ValueError("checkpoint session recovery state is invalid")
    if (
        session.branch_supported is not None
        and type(session.branch_supported) is not bool
    ):
        raise ValueError("checkpoint session branching capability is invalid")
    provider = session.provider
    provider_id = session.provider_session_id
    access = session.access
    anchored = provider is not None or provider_id is not None or access is not None
    if anchored and (
        not session.persistent
        or not isinstance(provider, str)
        or not provider
        or not isinstance(provider_id, str)
        or not provider_id
        or access not in {item.value for item in AgentAccess}
    ):
        raise ValueError("checkpoint provider session anchor is invalid")
    if session.copy_on_write and (
        not anchored or session.branch_supported is not True or session.tainted
    ):
        raise ValueError("checkpoint copy-on-write session is invalid")
    if session.tainted and not anchored:
        raise ValueError("unestablished checkpoint session cannot be tainted")


def _validate_session_bindings(
    snapshot: ContinuationSnapshot,
    sessions: Mapping[str, SessionSnapshot],
    /,
) -> None:
    referenced: set[str] = set()
    for frame in snapshot.frames:
        if not isinstance(frame, GraphFrameSnapshot):
            continue
        local: set[str] = set()
        for session_id, resource_id in frame.session_bindings:
            if not session_id or session_id in local or resource_id not in sessions:
                raise ValueError("checkpoint session binding is invalid")
            local.add(session_id)
            referenced.add(resource_id)
    if referenced != set(sessions):
        raise ValueError("checkpoint has unbound session resources")


def _validate_sessions(snapshot: ContinuationSnapshot, /) -> None:
    sessions: dict[str, SessionSnapshot] = {}
    raw_sessions = cast(tuple[object, ...], cast(object, snapshot.sessions))
    for raw_session in raw_sessions:
        if not isinstance(raw_session, SessionSnapshot):
            raise ValueError("checkpoint continuation sessions are invalid")
        session = raw_session
        if not session.resource_id or session.resource_id in sessions:
            raise ValueError("checkpoint session resource ids must be unique")
        _validate_session(session)
        sessions[session.resource_id] = session
    _validate_session_bindings(snapshot, sessions)


def _rebase_value(
    value: object,
    source: Path,
    target: Path,
    active: set[int],
    /,
) -> object:
    if isinstance(value, Path):
        return _rebase_path(value, source, target)
    if value is None or isinstance(value, (str, bytes, int, float, bool, type)):
        return value
    identity = id(value)
    if identity in active:
        raise ValueError("checkpoint values with reference cycles cannot be rebased")
    active.add(identity)
    try:
        return _rebase_composite(value, source, target, active)
    finally:
        active.remove(identity)


def _rebase_composite(
    value: object,
    source: Path,
    target: Path,
    active: set[int],
    /,
) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _rebase_dataclass(value, source, target, active)
    if isinstance(value, tuple):
        return _rebase_tuple(cast(tuple[object, ...], value), source, target, active)
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_rebase_value(item, source, target, active) for item in items]
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return _rebase_mapping(mapping, source, target, active)
    if isinstance(value, frozenset):
        items = cast(frozenset[object], value)
        rebased = frozenset(
            _rebase_value(item, source, target, active) for item in items
        )
        if len(rebased) != len(items):
            raise ValueError("checkpoint set items collide after path rebasing")
        return rebased
    if isinstance(value, set):
        items = cast(set[object], value)
        rebased = {_rebase_value(item, source, target, active) for item in items}
        if len(rebased) != len(items):
            raise ValueError("checkpoint set items collide after path rebasing")
        return rebased
    return value


def _rebase_path(value: Path, source: Path, target: Path, /) -> Path:
    if not value.is_absolute():
        return value
    resolved = value.resolve()
    try:
        relative = resolved.relative_to(source)
    except ValueError:
        return value
    return target / relative


def _rebase_dataclass(
    value: Any, source: Path, target: Path, active: set[int], /
) -> object:
    updates = {
        item.name: _rebase_value(
            cast(object, getattr(value, item.name)), source, target, active
        )
        for item in fields(value)
        if item.init
    }
    return cast(object, replace(value, **updates))


def _rebase_tuple(
    value: tuple[object, ...], source: Path, target: Path, active: set[int], /
) -> object:
    items = tuple(_rebase_value(item, source, target, active) for item in value)
    if hasattr(value, "_fields"):
        constructor = cast(Any, type(value))
        return cast(object, constructor(*items))
    return items


def _rebase_mapping(
    value: Mapping[object, object],
    source: Path,
    target: Path,
    active: set[int],
    /,
) -> object:
    items: dict[object, object] = {}
    for key, item in value.items():
        rebased_key = _rebase_value(key, source, target, active)
        if rebased_key in items:
            raise ValueError("checkpoint mapping keys collide after path rebasing")
        items[rebased_key] = _rebase_value(item, source, target, active)
    return MappingProxyType(items) if isinstance(value, MappingProxyType) else items


def _require_branchable_sessions(sessions: tuple[SessionSnapshot, ...], /) -> None:
    unavailable = [
        session.resource_id
        for session in sessions
        if session.persistent
        and session.provider_session_id is not None
        and (session.branch_supported is not True or session.tainted)
    ]
    if unavailable:
        raise ValueError(
            "checkpoint sessions cannot be branched independently: "
            + ", ".join(unavailable)
        )


def fork_continuation(
    snapshot: ContinuationSnapshot,
    /,
    *,
    run_id: RunId,
    source_output: Path,
    target_output: Path,
    sessions: SessionPolicy,
) -> ContinuationSnapshot:
    """Create a new-run continuation with run-owned paths and sessions rebased."""

    _validate_continuation(snapshot)
    source = source_output.resolve()
    target = target_output.resolve()
    if source == target:
        raise ValueError("a fork needs a distinct output directory")
    rebased = _rebase_value(snapshot, source, target, set())
    if not isinstance(rebased, ContinuationSnapshot):  # pragma: no cover - invariant
        raise TypeError("rebased checkpoint has the wrong type")
    policy = SessionPolicy(sessions)
    if policy is SessionPolicy.BRANCH:
        _require_branchable_sessions(rebased.sessions)
    transformed_sessions = tuple(
        replace(
            session,
            provider=None if policy is SessionPolicy.FRESH else session.provider,
            provider_session_id=(
                None if policy is SessionPolicy.FRESH else session.provider_session_id
            ),
            access=None if policy is SessionPolicy.FRESH else session.access,
            copy_on_write=(
                policy is SessionPolicy.BRANCH
                and session.persistent
                and session.provider_session_id is not None
                and session.branch_supported is True
                and not session.tainted
            ),
            branch_supported=(
                None if policy is SessionPolicy.FRESH else session.branch_supported
            ),
            tainted=False,
        )
        for session in rebased.sessions
    )
    transformed = replace(rebased, run_id=run_id, sessions=transformed_sessions)
    _validate_continuation(transformed)
    return transformed
