"""Serializable execution continuations for durable workflow checkpoints.

The interpreter deliberately snapshots declaration addresses and values rather
than pickling :class:`WorkflowState`.  WorkflowState uses object identities as
its in-memory keys, and those identities necessarily change when a workflow
definition is loaded in a new process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import types
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, cast

import cloudpickle

from verdog_runtime import _statistics, declarations
from verdog_runtime.declarations import ids, keys
from verdog_runtime.interpreter import features as feature_semantics
from verdog_runtime.interpreter import policies, validation

FORMAT_VERSION = 4


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DefinitionReference:
    kind: Literal["subroutine", "workflow"]
    id: ids.GraphId
    module: str
    project_path: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StateSlot:
    address: keys.StateAddress
    value: object


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Ready:
    incoming_edge_id: ids.EdgeId
    target_node_id: ids.NodeId


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WaitingForChild:
    call_frame_id: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ChildReturned:
    call_frame_id: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Terminal:
    outcome: Literal["success", "failure"]


GraphControl: TypeAlias = Ready | WaitingForChild | ChildReturned | Terminal


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class GraphFrameSnapshot:
    frame_id: str
    definition: DefinitionReference
    scope_current: ids.GraphId
    scope_root: ids.GraphId
    scope_root_workflow_id: ids.GraphId | None
    call_path: str
    entry_input: object
    params: object
    value: object
    state: tuple[StateSlot, ...]
    control: GraphControl
    visits: tuple[tuple[ids.NodeId, int], ...]
    session_bindings: tuple[tuple[str, str], ...]
    statistics: _statistics.StatisticsSnapshot = _statistics.EMPTY_STATISTICS
    # None denotes the older graph-wrapped layout, whose reports are one level
    # up.
    report_path: str | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CallFrameSnapshot:
    frame_id: str
    parent_graph_frame_id: str
    node_id: ids.NodeId
    incoming_edge_id: ids.EdgeId
    visit_path: str
    adapter_run_id: ids.RunId
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SessionSnapshot:
    resource_id: str
    persistent: bool
    provider: str | None
    provider_session_id: str | None
    access: str | None
    copy_on_write: bool = False
    branch_supported: bool | None = None
    tainted: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ParameterSlot:
    address: ids.ParameterAddress
    value: object


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationSnapshot:
    format_version: int
    run_id: ids.RunId
    transitions_remaining: int
    frames: tuple[FrameSnapshot, ...]
    sessions: tuple[SessionSnapshot, ...]
    parameters: tuple[ParameterSlot, ...] = ()


def snapshot_workflow_state(
    state: declarations.WorkflowState[Any], /
) -> tuple[StateSlot, ...]:
    """Return deterministic, definition-independent state slots."""
    raw = cast(
        dict[int, tuple[keys.StateKey[Any, Any], object]],
        state._states,  # pyright: ignore[reportPrivateUsage]
    )
    slots = (
        StateSlot(address=owner.state_key, value=value)
        for owner, value in raw.values()
    )
    return tuple(sorted(slots, key=lambda slot: slot.address))


def restore_workflow_state(
    graph: declarations.GraphDefinition[Any, Any, Any, Any],
    slots: tuple[StateSlot, ...],
    /,
) -> declarations.WorkflowState[Any]:
    """Validate slots against a fresh graph and create a state universe."""
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
    return declarations.WorkflowState[Any]._initial(  # pyright: ignore[reportPrivateUsage]
        restored
    )


def _state_owners(
    graph: declarations.GraphDefinition[Any, Any, Any, Any], /
) -> dict[keys.StateAddress, keys.StateKey[Any, Any]]:
    owners: dict[keys.StateAddress, keys.StateKey[Any, Any]] = {}
    for node in graph.nodes:
        if isinstance(node, declarations.FeatureNodeDefinition):
            continue
        if node.state_key in owners:
            raise ValueError(
                f"duplicate workflow state address: {node.state_key!r}"
            )
        owners[node.state_key] = node
    for feature in graph.features:
        if feature.state_key in owners:
            raise ValueError(
                f"duplicate workflow state address: {feature.state_key!r}"
            )
        owners[feature.state_key] = feature
    return owners


def _state_values(
    slots: tuple[StateSlot, ...], /
) -> dict[keys.StateAddress, object]:
    values: dict[keys.StateAddress, object] = {}
    for slot in slots:
        if slot.address in values:
            raise ValueError(
                f"duplicate checkpoint state address: {slot.address!r}"
            )
        values[slot.address] = slot.value
    return values


def _validated_state_value(
    owner: keys.StateKey[Any, Any], value: object, /
) -> object:
    if isinstance(owner, declarations.NodeDefinition):
        if type(value) is not owner.state_type:
            raise TypeError(
                f"checkpoint state for node {owner.id} has type "
                f"{type(value).__name__}, expected {owner.state_type.__name__}"
            )
        validation.require_immutable_state(value)
        return value
    feature = cast(declarations.FeatureDefinition[Any, Any], owner)
    if value is not None:
        feature_semantics.validate_feature_value(feature, value)
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
    return dataclasses.replace(
        value,
        frames=tuple(
            dataclasses.replace(
                frame, report_path=getattr(frame, "report_path", None)
            )
            if isinstance(frame, GraphFrameSnapshot)
            else frame
            for frame in value.frames
        ),
    )


def continuation_digest(payload: bytes, /) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise ValueError(
            "checkpoint continuation parameter addresses must be unique"
        )
    frame_ids = [frame.frame_id for frame in snapshot.frames]
    if any(not frame_id for frame_id in frame_ids) or len(frame_ids) != len(
        set(frame_ids)
    ):
        raise ValueError("checkpoint continuation frame ids must be unique")
    for frame in snapshot.frames:
        if isinstance(frame, GraphFrameSnapshot):
            _validate_graph_frame(frame)
        else:
            _validate_call_frame(frame)
    _validate_sessions(snapshot)


def _validate_graph_frame(frame: GraphFrameSnapshot, /) -> None:
    """Validate graph-local reports independently of call progress."""
    _statistics.validate_statistics_snapshot(frame.statistics)
    report_path: object = getattr(frame, "report_path", None)
    if report_path is None:
        return
    if not isinstance(report_path, str) or not report_path:
        raise ValueError("checkpoint graph report path is invalid")
    path = pathlib.Path(report_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != report_path
        or path
        not in {
            pathlib.Path(frame.call_path),
            pathlib.Path(frame.call_path).parent,
        }
    ):
        raise ValueError("checkpoint graph report path is invalid")


def _validate_call_frame(frame: CallFrameSnapshot, /) -> None:
    """Validate the child identity and results permitted in each call phase."""
    if (
        not isinstance(cast(object, frame.adapter_run_id), str)
        or not frame.adapter_run_id
    ):
        raise ValueError("checkpoint call adapter run id is invalid")
    raw_phase = cast(object, frame.phase)
    if not isinstance(raw_phase, str) or raw_phase not in {
        "child_pending",
        "child_active",
        "child_returned",
    }:
        raise ValueError("checkpoint call phase is invalid")
    operation_kind = cast(object, frame.operation.kind)
    if operation_kind not in {"subroutine", "workflow"}:
        raise ValueError("checkpoint call operation kind is invalid")
    if frame.phase == "child_pending":
        if (
            frame.child_activation_id is not None
            or frame.child_call_path is not None
        ):
            raise ValueError("pending checkpoint call has a child attempt")
        if (
            frame.child_graph_frame_id is not None
            or frame.child_output is not None
            or frame.child_error is not None
        ):
            raise ValueError("pending checkpoint call has child results")
        return
    _validate_started_child(frame)
    if frame.phase == "child_active":
        if frame.child_output is not None or frame.child_error is not None:
            raise ValueError("active checkpoint call has child results")
    else:
        raw_child_error = cast(object, frame.child_error)
        if raw_child_error is not None and not isinstance(
            raw_child_error, Exception
        ):
            raise ValueError(
                "returned checkpoint call has an invalid child error"
            )
        if frame.child_output is not None and raw_child_error is not None:
            raise ValueError(
                "returned checkpoint call has both output and error"
            )


def _validate_started_child(frame: CallFrameSnapshot, /) -> None:
    """Distinguish in-process activation IDs from external workflow paths."""
    child_call_path = cast(object, frame.child_call_path)
    if child_call_path is not None:
        if not isinstance(child_call_path, str) or not child_call_path:
            raise ValueError(
                "started checkpoint call has an invalid child path"
            )
        child_path = pathlib.Path(child_call_path)
        if (
            child_path.is_absolute()
            or ".." in child_path.parts
            or child_path.as_posix() != child_call_path
        ):
            raise ValueError(
                "started checkpoint call has an invalid child path"
            )
    activation_id = cast(object, frame.child_activation_id)
    if frame.operation.kind == "subroutine":
        if not isinstance(activation_id, str) or not activation_id:
            raise ValueError(
                "started local checkpoint call has no activation id"
            )
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
            raise ValueError(
                "workflow checkpoint call identifies a local graph"
            )


def _validate_session(session: SessionSnapshot, /) -> None:
    if type(session.persistent) is not bool:
        raise ValueError("checkpoint session persistence is invalid")
    if (
        type(session.copy_on_write) is not bool
        or type(session.tainted) is not bool
    ):
        raise ValueError("checkpoint session recovery state is invalid")
    if (
        session.branch_supported is not None
        and type(session.branch_supported) is not bool
    ):
        raise ValueError("checkpoint session branching capability is invalid")
    provider = session.provider
    provider_id = session.provider_session_id
    access = session.access
    anchored = (
        provider is not None or provider_id is not None or access is not None
    )
    if anchored and (
        not session.persistent
        or not isinstance(provider, str)
        or not provider
        or not isinstance(provider_id, str)
        or not provider_id
        or access not in {item.value for item in declarations.AgentAccess}
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
            if (
                not session_id
                or session_id in local
                or resource_id not in sessions
            ):
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
    source: pathlib.Path,
    target: pathlib.Path,
    active: set[int],
    /,
) -> object:
    if isinstance(value, pathlib.Path):
        return _rebase_path(value, source, target)
    if value is None or isinstance(value, (str, bytes, int, float, bool, type)):
        return value
    identity = id(value)
    if identity in active:
        raise ValueError(
            "checkpoint values with reference cycles cannot be rebased"
        )
    active.add(identity)
    try:
        return _rebase_composite(value, source, target, active)
    finally:
        active.remove(identity)


def _rebase_composite(
    value: object,
    source: pathlib.Path,
    target: pathlib.Path,
    active: set[int],
    /,
) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _rebase_dataclass(value, source, target, active)
    if isinstance(value, tuple):
        return _rebase_tuple(
            cast(tuple[object, ...], value), source, target, active
        )
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
        rebased = {
            _rebase_value(item, source, target, active) for item in items
        }
        if len(rebased) != len(items):
            raise ValueError("checkpoint set items collide after path rebasing")
        return rebased
    return value


def _rebase_path(
    value: pathlib.Path, source: pathlib.Path, target: pathlib.Path, /
) -> pathlib.Path:
    if not value.is_absolute():
        return value
    resolved = value.resolve()
    try:
        relative = resolved.relative_to(source)
    except ValueError:
        return value
    return target / relative


def _rebase_dataclass(
    value: Any, source: pathlib.Path, target: pathlib.Path, active: set[int], /
) -> object:
    updates = {
        item.name: _rebase_value(
            cast(object, getattr(value, item.name)), source, target, active
        )
        for item in dataclasses.fields(value)
        if item.init
    }
    return cast(object, dataclasses.replace(value, **updates))


def _rebase_tuple(
    value: tuple[object, ...],
    source: pathlib.Path,
    target: pathlib.Path,
    active: set[int],
    /,
) -> object:
    items = tuple(_rebase_value(item, source, target, active) for item in value)
    if hasattr(value, "_fields"):
        constructor = cast(Any, type(value))
        return cast(object, constructor(*items))
    return items


def _rebase_mapping(
    value: Mapping[object, object],
    source: pathlib.Path,
    target: pathlib.Path,
    active: set[int],
    /,
) -> object:
    items: dict[object, object] = {}
    for key, item in value.items():
        rebased_key = _rebase_value(key, source, target, active)
        if rebased_key in items:
            raise ValueError(
                "checkpoint mapping keys collide after path rebasing"
            )
        items[rebased_key] = _rebase_value(item, source, target, active)
    return (
        types.MappingProxyType(items)
        if isinstance(value, types.MappingProxyType)
        else items
    )


def _require_branchable_sessions(
    sessions: tuple[SessionSnapshot, ...], /
) -> None:
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
    run_id: ids.RunId,
    source_output: pathlib.Path,
    target_output: pathlib.Path,
    sessions: policies.SessionPolicy,
) -> ContinuationSnapshot:
    """Rebase paths and sessions for a new-run continuation."""
    _validate_continuation(snapshot)
    source = source_output.resolve()
    target = target_output.resolve()
    if source == target:
        raise ValueError("a fork needs a distinct output directory")
    rebased = _rebase_value(snapshot, source, target, set())
    if not isinstance(
        rebased, ContinuationSnapshot
    ):  # pragma: no cover - invariant
        raise TypeError("rebased checkpoint has the wrong type")
    policy = policies.SessionPolicy(sessions)
    if policy is policies.SessionPolicy.BRANCH:
        _require_branchable_sessions(rebased.sessions)
    transformed_sessions = tuple(
        dataclasses.replace(
            session,
            provider=None
            if policy is policies.SessionPolicy.FRESH
            else session.provider,
            provider_session_id=(
                None
                if policy is policies.SessionPolicy.FRESH
                else session.provider_session_id
            ),
            access=None
            if policy is policies.SessionPolicy.FRESH
            else session.access,
            copy_on_write=(
                policy is policies.SessionPolicy.BRANCH
                and session.persistent
                and session.provider_session_id is not None
                and session.branch_supported is True
                and not session.tainted
            ),
            branch_supported=(
                None
                if policy is policies.SessionPolicy.FRESH
                else session.branch_supported
            ),
            tainted=False,
        )
        for session in rebased.sessions
    )
    transformed = dataclasses.replace(
        rebased, run_id=run_id, sessions=transformed_sessions
    )
    _validate_continuation(transformed)
    return transformed
