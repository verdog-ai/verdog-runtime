"""Execute workflow graphs and checkpoint their explicit activation stack."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import enum
import json
import pathlib
import pickle
import traceback
import types
import urllib.parse
import uuid
from collections.abc import (
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar, cast

import cloudpickle

from verdog_runtime import (
    _artifact_references,
    _checkpoint_compatibility,
    _child_checkpoint,
    _configuration,
    _process,
    _protocol,
    declarations,
)
from verdog_runtime import _run_store as run_store
from verdog_runtime import _statistics as statistics_module
from verdog_runtime import cancellation as cancellation_module
from verdog_runtime import child as child_module
from verdog_runtime.child import (
    _OMITTED as _OMITTED_CHILD_PARAMS,  # pyright: ignore[reportPrivateUsage]
)
from verdog_runtime.declarations import graph as graph_declarations
from verdog_runtime.declarations import ids, operations
from verdog_runtime.interpreter import (
    _agents,
    _calls,
    _continuation,
    _errors,
    _invocations,
    policies,
    validation,
)
from verdog_runtime.interpreter import features as feature_semantics
from verdog_runtime.interpreter.nodes import agent, feature, python

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ParamsT = TypeVar("ParamsT")
ScopeT = TypeVar("ScopeT")
LifecycleResultT = TypeVar("LifecycleResultT")
FeatureNode: TypeAlias = declarations.FeatureNodeDefinition
Edge: TypeAlias = declarations.EdgeDefinition
_NO_PARAMETERS: Mapping[declarations.ParameterAddress, object] = (
    types.MappingProxyType({})
)
_NO_CHECKPOINT_SHARDS: Mapping[str, bytes] = types.MappingProxyType({})
_USE_REGISTERED_PARAMS = object()
_RESTART_SESSION_SHARD = "restart-sessions.json"
_RESTART_SESSION_FORMAT = 1
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


class _LifecycleExecutor(Protocol):
    def __call__(
        self, operation: Callable[[], LifecycleResultT], /
    ) -> LifecycleResultT: ...


class ExecutionStatus(enum.StrEnum):
    """Lifecycle status reported for a node or edge execution event."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclasses.dataclass(slots=True, kw_only=True)
class NodeExecution:
    """A node event carrying its state, status, and owning project address."""

    run_id: ids.RunId
    graph_id: ids.GraphId
    node_id: ids.NodeId
    status: ExecutionStatus
    state: object
    remote: bool = False
    project_path: str = "."


@dataclasses.dataclass(slots=True, kw_only=True)
class EdgeExecution:
    """An edge event carrying its workflow state and owning project address."""

    run_id: ids.RunId
    graph_id: ids.GraphId
    edge_id: ids.EdgeId
    status: ExecutionStatus
    state: declarations.WorkflowState[Any] | None
    remote: bool = False
    project_path: str = "."


ExecutionEvent: TypeAlias = NodeExecution | EdgeExecution
ExecutionHandler: TypeAlias = Callable[[ExecutionEvent], None]


@dataclasses.dataclass(frozen=True, slots=True)
class _Terminal(Generic[ScopeT]):
    output: object
    state: declarations.WorkflowState[ScopeT]


@dataclasses.dataclass(frozen=True, slots=True)
class _NodeHandler:
    kind: str
    execute: Callable[
        [
            graph_declarations.VisitImplementation,
            declarations.NodeContext[object],
        ],
        object,
    ]


@dataclasses.dataclass(slots=True, kw_only=True)
class _LiveGraphFrame:
    project_root: pathlib.Path
    project_path: str
    frame_id: str
    definition: _continuation.DefinitionReference
    graph: declarations.GraphDefinition[Any, Any, Any, Any]
    scope: _calls.CallScope
    entry_input: object
    params: object
    value: object
    state: declarations.WorkflowState[Any]
    control: (
        _continuation.Ready
        | _continuation.WaitingForChild
        | _continuation.ChildReturned
        | _continuation.Terminal
    )
    graph_output: _GraphOutput
    resources: _agents.Resources
    check_output_transport: Callable[[object], bool] | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class _LiveCallFrame:
    frame_id: str
    parent_graph_frame_id: str
    node_id: ids.NodeId
    incoming_edge_id: ids.EdgeId
    visit_path: str
    adapter_run_id: ids.RunId
    operation: _continuation.DefinitionReference
    input: object
    prior_state: object
    request: _CallRequest
    phase: Literal["child_pending", "child_active", "child_returned"]
    child_activation_id: str | None = None
    child_call_path: str | None = None
    child_graph_frame_id: str | None = None
    child_output: object | None = None
    child_error: Exception | None = None
    execution_request: _CallRequest | None = None
    target: _ResolvedLocalCall | None = None
    timing: statistics_module.TimingSpan | None = None
    running_emitted: bool = False
    restored: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _CallRequest:
    input: object
    params: object
    params_override: bool


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _ResolvedLocalCall:
    definition: declarations.SubroutineDefinition[
        object, object, object, object
    ]
    scope: _calls.CallScope
    project_root: pathlib.Path
    project_path: str
    params: object


class _ChildCallRequested(BaseException):
    def __init__(self, request: _CallRequest, /) -> None:
        super().__init__()
        self.request = request


class _CallReplayViolation(BaseException):
    """Unwinds authored code after a latched replay-control violation."""


@dataclasses.dataclass(slots=True, kw_only=True)
class _CallReplayController:
    node_id: ids.NodeId
    request: _CallRequest
    child_output: object
    child_error: Exception | None
    make_request: Callable[[object, object], _CallRequest]
    requests_match: Callable[[_CallRequest, _CallRequest], bool]
    invocations: int = 0
    violation: tuple[str, str] | None = None

    def _reject(self, code: str, message: str, /) -> object:
        if self.violation is None:
            self.violation = (code, message)
        raise _CallReplayViolation

    def invoke(self, child_input: object, selected_params: object, /) -> object:
        self.invocations += 1
        if self.invocations != 1:
            return self._reject(
                "call_invocation_count",
                "a call node must invoke its child exactly once",
            )
        actual = self.make_request(child_input, selected_params)
        if not self.requests_match(self.request, actual):
            return self._reject(
                "call_replay_diverged",
                "call visit produced a different child request during replay",
            )
        if self.child_error is not None:
            raise self.child_error
        return self.child_output

    def enforce(self) -> None:
        if self.violation is not None:
            code, message = self.violation
            _errors.fault(self.node_id, code, message)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _CheckpointEmission:
    kind: run_store.CheckpointKind
    completed: run_store.Boundary | None
    next: run_store.Boundary | None
    restore_available: bool
    branch_available: bool
    payload: bytes | None
    shards: Mapping[str, bytes]
    artifact_references: dict[str, object] | None = None
    unavailable_code: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.branch_available and not self.restore_available:
            raise ValueError(
                "a non-restorable checkpoint cannot preserve sessions"
            )
        if self.restore_available != (self.payload is not None):
            raise ValueError(
                "checkpoint payload availability does not match "
                "restore availability"
            )
        if not self.restore_available and self.shards:
            raise ValueError(
                "a non-restorable checkpoint cannot carry child shards"
            )
        if not self.restore_available and self.artifact_references is not None:
            raise ValueError(
                "a non-restorable checkpoint cannot reference artifacts"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class _ParameterRegistry:
    values: Mapping[declarations.ParameterAddress, object]

    @classmethod
    def create(
        cls,
        params_types: Mapping[
            declarations.ParameterAddress, declarations.ParameterType
        ],
        params: Mapping[declarations.ParameterAddress, object],
        /,
        *,
        base: str = ".",
    ) -> _ParameterRegistry:
        base = _process.normalize_project_path(base)
        relative_types: dict[
            declarations.ParameterAddress, declarations.ParameterType
        ] = {}
        for raw_address, params_type in params_types.items():
            address = _process.normalize_parameter_address(raw_address)
            if address in relative_types:
                raise ValueError(f"duplicate parameter address: {address!r}")
            relative_types[address] = params_type
        relative_values: dict[declarations.ParameterAddress, object] = {}
        for raw_address, value in params.items():
            address = _process.normalize_parameter_address(raw_address)
            if address in relative_values:
                raise ValueError(f"duplicate parameter value: {address!r}")
            relative_values[address] = value
        extra = relative_values.keys() - relative_types.keys()
        if extra:
            raise ValueError(f"undeclared parameter value: {sorted(extra)!r}")

        values: dict[declarations.ParameterAddress, object] = {}
        for relative_address, params_type in relative_types.items():
            project_path, graph_id = relative_address
            address = (
                _process.compose_project_path(base, project_path),
                graph_id,
            )
            if address in values:
                raise ValueError(
                    f"duplicate normalized parameter address: {address!r}"
                )
            if relative_address in relative_values:
                value = relative_values[relative_address]
            else:
                try:
                    value = cast(Callable[[], object], params_type)()
                except TypeError as error:
                    error.add_note(
                        f"Verdog parameter value is missing: "
                        f"{relative_address!r}"
                    )
                    raise
            values[address] = value
        return cls(types.MappingProxyType(values))

    def value(
        self,
        project_path: str,
        graph_id: ids.GraphId,
        /,
    ) -> object:
        address = (_process.normalize_project_path(project_path), graph_id)
        if address not in self.values:
            raise ValueError(f"parameter declaration is missing: {address!r}")
        return self.values[address]

    def snapshot(self) -> tuple[_continuation.ParameterSlot, ...]:
        return tuple(
            _continuation.ParameterSlot(address=address, value=value)
            for address, value in sorted(self.values.items())
        )

    @classmethod
    def restore(
        cls,
        params_types: Mapping[
            declarations.ParameterAddress, declarations.ParameterType
        ],
        slots: tuple[_continuation.ParameterSlot, ...],
        /,
        *,
        base: str = ".",
    ) -> _ParameterRegistry:
        expected = {
            (
                _process.compose_project_path(
                    base, _process.normalize_project_path(project_path)
                ),
                graph_id,
            )
            for project_path, graph_id in params_types
        }
        values = {slot.address: slot.value for slot in slots}
        if values.keys() != expected:
            raise ValueError(
                "checkpoint parameter addresses do not match the workflow "
                f"definition: missing={sorted(expected - values.keys())!r} "
                f"extra={sorted(values.keys() - expected)!r}"
            )
        return cls(types.MappingProxyType(values))


def _encoded_id(value: str, /) -> str:
    encoded = urllib.parse.quote(value, safe="._-")
    if not encoded:
        return "%00"
    trailing_dots = len(encoded) - len(encoded.rstrip("."))
    if trailing_dots:
        encoded = encoded[:-trailing_dots] + "%2E" * trailing_dots
    if encoded.split(".", 1)[0].casefold() in _WINDOWS_RESERVED:
        encoded = "%5F" + encoded
    if encoded == ".verdog":
        encoded = "%2Everdog"
    return encoded


def _contained(root: pathlib.Path, relative: pathlib.Path, /) -> pathlib.Path:
    if relative.is_absolute():
        raise ValueError("output path must be relative")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("output path escapes the run directory")
    return target


def _attempt_number(name: str, /) -> int | None:
    prefix = "attempt-"
    number = name.removeprefix(prefix)
    if not number.isascii() or not number.isdecimal():
        return None
    value = int(number)
    return value if value > 0 and name == f"{prefix}{value:06d}" else None


def _run_compatibility(
    project_root: pathlib.Path,
    checkpointing: run_store.CheckpointPolicy,
    supplied: Mapping[str, str] | None,
    /,
) -> Mapping[str, str]:
    if supplied is not None:
        return supplied
    if checkpointing is run_store.CheckpointPolicy.OFF:
        return {}
    return _checkpoint_compatibility.checkpoint_compatibility(project_root)


def _checkpoint_summary(
    store: run_store.RunStore, sequence: int, /
) -> run_store.CheckpointSummary:
    summary = next(
        (item for item in store.checkpoints() if item.sequence == sequence),
        None,
    )
    if summary is None:
        raise ValueError(f"checkpoint does not exist: {sequence}")
    return summary


def _reuse_source_store(
    output_dir: pathlib.Path, supplied: run_store.RunStore | None, /
) -> run_store.RunStore:
    output = output_dir.resolve()
    if supplied is None:
        return run_store.RunStore.open(output)
    if supplied.output_dir != output:
        raise ValueError(
            "source run store does not match the requested output directory"
        )
    return supplied


def _latest_branchable_checkpoint(store: run_store.RunStore, /) -> int | None:
    return next(
        (
            item.sequence
            for item in reversed(store.checkpoints())
            if item.fork_with_branch_available
        ),
        None,
    )


def _fork_runtime_payload(
    payload: bytes,
    /,
    *,
    run_id: ids.RunId,
    source_output: pathlib.Path,
    target_output: pathlib.Path,
    sessions: policies.SessionPolicy,
) -> tuple[bytes, _continuation.ContinuationSnapshot]:
    snapshot = _continuation.fork_continuation(
        _continuation.decode_continuation(payload),
        run_id=run_id,
        source_output=source_output,
        target_output=target_output,
        sessions=sessions,
    )
    return _continuation.encode_continuation(snapshot), snapshot


def _summarize_sessions(
    sessions: Iterable[
        tuple[str, _continuation.SessionSnapshot | _agents.SessionResource]
    ],
    /,
) -> run_store.SessionState:
    persistent = 0
    issues: list[run_store.SessionIssue] = []
    for address, session in sessions:
        if not session.persistent:
            continue
        persistent += 1
        unavailable = (
            session.provider_session_id is not None
            and session.branch_supported is not True
        )
        if not session.tainted and not unavailable:
            continue
        issues.append(
            run_store.SessionIssue(
                address=address,
                provider=session.provider or "unestablished",
                code=(
                    "session.tainted"
                    if session.tainted
                    else "session.branch_unsupported"
                ),
                message=(
                    "the provider returned an invalid fork"
                    if session.tainted
                    else "the provider does not support copy-on-write forks"
                ),
            )
        )
    return run_store.SessionState(
        persistent=persistent,
        model="copy-on-write" if not issues else "legacy",
        branch_available=not issues,
        issues=tuple(issues),
    )


def _snapshot_session_state(
    snapshot: _continuation.ContinuationSnapshot, /
) -> run_store.SessionState:
    return _summarize_sessions(
        (session.resource_id, session) for session in snapshot.sessions
    )


def _restart_session_payload(
    snapshot: _continuation.ContinuationSnapshot, /
) -> bytes:
    if not snapshot.frames or not isinstance(
        snapshot.frames[0], _continuation.GraphFrameSnapshot
    ):
        raise ValueError("checkpoint continuation has no root graph frame")
    bindings = snapshot.frames[0].session_bindings
    by_id = {session.resource_id: session for session in snapshot.sessions}
    referenced = sorted({resource_id for _, resource_id in bindings})
    missing = [
        resource_id for resource_id in referenced if resource_id not in by_id
    ]
    if missing:
        raise ValueError(
            f"checkpoint root session resources are missing: {missing!r}"
        )
    value = {
        "format_version": _RESTART_SESSION_FORMAT,
        "bindings": [list(binding) for binding in bindings],
        "sessions": [
            {
                "resource_id": session.resource_id,
                "persistent": session.persistent,
                "provider": session.provider,
                "provider_session_id": session.provider_session_id,
                "access": session.access,
                "copy_on_write": session.copy_on_write,
                "branch_supported": session.branch_supported,
                "tainted": session.tainted,
            }
            for session in (by_id[resource_id] for resource_id in referenced)
        ],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _restart_session_resource(
    raw: object, /, *, require_copy_on_write: bool
) -> tuple[str, _agents.SessionResource]:
    """Decode a JSON session record before resolving bindings."""
    if not isinstance(raw, dict):
        raise ValueError("restart session checkpoint has an invalid session")
    item = cast(dict[object, object], raw)
    resource_id = item.get("resource_id")
    persistent = item.get("persistent")
    provider = item.get("provider")
    provider_session_id = item.get("provider_session_id")
    access = item.get("access")
    copy_on_write = item.get("copy_on_write")
    branch_supported = item.get("branch_supported")
    tainted = item.get("tainted")
    if (
        not isinstance(resource_id, str)
        or not resource_id
        or not isinstance(persistent, bool)
        or (provider is not None and not isinstance(provider, str))
        or (
            provider_session_id is not None
            and not isinstance(provider_session_id, str)
        )
        or (access is not None and not isinstance(access, str))
        or not isinstance(copy_on_write, bool)
        or (
            branch_supported is not None
            and not isinstance(branch_supported, bool)
        )
        or not isinstance(tainted, bool)
    ):
        raise ValueError("restart session checkpoint has an invalid session")
    resource = _agents.SessionResource(
        persistent=persistent,
        provider=provider,
        provider_session_id=(
            None
            if provider_session_id is None
            else ids.ProviderSessionId(provider_session_id)
        ),
        access=None if access is None else declarations.AgentAccess(access),
        copy_on_write=copy_on_write,
        require_copy_on_write=require_copy_on_write,
        branch_supported=branch_supported,
        tainted=tainted,
    )
    return resource_id, resource


def _restart_session_seed(
    payload: bytes,
    /,
    *,
    require_copy_on_write: bool,
) -> Mapping[str, _agents.SessionResource]:
    try:
        decoded: object = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "restart session checkpoint is invalid JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError("restart session checkpoint must be an object")
    body = cast(dict[object, object], decoded)
    if body.get("format_version") != _RESTART_SESSION_FORMAT:
        raise ValueError("unsupported restart session checkpoint format")
    raw_bindings = body.get("bindings")
    raw_sessions = body.get("sessions")
    if not isinstance(raw_bindings, list) or not isinstance(raw_sessions, list):
        raise ValueError("restart session checkpoint has invalid collections")

    stored: dict[str, _agents.SessionResource] = {}
    for raw in cast(list[object], raw_sessions):
        resource_id, resource = _restart_session_resource(
            raw, require_copy_on_write=require_copy_on_write
        )
        if resource_id in stored:
            raise ValueError(
                "restart session checkpoint has an invalid session"
            )
        stored[resource_id] = resource

    seeded: dict[str, _agents.SessionResource] = {}
    for raw in cast(list[object], raw_bindings):
        if (
            not isinstance(raw, list)
            or len(cast(list[object], raw)) != 2
            or not isinstance(raw[0], str)
            or not isinstance(raw[1], str)
            or raw[0] in seeded
            or raw[1] not in stored
        ):
            raise ValueError(
                "restart session checkpoint has an invalid binding"
            )
        seeded[raw[0]] = stored[raw[1]]
    return types.MappingProxyType(seeded)


@dataclasses.dataclass(slots=True)
class _GraphOutput:
    root: pathlib.Path
    relative: pathlib.Path
    graph_id: ids.GraphId
    project_path: str
    statistics: statistics_module.RunStatistics
    report_relative: pathlib.Path
    visits: dict[ids.NodeId, int] = dataclasses.field(
        default_factory=dict[ids.NodeId, int]
    )

    @classmethod
    def create(
        cls,
        root: pathlib.Path,
        parent: pathlib.Path,
        graph_id: ids.GraphId,
        project_path: str,
        statistics: statistics_module.RunStatistics,
        /,
    ) -> _GraphOutput:
        directory = _contained(root, parent)
        if not directory.is_dir():
            raise ValueError(f"graph output is unavailable: {directory}")
        return cls(root, parent, graph_id, project_path, statistics, parent)

    @classmethod
    def restore(
        cls,
        root: pathlib.Path,
        relative: pathlib.Path,
        graph: declarations.GraphDefinition[Any, Any, Any, Any],
        project_path: str,
        statistics: statistics_module.RunStatistics,
        visits: Mapping[ids.NodeId, int],
        /,
        *,
        report_relative: pathlib.Path,
    ) -> _GraphOutput:
        directory = _contained(root, relative)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(
                f"checkpoint graph output is unavailable: {directory}"
            )
        restored = dict(visits)
        node_ids = (
            graph.enter.id,
            graph.exit.id,
            graph.failure.id,
            *(node.id for node in graph.nodes),
        )
        for node_id in node_ids:
            node_directory = directory / _encoded_id(str(node_id))
            if not node_directory.is_dir() or node_directory.is_symlink():
                continue
            maximum = 0
            for visit_directory in node_directory.iterdir():
                if (
                    not visit_directory.is_dir()
                    or not visit_directory.name.isdigit()
                ):
                    continue
                maximum = max(maximum, int(visit_directory.name))
            if maximum:
                restored[node_id] = max(restored.get(node_id, 0), maximum)
        return cls(
            root,
            relative,
            graph.id,
            project_path,
            statistics,
            report_relative,
            restored,
        )

    def visit(self, node_id: ids.NodeId, /) -> pathlib.Path:
        node_relative = self.relative / _encoded_id(str(node_id))
        directory = _contained(self.root, node_relative)
        if node_id not in self.visits and directory.is_dir():
            self.visits[node_id] = max(
                (
                    int(path.name)
                    for path in directory.iterdir()
                    if path.name.isdecimal()
                ),
                default=0,
            )
        visit = self.visits.get(node_id, 0) + 1
        while True:
            output_dir = directory / f"{visit:06d}"
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                visit += 1
            else:
                break
        self.visits[node_id] = visit
        return output_dir

    def start_node(
        self,
        node_id: ids.NodeId,
        node_type: str,
        /,
        *,
        status: statistics_module.TimingStatus = "succeeded",
    ) -> tuple[pathlib.Path, statistics_module.TimingSpan]:
        output_dir = self.visit(node_id)
        timing = self.statistics.start(
            path=output_dir.relative_to(self.root).as_posix(),
            project_path=self.project_path,
            graph_id=str(self.graph_id),
            node_id=str(node_id),
            node_type=node_type,
            status=status,
        )
        return output_dir, timing

    @contextlib.contextmanager
    def node(
        self,
        node_id: ids.NodeId,
        node_type: str,
        *,
        status: statistics_module.TimingStatus = "succeeded",
    ) -> Generator[pathlib.Path]:
        output_dir, timing = self.start_node(node_id, node_type, status=status)
        try:
            yield output_dir
        except BaseException as error:
            timing.finish(
                "cancelled"
                if isinstance(error, cancellation_module.ExecutionCancelled)
                else "failed"
            )
            raise
        else:
            timing.finish()

    def latest(self, node_id: ids.NodeId, /) -> pathlib.Path | None:
        visit = self.visits.get(node_id)
        if visit is None:
            return None
        relative = self.relative / _encoded_id(str(node_id)) / f"{visit:06d}"
        return _contained(self.root, relative)


def initial_workflow_state(
    graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
    /,
) -> declarations.WorkflowState[ScopeT]:
    """Initialize node records, leaving feature values uninitialized."""
    owners: tuple[declarations.NodeDefinition[Any, ScopeT], ...] = tuple(
        node
        for node in graph.nodes
        if not isinstance(node, declarations.FeatureNodeDefinition)
    )
    initial: list[
        tuple[
            declarations.NodeDefinition[Any, ScopeT]
            | declarations.FeatureDefinition[Any, ScopeT],
            object,
        ]
    ] = []
    for owner in owners:
        try:
            value = owner.state_type()
            if type(value) is not owner.state_type:
                raise TypeError("constructed state has the wrong type")
            validation.require_immutable_state(value)
        except Exception as error:
            error.add_note(f"Verdog state initialization: node={owner.id}")
            raise
        initial.append((owner, value))
    initial.extend((feature, None) for feature in graph.features)
    return declarations.WorkflowState[ScopeT]._initial(  # pyright: ignore[reportPrivateUsage]
        initial
    )


class Dispatcher:
    """Execute a workflow definition and its lexically visible subroutines."""

    def __init__(
        self,
        *,
        execution_handler: ExecutionHandler | None = None,
        transition_limit: int = 10_000,
        cancellation: cancellation_module.CancellationToken | None = None,
        project_root: pathlib.Path | str | None = None,
        _statistics: statistics_module.RunStatistics | None = None,
    ) -> None:
        """Create an executor with a shared cancellation and transition budget.

        Args:
            execution_handler: Optional observer called for node and edge
                events.
            transition_limit: Maximum graph transitions across the execution.
            cancellation: A caller-owned token; otherwise a fresh token is
                created.
            project_root: Owning project directory, defaulting to the current
                directory.
            _statistics: Internal statistics owner shared by nested executions.
        """
        if transition_limit <= 0:
            raise ValueError("transition_limit must be positive")
        self._root_resource_arguments = _agents.NO_RESOURCES
        self._execution_handler = execution_handler
        self._transition_limit = transition_limit
        self._cancellation = (
            cancellation or cancellation_module.CancellationToken()
        )
        self._project_root = (
            pathlib.Path.cwd()
            if project_root is None
            else pathlib.Path(project_root)
        )
        self._statistics = _statistics
        self._checkpointing = run_store.CheckpointPolicy.OFF
        self._run_store: run_store.RunStore | None = None
        self._checkpoint_handler: (
            Callable[[_CheckpointEmission], None] | None
        ) = None
        self._checkpoint_by_reference = False
        self._artifact_cache: _artifact_references.ArtifactCache = {}
        self._previous_artifacts: (
            _artifact_references.ArtifactReferences | None
        ) = None
        self._resume_checkpoint_shards: Mapping[str, bytes] = (
            types.MappingProxyType({})
        )
        self._resume_checkpoint_sequence: int | None = None
        self._retry_incomplete = False
        self._continuation_frames: list[_LiveGraphFrame | _LiveCallFrame] = []
        self._root_session_seed: (
            Mapping[str, _agents.SessionResource] | None
        ) = None
        self._parameter_registry: _ParameterRegistry | None = None
        self._activation_root = pathlib.Path()

    def _configure_invocation_journal(
        self,
        output_root: pathlib.Path,
        /,
        *,
        retry_incomplete: bool = False,
    ) -> None:
        """Attach provider recovery state to local graph resources."""
        self._retry_incomplete = retry_incomplete
        if self._checkpointing is run_store.CheckpointPolicy.OFF:
            return
        current = self._root_resource_arguments
        self._root_resource_arguments = _agents.Resources(
            current.profiles,
            current.sessions,
            _invocations.InvocationJournal(
                output_root,
                retry_incomplete=retry_incomplete,
            ),
        )

    @staticmethod
    def _exception_members(
        error: BaseException, /
    ) -> tuple[BaseException, ...]:
        pending = [error]
        members: list[BaseException] = []
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            members.append(current)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
            if isinstance(current, BaseExceptionGroup):
                group = cast(BaseExceptionGroup[BaseException], current)
                pending.extend(group.exceptions)
        return tuple(members)

    @classmethod
    def _checkpoint_exception(
        cls, error: Exception | None, /
    ) -> Exception | None:
        if error is None:
            return None
        members = cls._exception_members(error)
        tracebacks = tuple(member.__traceback__ for member in members)
        try:
            for member in members:
                member.__traceback__ = None
            copied: object = cloudpickle.loads(cloudpickle.dumps(error))
        finally:
            for member, active_traceback in zip(
                members, tracebacks, strict=True
            ):
                member.__traceback__ = active_traceback
        if not isinstance(copied, Exception):
            raise TypeError("copied child error has the wrong type")
        for member in cls._exception_members(copied):
            member.__traceback__ = None
        return copied

    def _snapshot_continuation(
        self,
        run_id: ids.RunId,
        budget: _calls.Budget,
        /,
    ) -> _continuation.ContinuationSnapshot:
        if self._parameter_registry is None:
            raise RuntimeError("checkpoint execution has no parameter registry")
        resource_ids: dict[int, str] = {}
        sessions: list[_continuation.SessionSnapshot] = []
        frames: list[
            _continuation.GraphFrameSnapshot | _continuation.CallFrameSnapshot
        ] = []
        for frame in self._continuation_frames:
            if isinstance(frame, _LiveCallFrame):
                frames.append(self._call_snapshot(frame))
                continue
            bindings: list[tuple[str, str]] = []
            for session_id, resource in frame.resources.sessions.items():
                identity = id(resource)
                resource_id = resource_ids.get(identity)
                if resource_id is None:
                    resource_id = f"session-{len(resource_ids) + 1:06d}"
                    resource_ids[identity] = resource_id
                    sessions.append(
                        _continuation.SessionSnapshot(
                            resource_id=resource_id,
                            persistent=resource.persistent,
                            provider=resource.provider,
                            provider_session_id=(
                                None
                                if resource.provider_session_id is None
                                else str(resource.provider_session_id)
                            ),
                            access=(
                                None
                                if resource.access is None
                                else resource.access.value
                            ),
                            # A committed provider anchor is immutable.  The
                            # live
                            # resource may already have consumed its previous
                            # COW
                            # flag, but every independently restorable
                            # checkpoint
                            # must fork again on its first later invocation.
                            copy_on_write=(
                                resource.persistent
                                and resource.provider_session_id is not None
                                and resource.branch_supported is True
                                and not resource.tainted
                            ),
                            branch_supported=resource.branch_supported,
                            tainted=resource.tainted,
                        )
                    )
                bindings.append((str(session_id), resource_id))
            frames.append(
                _continuation.GraphFrameSnapshot(
                    frame_id=frame.frame_id,
                    definition=frame.definition,
                    scope_current=frame.scope.current,
                    scope_root=frame.scope.root,
                    scope_root_workflow_id=frame.scope.root_workflow_id,
                    call_path=frame.graph_output.relative.as_posix(),
                    report_path=frame.graph_output.report_relative.as_posix(),
                    entry_input=frame.entry_input,
                    params=frame.params,
                    value=frame.value,
                    state=_continuation.snapshot_workflow_state(frame.state),
                    control=frame.control,
                    visits=tuple(
                        sorted(
                            frame.graph_output.visits.items(),
                            key=lambda item: item[0],
                        )
                    ),
                    session_bindings=tuple(sorted(bindings)),
                    statistics=frame.graph_output.statistics.snapshot(),
                )
            )
        return _continuation.ContinuationSnapshot(
            format_version=_continuation.FORMAT_VERSION,
            run_id=run_id,
            transitions_remaining=budget.remaining,
            frames=tuple(frames),
            sessions=tuple(sessions),
            parameters=self._parameter_registry.snapshot(),
        )

    def _session_state(self) -> run_store.SessionState:
        seen: set[int] = set()
        sessions: list[tuple[str, _agents.SessionResource]] = []
        for frame in self._continuation_frames:
            if isinstance(frame, _LiveCallFrame):
                continue
            for session_id, resource in frame.resources.sessions.items():
                identity = id(resource)
                if identity in seen:
                    continue
                seen.add(identity)
                sessions.append((f"{frame.frame_id}/{session_id}", resource))
        return _summarize_sessions(sessions)

    def _rearm_checkpoint_sessions(self) -> None:
        """Make the just-committed provider anchors copy-on-write again."""
        seen: set[int] = set()
        for frame in self._continuation_frames:
            if isinstance(frame, _LiveCallFrame):
                continue
            for resource in frame.resources.sessions.values():
                identity = id(resource)
                if identity in seen:
                    continue
                seen.add(identity)
                if (
                    resource.persistent
                    and resource.provider_session_id is not None
                    and resource.branch_supported is True
                    and not resource.tainted
                ):
                    resource.copy_on_write = True

    def _prepare_checkpoint_emission(
        self,
        run_id: ids.RunId,
        budget: _calls.Budget,
        kind: run_store.CheckpointKind,
        completed: run_store.Boundary | None,
        next_boundary: run_store.Boundary | None,
        /,
        *,
        restorable: bool,
        unavailable_code: str | None,
        unavailable_reason: str | None,
        extra_shards: Mapping[str, bytes],
        nested_branch_available: bool,
    ) -> tuple[_CheckpointEmission, run_store.SessionState]:
        payload: bytes | None = None
        checkpoint_shards = extra_shards
        if restorable:
            try:
                snapshot = self._snapshot_continuation(run_id, budget)
                payload = (
                    pickle.dumps(snapshot)
                    if self._checkpoint_by_reference
                    else _continuation.encode_continuation(snapshot)
                )
                checkpoint_shards = {
                    **extra_shards,
                    _RESTART_SESSION_SHARD: _restart_session_payload(snapshot),
                }
            except Exception as error:
                restorable = False
                payload = None
                checkpoint_shards = _NO_CHECKPOINT_SHARDS
                unavailable_code = "checkpoint.serialization_failed"
                unavailable_reason = str(error)
        session_state = self._session_state()
        branch_available = (
            session_state.branch_available and nested_branch_available
        )
        return (
            _CheckpointEmission(
                kind=kind,
                completed=completed,
                next=next_boundary,
                restore_available=restorable,
                branch_available=restorable and branch_available,
                payload=payload,
                shards=(
                    checkpoint_shards if restorable else _NO_CHECKPOINT_SHARDS
                ),
                unavailable_code=None if restorable else unavailable_code,
                unavailable_reason=None if restorable else unavailable_reason,
            ),
            session_state,
        )

    @staticmethod
    def _checkpoint_summary_from_emission(
        sequence: int,
        emission: _CheckpointEmission,
        /,
    ) -> run_store.CheckpointSummary:
        return run_store.CheckpointSummary(
            sequence=sequence,
            created_at=run_store.utc_now(),
            kind=emission.kind,
            completed=emission.completed,
            next=emission.next,
            restore_available=emission.restore_available,
            fork_with_branch_available=emission.branch_available,
            fork_with_fresh_available=emission.restore_available,
            unavailable_code=emission.unavailable_code,
            unavailable_reason=emission.unavailable_reason,
        )

    def _commit_checkpoint_emission(
        self,
        store: run_store.RunStore,
        emission: _CheckpointEmission,
        session_state: run_store.SessionState,
        /,
    ) -> _CheckpointEmission:
        sequence = store.next_checkpoint_sequence()
        summary = self._checkpoint_summary_from_emission(sequence, emission)
        shards = (
            {}
            if emission.payload is None
            else {"runtime.pkl": emission.payload, **emission.shards}
        )
        try:
            store.commit_checkpoint(
                summary,
                shards=shards,
                sessions=session_state,
                artifact_references=emission.artifact_references,
            )
        except run_store.RunStoreError as error:
            if not error.code.startswith("checkpoint.artifact_"):
                raise
            # Artifact paths are part of an exact fork. In automatic mode a bad
            # artifact makes this boundary visible but non-restorable; required
            # mode records the same diagnosis before failing below.
            emission = _CheckpointEmission(
                kind=emission.kind,
                completed=emission.completed,
                next=emission.next,
                restore_available=False,
                branch_available=False,
                payload=None,
                shards=_NO_CHECKPOINT_SHARDS,
                unavailable_code="checkpoint.artifact_capture_failed",
                unavailable_reason=str(error),
            )
            unavailable = dataclasses.replace(
                summary,
                restore_available=False,
                fork_with_branch_available=False,
                fork_with_fresh_available=False,
                unavailable_code=emission.unavailable_code,
                unavailable_reason=emission.unavailable_reason,
            )
            store.commit_checkpoint(unavailable, sessions=session_state)
        return emission

    def _require_checkpoint_policy(
        self, emission: _CheckpointEmission, /
    ) -> None:
        if self._checkpointing is not run_store.CheckpointPolicy.REQUIRED:
            return
        if not emission.restore_available:
            raise RuntimeError(
                "the completed workflow boundary could not be checkpointed: "
                f"{emission.unavailable_reason} [checkpoint_required]"
            )
        if not emission.branch_available:
            raise RuntimeError(
                "the completed workflow boundary cannot preserve persistent "
                "conversations independently [session_branch_required]"
            )

    def _checkpoint(
        self,
        run_id: ids.RunId,
        budget: _calls.Budget,
        kind: run_store.CheckpointKind,
        completed: run_store.Boundary | None,
        next_boundary: run_store.Boundary | None,
        /,
        *,
        restorable: bool = True,
        unavailable_code: str | None = None,
        unavailable_reason: str | None = None,
        extra_shards: Mapping[str, bytes] = _NO_CHECKPOINT_SHARDS,
        nested_branch_available: bool = True,
        artifact_references: dict[str, object] | None = None,
    ) -> None:
        if self._checkpointing is run_store.CheckpointPolicy.OFF:
            return
        if self._run_store is None and self._checkpoint_handler is None:
            return
        if self._run_store is not None and self._checkpoint_handler is not None:
            raise RuntimeError("checkpoint output has two owners")
        if (
            "runtime.pkl" in extra_shards
            or _RESTART_SESSION_SHARD in extra_shards
        ):
            raise ValueError(
                "checkpoint child shards cannot replace runtime metadata"
            )
        emission, session_state = self._prepare_checkpoint_emission(
            run_id,
            budget,
            kind,
            completed,
            next_boundary,
            restorable=restorable,
            unavailable_code=unavailable_code,
            unavailable_reason=unavailable_reason,
            extra_shards=extra_shards,
            nested_branch_available=nested_branch_available,
        )
        if emission.restore_available:
            try:
                if artifact_references is None:
                    if self._run_store is not None:
                        artifact_references = (
                            self._run_store.capture_artifacts()
                        )
                    else:
                        if self._statistics is None:
                            raise RuntimeError(
                                "checkpoint has no artifact root"
                            )
                        artifact_references = (
                            _artifact_references.capture_artifacts(
                                self._statistics.root,
                                cache=self._artifact_cache,
                                previous=self._previous_artifacts,
                            )
                        )
                self._previous_artifacts = (
                    _artifact_references.decode_artifact_references(
                        artifact_references, pathlib.Path("checkpoint")
                    )
                )
                emission = dataclasses.replace(
                    emission, artifact_references=artifact_references
                )
            except run_store.RunStoreError as error:
                if not error.code.startswith("checkpoint.artifact_"):
                    raise
                emission = dataclasses.replace(
                    emission,
                    restore_available=False,
                    branch_available=False,
                    payload=None,
                    shards=_NO_CHECKPOINT_SHARDS,
                    artifact_references=None,
                    unavailable_code="checkpoint.artifact_capture_failed",
                    unavailable_reason=str(error),
                )
        if self._checkpoint_handler is not None:
            self._checkpoint_handler(emission)
        else:
            store = self._run_store
            if store is None:
                raise RuntimeError("checkpoint output has no owner")
            emission = self._commit_checkpoint_emission(
                store,
                emission,
                session_state,
            )
        if emission.restore_available:
            self._rearm_checkpoint_sessions()
        self._require_checkpoint_policy(emission)

    @staticmethod
    def _boundary(
        graph_output: _GraphOutput,
        node_id: ids.NodeId,
        visit: int,
        /,
    ) -> run_store.Boundary:
        return run_store.Boundary(
            project_path=graph_output.project_path,
            graph=str(graph_output.graph_id),
            node=str(node_id),
            visit=visit,
            call_path=graph_output.relative.as_posix(),
        )

    @staticmethod
    def _remote_shard_name(call: _LiveCallFrame, /) -> str:
        return f"children/{call.frame_id}.pkl"

    def _accept_remote_checkpoint(
        self,
        call: _LiveCallFrame,
        frame: _protocol.CheckpointFrame,
        run_id: ids.RunId,
        budget: _calls.Budget,
        /,
    ) -> None:
        if call.phase != "child_active" or call.operation.kind != "workflow":
            raise ValueError(
                "a child checkpoint arrived outside its active workflow call"
            )
        if frame.run_id != run_id:
            raise ValueError("child checkpoint belongs to a different run")
        budget.remaining = frame.transitions_remaining
        child_payload = (
            None
            if frame.continuation is None
            else _protocol.decode_binary_payload(frame.continuation)
        )
        self._checkpoint(
            run_id,
            budget,
            frame.kind,
            frame.completed,
            frame.next,
            restorable=frame.restore_available,
            unavailable_code=frame.unavailable_code,
            unavailable_reason=frame.unavailable_reason,
            extra_shards=(
                _NO_CHECKPOINT_SHARDS
                if child_payload is None
                else {self._remote_shard_name(call): child_payload}
            ),
            nested_branch_available=frame.session_branch_available,
            artifact_references=frame.artifact_references,
        )

    def _remote_resume_payload(self, call: _LiveCallFrame, /) -> bytes:
        name = self._remote_shard_name(call)
        payload = self._resume_checkpoint_shards.get(name)
        if payload is not None:
            return payload
        if (
            self._run_store is not None
            and self._resume_checkpoint_sequence is not None
        ):
            return self._run_store.checkpoint_shard(
                self._resume_checkpoint_sequence,
                name,
            )
        raise ValueError("checkpoint remote child continuation is unavailable")

    @contextlib.contextmanager
    def _lifecycle_execution(
        self,
        output_root: pathlib.Path,
        store: run_store.RunStore | None,
        /,
        *,
        mark_running: bool = False,
    ) -> Generator[_LifecycleExecutor, None, None]:
        reports = _configuration.InvocationReports(output_root)
        self._statistics = statistics_module.RunStatistics(
            output_root, record_handler=reports
        )

        def execute(
            operation: Callable[[], LifecycleResultT], /
        ) -> LifecycleResultT:
            try:
                result = operation()
            except BaseException as error:
                if store is not None:
                    interrupted = isinstance(
                        error,
                        (
                            cancellation_module.ExecutionCancelled,
                            KeyboardInterrupt,
                        ),
                    )
                    store.update(
                        status=(
                            run_store.RunStatus.INTERRUPTED
                            if interrupted
                            else run_store.RunStatus.FAILED
                        )
                    )
                raise
            else:
                if store is not None:
                    store.update(status=run_store.RunStatus.SUCCEEDED)
                return result

        try:
            if mark_running:
                if store is None:
                    raise RuntimeError(
                        "a resumed execution requires a run store"
                    )
                store.update(status=run_store.RunStatus.RUNNING)
            yield execute
        finally:
            reports.finish()

    def run(
        self,
        definition: declarations.WorkflowDefinition[
            InputT, OutputT, ParamsT, ScopeT
        ],
        input: InputT,
        /,
        *,
        output_dir: pathlib.Path,
        params: Mapping[declarations.ParameterAddress, object] = _NO_PARAMETERS,
        run_id: ids.RunId | None = None,
        runtime_options: object = None,
        checkpointing: run_store.CheckpointPolicy = (
            run_store.CheckpointPolicy.OFF
        ),
        workflow_arguments: Sequence[str] = (),
        _run_store: run_store.RunStore | None = None,
        _record_run: bool = False,
        _parent: run_store.ParentRun | None = None,
        _compatibility: Mapping[str, str] | None = None,
        _session_seed: Mapping[str, _agents.SessionResource] | None = None,
    ) -> declarations.Success[OutputT, declarations.WorkflowState[ScopeT]]:
        """Execute a workflow into an empty output directory.

        Args:
            definition: Workflow entry point and declared resource
                configuration.
            input: Input value satisfying the workflow input contract.
            output_dir: New or empty directory for reports and checkpoint
                metadata.
            params: Parameter overrides indexed by project path and graph ID.
            run_id: Optional stable identity; otherwise a UUID is generated.
            runtime_options: Values recorded alongside input in configuration
                reports.
            checkpointing: Policy controlling durable continuation capture.
            workflow_arguments: Original launch arguments retained for restarts.
            _run_store: Internal run metadata owner when execution is already
                recorded.
            _record_run: Record lifecycle metadata even when checkpoints are
                disabled.
            _parent: Internal provenance for a forked or restarted run.
            _compatibility: Internal source fingerprints supplied by a parent
                launch.
            _session_seed: Restored root sessions used when restarting a run.

        Returns:
            Successful output and the final immutable workflow state.

        Raises:
            ValueError: If declarations, parameters, or the output directory are
                invalid.
            ExecutionCancelled: If the cancellation token or deadline stops
                execution.
        """
        root, scope = _calls.workflow_subroutine(
            self._project_root,
            definition,
        )
        graph = root.graph
        self._configure(definition, graph)
        output_root = output_dir.resolve()
        if output_root.exists():
            if not output_root.is_dir():
                raise ValueError("output directory must be a directory")
            if any(output_root.iterdir()):
                raise ValueError("output directory must be empty")
        else:
            output_root.mkdir(parents=True)
        (output_root / "trace.log").touch(exist_ok=False)
        identifier = run_id or ids.RunId(str(uuid.uuid4()))
        self._checkpointing = run_store.CheckpointPolicy(checkpointing)
        self._run_store = _run_store
        self._root_session_seed = _session_seed
        if (
            self._checkpointing is not run_store.CheckpointPolicy.OFF
            or _record_run
        ) and self._run_store is None:
            self._run_store = run_store.RunStore.create(
                output_root,
                project_root=self._project_root,
                workflow_id=str(definition.id),
                definition_id=str(definition.entry.definition_id),
                module=definition.entry.definition_module,
                checkpointing=self._checkpointing,
                run_id=str(identifier),
                workflow_arguments=workflow_arguments,
                parent=_parent,
                compatibility=_run_compatibility(
                    self._project_root,
                    self._checkpointing,
                    _compatibility,
                ),
            )
        self._configure_invocation_journal(output_root)
        budget = _calls.Budget(self._transition_limit)
        registry = _ParameterRegistry.create(
            definition.params_types,
            params,
        )
        launch_values = (_configuration.ConfigurationValue("input", input),)
        if runtime_options is not None:
            launch_values += (
                _configuration.ConfigurationValue("runtime", runtime_options),
            )
        lease = (
            contextlib.nullcontext()
            if self._run_store is None
            else self._run_store.lease()
        )
        try:
            with (
                self._lifecycle_execution(
                    output_root, self._run_store
                ) as execute,
                lease,
            ):
                result = execute(
                    lambda: self._run_graph(
                        graph,
                        scope,
                        input,
                        identifier,
                        budget,
                        output_root,
                        pathlib.Path(),
                        ".",
                        registry,
                        self._root_resource_arguments,
                        definition_reference=_continuation.DefinitionReference(
                            kind="subroutine",
                            id=definition.entry.definition_id,
                            module=definition.entry.definition_module,
                            project_path=definition.entry.project_path,
                        ),
                        entry_values=launch_values,
                    )
                )
        finally:
            self._root_session_seed = None
        return cast(
            declarations.Success[OutputT, declarations.WorkflowState[ScopeT]],
            result,
        )

    def _restore_session_resources(
        self,
        snapshot: _continuation.ContinuationSnapshot,
        /,
    ) -> Mapping[str, _agents.SessionResource]:
        stored: dict[str, _agents.SessionResource] = {}
        for session in snapshot.sessions:
            if session.resource_id in stored:
                raise ValueError(
                    f"duplicate checkpoint session resource: "
                    f"{session.resource_id}"
                )
            stored[session.resource_id] = _agents.SessionResource(
                persistent=session.persistent,
                provider=session.provider,
                provider_session_id=(
                    None
                    if session.provider_session_id is None
                    else ids.ProviderSessionId(session.provider_session_id)
                ),
                access=(
                    None
                    if session.access is None
                    else declarations.AgentAccess(session.access)
                ),
                copy_on_write=session.copy_on_write,
                require_copy_on_write=(
                    self._checkpointing is run_store.CheckpointPolicy.REQUIRED
                ),
                branch_supported=session.branch_supported,
                tainted=session.tainted,
            )
        return types.MappingProxyType(stored)

    @staticmethod
    def _restore_resources(
        frame: _continuation.GraphFrameSnapshot,
        resources: _agents.Resources,
        stored: Mapping[str, _agents.SessionResource],
        /,
    ) -> _agents.Resources:
        bindings = dict(frame.session_bindings)
        expected = {str(session_id) for session_id in resources.sessions}
        if set(bindings) != expected:
            raise ValueError(
                "checkpoint session bindings do not match the "
                "workflow definition"
            )
        restored_sessions: dict[
            ids.AgentSessionId, _agents.SessionResource
        ] = {}
        for raw_session_id, resource_id in bindings.items():
            resource = stored.get(resource_id)
            if resource is None:
                raise ValueError(
                    f"checkpoint session resource is missing: {resource_id}"
                )
            restored_sessions[ids.AgentSessionId(raw_session_id)] = resource
        return _agents.Resources(
            resources.profiles,
            types.MappingProxyType(restored_sessions),
            resources.invocation_journal,
            resources.invocation_epoch,
        )

    def _restore_graph_frame(
        self,
        stored_frame: _continuation.GraphFrameSnapshot,
        graph: declarations.GraphDefinition[Any, Any, Any, Any],
        scope: _calls.CallScope,
        project_root: pathlib.Path,
        project_path: str,
        resource_arguments: _agents.Resources,
        sessions: Mapping[str, _agents.SessionResource],
        output_root: pathlib.Path,
        statistics: statistics_module.RunStatistics,
        /,
    ) -> _LiveGraphFrame:
        validation.validate_graph(graph)
        expected_scope = (
            scope.current,
            scope.root,
            scope.root_workflow_id,
        )
        stored_scope = (
            stored_frame.scope_current,
            stored_frame.scope_root,
            stored_frame.scope_root_workflow_id,
        )
        if stored_scope != expected_scope:
            raise ValueError(
                "checkpoint call scope does not match the workflow"
            )
        state = _continuation.restore_workflow_state(graph, stored_frame.state)
        resources = _agents.invocation_resources(
            graph,
            stored_frame.entry_input,
            stored_frame.params,
            resource_arguments,
        )
        resources = self._restore_resources(stored_frame, resources, sessions)
        report_relative = (
            pathlib.Path(stored_frame.call_path).parent
            if stored_frame.report_path is None
            else pathlib.Path(stored_frame.report_path)
        )
        graph_statistics = statistics.scoped(
            _contained(output_root, report_relative)
        )
        graph_statistics.restore(stored_frame.statistics)
        graph_output = _GraphOutput.restore(
            output_root,
            pathlib.Path(stored_frame.call_path),
            graph,
            project_path,
            graph_statistics,
            dict(stored_frame.visits),
            report_relative=report_relative,
        )
        report_dir = _contained(output_root, report_relative)
        if not (report_dir / "config.md").exists():
            _configuration.write_configuration(
                report_dir,
                (
                    _configuration.ConfigurationValue(
                        "input", stored_frame.entry_input
                    ),
                    _configuration.ConfigurationValue(
                        "params", stored_frame.params
                    ),
                ),
            )
        return _LiveGraphFrame(
            project_root=project_root,
            project_path=project_path,
            frame_id=stored_frame.frame_id,
            definition=stored_frame.definition,
            graph=graph,
            scope=scope,
            entry_input=stored_frame.entry_input,
            params=stored_frame.params,
            value=stored_frame.value,
            state=state,
            control=stored_frame.control,
            graph_output=graph_output,
            resources=resources,
        )

    def _call_site(
        self,
        parent: _LiveGraphFrame,
        stored_call: _continuation.CallFrameSnapshot | _LiveCallFrame,
        /,
    ) -> tuple[
        declarations.NodeDefinition[Any, Any],
        Edge,
        declarations.CallVisitDefinition[Any, Any, Any, Any, Any, Any, Any],
        declarations.SubroutineCall | declarations.WorkflowCall,
    ]:
        node = self._nodes(parent.graph).get(stored_call.node_id)
        if node is None or isinstance(node, declarations.FeatureNodeDefinition):
            raise ValueError("checkpoint call node does not exist")
        operation = node.operation
        if not isinstance(
            operation, (declarations.SubroutineCall, declarations.WorkflowCall)
        ):
            raise ValueError("nested checkpoint does not describe a call node")
        edge = next(
            (
                candidate
                for candidate in parent.graph.edges
                if candidate.id == stored_call.incoming_edge_id
            ),
            None,
        )
        if edge is None or edge.target != node.id:
            raise ValueError("checkpoint call edge does not match the graph")
        visit = edge.visit
        if not isinstance(visit, declarations.CallVisitDefinition):
            raise ValueError("checkpoint call edge is not a durable call visit")
        expected_operation = _continuation.DefinitionReference(
            kind=(
                "subroutine"
                if isinstance(operation, declarations.SubroutineCall)
                else "workflow"
            ),
            id=operation.definition_id,
            module=operation.definition_module,
            project_path=operation.project_path,
        )
        if stored_call.operation != expected_operation:
            raise ValueError("checkpoint call target does not match the graph")
        expected_control: (
            _continuation.WaitingForChild | _continuation.ChildReturned
        )
        if stored_call.phase in {"child_pending", "child_active"}:
            expected_control = _continuation.WaitingForChild(
                call_frame_id=stored_call.frame_id
            )
        else:
            expected_control = _continuation.ChildReturned(
                call_frame_id=stored_call.frame_id
            )
        if parent.control != expected_control:
            raise ValueError("checkpoint graph and call controls do not match")
        latest = parent.graph_output.latest(node.id)
        if (
            latest is None
            or latest.relative_to(parent.graph_output.root).as_posix()
            != stored_call.visit_path
        ):
            raise ValueError(
                "checkpoint call output does not match the node visit"
            )
        return node, edge, visit, operation

    @staticmethod
    def _live_call_frame(
        stored: _continuation.CallFrameSnapshot, /
    ) -> _LiveCallFrame:
        return _LiveCallFrame(
            frame_id=stored.frame_id,
            parent_graph_frame_id=stored.parent_graph_frame_id,
            node_id=stored.node_id,
            incoming_edge_id=stored.incoming_edge_id,
            visit_path=stored.visit_path,
            adapter_run_id=stored.adapter_run_id,
            operation=stored.operation,
            input=stored.input,
            prior_state=stored.prior_state,
            request=_CallRequest(
                input=stored.child_input,
                params=stored.child_params,
                params_override=stored.child_params_override,
            ),
            phase=stored.phase,
            child_activation_id=stored.child_activation_id,
            child_call_path=stored.child_call_path,
            child_graph_frame_id=stored.child_graph_frame_id,
            child_output=stored.child_output,
            child_error=stored.child_error,
        )

    def _restore_stack_call(
        self,
        parent: _LiveGraphFrame,
        stored: _continuation.GraphFrameSnapshot
        | _continuation.CallFrameSnapshot,
        /,
    ) -> tuple[
        _LiveCallFrame,
        declarations.NodeDefinition[Any, Any],
        declarations.SubroutineCall | declarations.WorkflowCall,
    ]:
        if not isinstance(stored, _continuation.CallFrameSnapshot):
            raise ValueError("checkpoint continuation stack is not alternating")
        if stored.parent_graph_frame_id != parent.frame_id:
            raise ValueError("checkpoint call parent does not match its graph")
        node, _, _, operation = self._call_site(parent, stored)
        return self._live_call_frame(stored), node, operation

    @staticmethod
    def _restore_stack_child_snapshot(
        snapshot: _continuation.ContinuationSnapshot,
        position: int,
        call: _LiveCallFrame,
        operation: declarations.SubroutineCall | declarations.WorkflowCall,
        /,
    ) -> (
        tuple[_continuation.GraphFrameSnapshot, declarations.SubroutineCall]
        | None
    ):
        if call.phase == "child_pending":
            if (
                call.child_activation_id is not None
                or call.child_call_path is not None
                or call.child_graph_frame_id is not None
                or call.child_output is not None
                or call.child_error is not None
            ):
                raise ValueError("a pending checkpoint call has child results")
            if position != len(snapshot.frames):
                raise ValueError(
                    "a pending checkpoint call must terminate the live stack"
                )
            return None
        if call.phase == "child_returned":
            if position != len(snapshot.frames):
                raise ValueError(
                    "a returned checkpoint call must terminate the live stack"
                )
            return None
        if call.child_output is not None or call.child_error is not None:
            raise ValueError("an active checkpoint call has child results")
        if isinstance(operation, declarations.WorkflowCall):
            if (
                call.child_activation_id is not None
                or call.child_call_path is None
                or call.child_graph_frame_id is not None
            ):
                raise ValueError(
                    "an active workflow checkpoint has invalid "
                    "child activation state"
                )
            if position != len(snapshot.frames):
                raise ValueError(
                    "an isolated workflow checkpoint cannot embed "
                    "child graph frames"
                )
            return None
        if (
            call.child_activation_id is None
            or call.child_graph_frame_id != call.child_activation_id
        ):
            raise ValueError(
                "an active local checkpoint call has no matching child graph id"
            )
        if position >= len(snapshot.frames):
            raise ValueError("an active checkpoint call has no child graph")
        stored_child = snapshot.frames[position]
        if not isinstance(stored_child, _continuation.GraphFrameSnapshot):
            raise ValueError("an active checkpoint call has no child graph")
        if call.child_graph_frame_id != stored_child.frame_id:
            raise ValueError(
                "checkpoint call does not identify its child graph"
            )
        return stored_child, operation

    def _restore_nested_child(
        self,
        parent: _LiveGraphFrame,
        node: declarations.NodeDefinition[Any, Any],
        operation: declarations.SubroutineCall,
        call: _LiveCallFrame,
        stored_child: _continuation.GraphFrameSnapshot,
        sessions: Mapping[str, _agents.SessionResource],
        params_registry: _ParameterRegistry,
        output_root: pathlib.Path,
        statistics: statistics_module.RunStatistics,
        /,
    ) -> _LiveGraphFrame:
        target = self._resolve_local_call(
            operation, parent, node.id, params_registry
        )
        if (
            stored_child.definition.kind != "subroutine"
            or stored_child.definition.id != target.definition.graph.id
            or stored_child.definition.module != operation.definition_module
            or _process.normalize_project_path(
                stored_child.definition.project_path
            )
            != target.project_path
        ):
            raise ValueError(
                "checkpoint child definition does not match the call"
            )
        child_parent = (
            pathlib.Path(stored_child.call_path).parent
            if stored_child.report_path is None
            else pathlib.Path(stored_child.report_path)
        )
        if call.child_activation_id is None:
            raise ValueError(
                "active local checkpoint call has no activation id"
            )
        if child_parent != self._child_activation_parent(call):
            raise ValueError(
                "checkpoint child output is outside its activation"
            )
        arguments = _agents.child_resource_arguments(
            operation,
            target.definition.graph,
            parent.resources,
        )
        call.target = target
        return self._restore_graph_frame(
            stored_child,
            target.definition.graph,
            target.scope,
            target.project_root,
            target.project_path,
            arguments,
            sessions,
            output_root,
            statistics,
        )

    def _restore_nested_stack(
        self,
        definition: declarations.WorkflowDefinition[Any, Any, Any, Any],
        snapshot: _continuation.ContinuationSnapshot,
        output_root: pathlib.Path,
        statistics: statistics_module.RunStatistics,
        /,
        *,
        root_project_path: str = ".",
    ) -> _ParameterRegistry:
        if not snapshot.frames or not isinstance(
            snapshot.frames[0], _continuation.GraphFrameSnapshot
        ):
            raise ValueError("checkpoint continuation has no root graph frame")
        root, root_scope = _calls.workflow_subroutine(
            self._project_root, definition
        )
        stored_root = snapshot.frames[0]
        expected_root = _continuation.DefinitionReference(
            kind="subroutine",
            id=root.graph.id,
            module=definition.entry.definition_module,
            project_path=root_project_path,
        )
        if stored_root.definition != expected_root:
            raise ValueError(
                "checkpoint definition does not match the workflow"
            )

        registry = _ParameterRegistry.restore(
            definition.params_types,
            snapshot.parameters,
            base=root_project_path,
        )
        self._activation_root = (
            pathlib.Path(stored_root.call_path).parent
            if stored_root.report_path is None
            else pathlib.Path(stored_root.report_path)
        )
        sessions = self._restore_session_resources(snapshot)
        root_frame = self._restore_graph_frame(
            stored_root,
            root.graph,
            root_scope,
            self._project_root,
            root_project_path,
            self._root_resource_arguments,
            sessions,
            output_root,
            statistics,
        )
        live_frames: list[_LiveGraphFrame | _LiveCallFrame] = [root_frame]
        position = 1
        parent = root_frame
        while position < len(snapshot.frames):
            stored_call = snapshot.frames[position]
            call, node, operation = self._restore_stack_call(
                parent, stored_call
            )
            live_frames.append(call)
            position += 1
            child_source = self._restore_stack_child_snapshot(
                snapshot,
                position,
                call,
                operation,
            )
            if child_source is None:
                break
            stored_child, child_operation = child_source
            child = self._restore_nested_child(
                parent,
                node,
                child_operation,
                call,
                stored_child,
                sessions,
                registry,
                output_root,
                statistics,
            )
            live_frames.append(child)
            parent = child
            position += 1

        if isinstance(
            parent.control,
            (_continuation.WaitingForChild, _continuation.ChildReturned),
        ) and (
            not live_frames or not isinstance(live_frames[-1], _LiveCallFrame)
        ):
            raise ValueError("checkpoint graph is missing its live call frame")
        self._continuation_frames = live_frames
        self._parameter_registry = registry
        return registry

    def _resume_continuation(
        self,
        definition: declarations.WorkflowDefinition[Any, Any, Any, Any],
        snapshot: _continuation.ContinuationSnapshot,
        output_root: pathlib.Path,
        /,
        *,
        checkpoint_shards: Mapping[str, bytes] = _NO_CHECKPOINT_SHARDS,
        retry_incomplete: bool = False,
        budget: _calls.Budget | None = None,
        root_project_path: str = ".",
    ) -> tuple[
        declarations.Success[object, declarations.WorkflowState[Any]], int
    ]:
        """Resume a decoded stack in the child environment."""
        self._cancellation.raise_if_cancelled()
        if self._statistics is None:
            raise RuntimeError("continuation resume has no statistics owner")
        self._retry_incomplete = retry_incomplete
        self._resume_checkpoint_shards = checkpoint_shards
        registry = self._restore_nested_stack(
            definition,
            snapshot,
            output_root,
            self._statistics,
            root_project_path=root_project_path,
        )
        active_budget = (
            _calls.Budget(snapshot.transitions_remaining)
            if budget is None
            else budget
        )
        if budget is not None:
            active_budget.remaining = snapshot.transitions_remaining
        try:
            for position, frame in enumerate(self._continuation_frames):
                if not isinstance(frame, _LiveCallFrame):
                    continue
                if position == 0:
                    raise ValueError(
                        "checkpoint continuation starts with a call frame"
                    )
                parent = self._continuation_frames[position - 1]
                if not isinstance(parent, _LiveGraphFrame):
                    raise ValueError(
                        "checkpoint call has no parent graph frame"
                    )
                _, _, _, operation = self._call_site(parent, frame)
                if isinstance(operation, declarations.WorkflowCall):
                    shard_name = self._remote_shard_name(frame)
                    if frame.phase == "child_active":
                        self._remote_resume_payload(frame)
                    elif frame.phase == "child_pending":
                        has_shard = shard_name in self._resume_checkpoint_shards
                        if (
                            not has_shard
                            and self._run_store is not None
                            and self._resume_checkpoint_sequence is not None
                        ):
                            has_shard = (
                                shard_name
                                in self._run_store.checkpoint_shards(
                                    self._resume_checkpoint_sequence
                                )
                            )
                        if has_shard:
                            raise ValueError(
                                "pending workflow call has a "
                                "child continuation shard"
                            )
                frame.timing = parent.graph_output.statistics.start(
                    path=frame.visit_path,
                    project_path=parent.graph_output.project_path,
                    graph_id=str(parent.graph.id),
                    node_id=str(frame.node_id),
                    node_type=self._call_node_type(operation),
                )
                self._emit_node(
                    parent.graph.id,
                    frame.node_id,
                    snapshot.run_id,
                    ExecutionStatus.RUNNING,
                    frame.prior_state,
                    parent.graph_output.project_path,
                )
                frame.running_emitted = True
                frame.restored = True
            result = self._drive_activations(
                snapshot.run_id,
                active_budget,
                registry,
            )
            return result, active_budget.remaining
        finally:
            if self._continuation_frames:
                self._abort_activations("failed")
            self._resume_checkpoint_shards = _NO_CHECKPOINT_SHARDS

    def _configure_restored_execution(
        self,
        definition: declarations.WorkflowDefinition[
            InputT, OutputT, ParamsT, ScopeT
        ],
        output_root: pathlib.Path,
        store: run_store.RunStore,
        checkpoint: int,
        checkpointing: run_store.CheckpointPolicy,
        /,
        *,
        retry_incomplete: bool = False,
    ) -> None:
        root, _ = _calls.workflow_subroutine(self._project_root, definition)
        self._configure(definition, root.graph)
        self._checkpointing = checkpointing
        self._run_store = store
        self._resume_checkpoint_sequence = checkpoint
        self._configure_invocation_journal(
            output_root,
            retry_incomplete=retry_incomplete,
        )

    def resume(
        self,
        definition: declarations.WorkflowDefinition[
            InputT, OutputT, ParamsT, ScopeT
        ],
        /,
        *,
        output_dir: pathlib.Path,
        retry_incomplete: bool = False,
        _source_store: run_store.RunStore | None = None,
    ) -> declarations.Success[OutputT, declarations.WorkflowState[ScopeT]]:
        """Resume the same run at its latest exactly committed boundary."""
        output_root = output_dir.resolve()
        store = _reuse_source_store(output_root, _source_store)
        with store.lease():
            manifest = store.manifest()
            if manifest.workflow.id != str(definition.id):
                raise ValueError(
                    "run workflow does not match the supplied "
                    "workflow definition"
                )
            drift = _checkpoint_compatibility.compatibility_drift(
                manifest.compatibility,
                _checkpoint_compatibility.checkpoint_compatibility(
                    self._project_root
                ),
            )
            if drift is not None:
                raise ValueError(
                    "run compatibility fingerprints do not match the current "
                    f"workflow environment: {drift}"
                )
            if manifest.status is run_store.RunStatus.SUCCEEDED:
                raise ValueError("a succeeded run cannot be resumed")
            latest = manifest.checkpoints.latest_completed
            if (
                latest is None
                or latest != manifest.checkpoints.latest_restorable
                or not manifest.checkpoints.resume_available
            ):
                raise ValueError(
                    "the latest completed workflow boundary is "
                    "not exactly resumable"
                )
            summary = _checkpoint_summary(store, latest)
            if not summary.restore_available:
                raise ValueError("the latest checkpoint is not restorable")
            store.validate_artifacts(latest)
            snapshot = _continuation.decode_continuation(
                store.checkpoint_shard(latest, "runtime.pkl")
            )
            if str(snapshot.run_id) != manifest.id:
                raise ValueError(
                    "checkpoint run id does not match its run manifest"
                )
            self._configure_restored_execution(
                definition,
                output_root,
                store,
                latest,
                manifest.launch.checkpointing,
                retry_incomplete=retry_incomplete,
            )
            with self._lifecycle_execution(
                output_root, store, mark_running=True
            ) as execute:
                result, _ = execute(
                    lambda: self._resume_continuation(
                        definition,
                        snapshot,
                        output_root,
                        retry_incomplete=retry_incomplete,
                    )
                )
        return cast(
            declarations.Success[OutputT, declarations.WorkflowState[ScopeT]],
            result,
        )

    def restart(
        self,
        definition: declarations.WorkflowDefinition[
            InputT, OutputT, ParamsT, ScopeT
        ],
        input: InputT,
        /,
        *,
        source_output_dir: pathlib.Path,
        output_dir: pathlib.Path,
        sessions: policies.SessionPolicy,
        source_checkpoint: int | None = None,
        params: Mapping[declarations.ParameterAddress, object] = _NO_PARAMETERS,
        runtime_options: object = None,
        workflow_arguments: Sequence[str] = (),
        arguments_mode: Literal["reused", "overridden"] = "reused",
        _source_store: run_store.RunStore | None = None,
    ) -> declarations.Success[OutputT, declarations.WorkflowState[ScopeT]]:
        """Start a new lineage child from the source run's initial state."""
        policy = policies.SessionPolicy(sessions)
        if arguments_mode not in {"reused", "overridden"}:
            raise ValueError(
                "restart arguments mode must be reused or overridden"
            )
        source = _reuse_source_store(source_output_dir, _source_store)
        with source.lease():
            manifest = source.manifest()
            if manifest.workflow.id != str(definition.id):
                raise ValueError(
                    "source run workflow does not match the "
                    "supplied workflow definition"
                )
            checkpoint = source_checkpoint
            seed: Mapping[str, _agents.SessionResource] | None = None
            if policy is policies.SessionPolicy.BRANCH:
                if checkpoint is None:
                    checkpoint = _latest_branchable_checkpoint(source)
                if checkpoint is None:
                    raise ValueError(
                        "source run has no checkpoint from which "
                        "conversations can branch"
                    )
                summary = _checkpoint_summary(source, checkpoint)
                if not summary.fork_with_branch_available:
                    raise ValueError(
                        "persistent conversations cannot branch "
                        "from the selected checkpoint"
                    )
                seed = _restart_session_seed(
                    source.checkpoint_shard(checkpoint, _RESTART_SESSION_SHARD),
                    require_copy_on_write=(
                        manifest.launch.checkpointing
                        is run_store.CheckpointPolicy.REQUIRED
                    ),
                )
            else:
                checkpoint = None
        return self.run(
            definition,
            input,
            output_dir=output_dir,
            params=params,
            runtime_options=runtime_options,
            checkpointing=manifest.launch.checkpointing,
            workflow_arguments=workflow_arguments,
            _record_run=True,
            _parent=run_store.ParentRun(
                run_id=manifest.id,
                operation="restart",
                checkpoint=checkpoint,
                arguments=arguments_mode,
            ),
            _session_seed=seed,
        )

    def fork(
        self,
        definition: declarations.WorkflowDefinition[
            InputT, OutputT, ParamsT, ScopeT
        ],
        /,
        *,
        source_output_dir: pathlib.Path,
        checkpoint: int,
        output_dir: pathlib.Path,
        sessions: policies.SessionPolicy,
        _source_store: run_store.RunStore | None = None,
    ) -> declarations.Success[OutputT, declarations.WorkflowState[ScopeT]]:
        """Continue a selected source checkpoint as a new lineage child."""
        policy = policies.SessionPolicy(sessions)
        source_output = source_output_dir.resolve()
        target_output = output_dir.resolve()
        source = _reuse_source_store(source_output, _source_store)
        actual_compatibility = (
            _checkpoint_compatibility.checkpoint_compatibility(
                self._project_root
            )
        )
        with source.lease():
            manifest = source.manifest()
            if manifest.workflow.id != str(definition.id):
                raise ValueError(
                    "source run workflow does not match the "
                    "supplied workflow definition"
                )
            drift = _checkpoint_compatibility.compatibility_drift(
                manifest.compatibility, actual_compatibility
            )
            if drift is not None:
                raise ValueError(
                    "source compatibility fingerprints do not match "
                    "the current "
                    f"workflow environment: {drift}"
                )
            source_summary = _checkpoint_summary(source, checkpoint)
            available = (
                source_summary.fork_with_branch_available
                if policy is policies.SessionPolicy.BRANCH
                else source_summary.fork_with_fresh_available
            )
            if not source_summary.restore_available or not available:
                raise ValueError(
                    "selected checkpoint cannot be forked with "
                    f"{policy.value} conversations"
                )
            if not source.artifact_references_available(checkpoint):
                raise ValueError(
                    "selected checkpoint has no artifact references"
                )

            raw_shards = dict(source.checkpoint_shards(checkpoint))
            runtime = raw_shards.pop("runtime.pkl", None)
            raw_shards.pop(_RESTART_SESSION_SHARD, None)
            if runtime is None:
                raise ValueError(
                    "selected checkpoint has no runtime continuation"
                )
            run_id = ids.RunId(str(uuid.uuid4()))
            transformed_runtime, snapshot = _fork_runtime_payload(
                runtime,
                run_id=run_id,
                source_output=source_output,
                target_output=target_output,
                sessions=policy,
            )
            transformed_shards: dict[str, bytes] = {
                "runtime.pkl": transformed_runtime
            }
            for name, value in raw_shards.items():
                transformed_shards[name] = (
                    _child_checkpoint.mark_child_checkpoint_fork(
                        value,
                        run_id=str(run_id),
                        source_output=source_output,
                        target_output=target_output,
                        sessions=policy.value,
                    )
                    if name.startswith("children/")
                    else value
                )
            transformed_shards[_RESTART_SESSION_SHARD] = (
                _restart_session_payload(snapshot)
            )
            source.materialize_artifacts(checkpoint, target_output)
            (target_output / "trace.log").touch(exist_ok=False)

        store = run_store.RunStore.create(
            target_output,
            project_root=self._project_root,
            workflow_id=str(definition.id),
            definition_id=str(definition.entry.definition_id),
            module=definition.entry.definition_module,
            workflow_arguments=manifest.launch.workflow_arguments,
            checkpointing=manifest.launch.checkpointing,
            run_id=str(run_id),
            parent=run_store.ParentRun(
                run_id=manifest.id,
                operation="fork",
                checkpoint=checkpoint,
                arguments="checkpoint",
            ),
            compatibility=actual_compatibility,
        )
        base = run_store.CheckpointSummary(
            sequence=1,
            created_at=run_store.utc_now(),
            kind=source_summary.kind,
            completed=source_summary.completed,
            next=source_summary.next,
            restore_available=True,
            fork_with_branch_available=True,
            fork_with_fresh_available=True,
        )
        store.commit_checkpoint(
            base,
            shards=transformed_shards,
            sessions=_snapshot_session_state(snapshot),
            capture_artifacts=True,
        )

        self._configure_restored_execution(
            definition,
            target_output,
            store,
            1,
            manifest.launch.checkpointing,
        )
        with (
            self._lifecycle_execution(target_output, store) as execute,
            store.lease(),
        ):
            result, _ = execute(
                lambda: self._resume_continuation(
                    definition,
                    snapshot,
                    target_output,
                    checkpoint_shards={
                        name: value
                        for name, value in transformed_shards.items()
                        if name != "runtime.pkl"
                    },
                )
            )
        return cast(
            declarations.Success[OutputT, declarations.WorkflowState[ScopeT]],
            result,
        )

    def _configure(
        self,
        definition: declarations.WorkflowDefinition[
            InputT, OutputT, ParamsT, ScopeT
        ],
        graph: declarations.GraphDefinition[Any, Any, Any, Any],
        /,
    ) -> None:
        self._artifact_cache.clear()
        self._previous_artifacts = None
        configuration = definition.configuration
        entry = definition.entry
        target_profiles = {
            parameter.id for parameter in graph.profile_parameters
        }
        caller_profile_ids = {
            caller_id
            for parameter_id, caller_id in entry.profile_arguments.items()
            if parameter_id in target_profiles
        }
        missing = caller_profile_ids - configuration.profile_arguments.keys()
        if missing:
            raise ValueError(
                "workflow configuration is missing profile arguments: "
                + ", ".join(sorted(map(str, missing)))
            )
        profiles = {
            profile_id: _agents.invoker(
                configuration.profile_arguments[profile_id],
                f"workflow profile argument {profile_id}",
            )
            for profile_id in caller_profile_ids
        }
        sessions: dict[ids.AgentSessionId, _agents.SessionResource] = {}
        for session in definition.sessions:
            if session.id in sessions:
                raise ValueError(
                    f"duplicate workflow agent session id: {session.id}"
                )
            if not session.id:
                raise ValueError("workflow agent session id must not be empty")
            if not session.name:
                raise ValueError(
                    f"workflow agent session {session.id} name "
                    f"must not be empty"
                )
            validation.require_instance(
                session.persistent,
                bool,
                (
                    f"workflow agent session {session.id} persistent "
                    f"must be boolean"
                ),
            )
            sessions[session.id] = _agents.SessionResource(
                persistent=session.persistent
            )
        caller = _agents.Resources(
            types.MappingProxyType(profiles),
            types.MappingProxyType(sessions),
        )
        self._root_resource_arguments = _agents.child_resource_arguments(
            entry, graph, caller
        )

    def _call_snapshot(
        self,
        call: _LiveCallFrame,
        /,
    ) -> _continuation.CallFrameSnapshot:
        return _continuation.CallFrameSnapshot(
            frame_id=call.frame_id,
            parent_graph_frame_id=call.parent_graph_frame_id,
            node_id=call.node_id,
            incoming_edge_id=call.incoming_edge_id,
            visit_path=call.visit_path,
            adapter_run_id=call.adapter_run_id,
            operation=call.operation,
            input=call.input,
            prior_state=call.prior_state,
            child_input=call.request.input,
            child_params=call.request.params,
            child_params_override=call.request.params_override,
            phase=call.phase,
            child_activation_id=call.child_activation_id,
            child_call_path=call.child_call_path,
            child_graph_frame_id=call.child_graph_frame_id,
            child_output=call.child_output,
            child_error=self._checkpoint_exception(call.child_error),
        )

    def _activation_resources(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, Any],
        value: object,
        params: object,
        resource_arguments: _agents.Resources,
    ) -> _agents.Resources:
        """Restore root sessions and prepare fresh sessions for checkpoints."""
        resources = _agents.invocation_resources(
            graph, value, params, resource_arguments
        )
        is_root = not self._continuation_frames
        if is_root and self._root_session_seed is not None:
            expected = {str(session_id) for session_id in resources.sessions}
            if set(self._root_session_seed) != expected:
                raise ValueError(
                    "restart session bindings do not match "
                    "the workflow definition"
                )
            seeded: dict[ids.AgentSessionId, _agents.SessionResource] = {}
            for session_id, declared in resources.sessions.items():
                restored = self._root_session_seed[str(session_id)]
                if restored.persistent != declared.persistent:
                    raise ValueError(
                        "restart session persistence does not match the "
                        "workflow definition"
                    )
                seeded[session_id] = restored
            resources = _agents.Resources(
                resources.profiles,
                types.MappingProxyType(seeded),
                resources.invocation_journal,
                resources.invocation_epoch,
            )
        for session in resources.sessions.values():
            if not session.persistent or session.provider is not None:
                continue
            session.copy_on_write = (
                self._checkpointing is not run_store.CheckpointPolicy.OFF
            )
            session.require_copy_on_write = (
                self._checkpointing is run_store.CheckpointPolicy.REQUIRED
            )
        return resources

    def _start_graph_activation(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, Any],
        scope: _calls.CallScope,
        value: object,
        run_id: ids.RunId,
        budget: _calls.Budget,
        output_root: pathlib.Path,
        activation_parent: pathlib.Path,
        project_root: pathlib.Path,
        project_path: str,
        params_registry: _ParameterRegistry,
        resource_arguments: _agents.Resources,
        *,
        definition_reference: _continuation.DefinitionReference,
        params_override: object = _USE_REGISTERED_PARAMS,
        check_output_transport: Callable[[object], bool] | None = None,
        entry_values: tuple[_configuration.ConfigurationValue, ...] = (),
        parent_call: _LiveCallFrame | None = None,
        frame_id: str | None = None,
    ) -> _LiveGraphFrame:
        self._cancellation.raise_if_cancelled()
        if self._statistics is None:
            raise RuntimeError("graph activation has no statistics owner")
        self._parameter_registry = params_registry
        statistics = self._statistics.scoped(
            _contained(output_root, activation_parent)
        )
        graph_output: _GraphOutput | None = None
        frame: _LiveGraphFrame | None = None
        record_failure = False
        try:
            graph_output = _GraphOutput.create(
                output_root,
                activation_parent,
                graph.id,
                project_path,
                statistics,
            )
            validation.validate_graph(graph)
            registered_params = params_registry.value(project_path, graph.id)
            params = (
                registered_params
                if params_override is _USE_REGISTERED_PARAMS
                else params_override
            )
            record_failure = True
            resources = self._activation_resources(
                graph, value, params, resource_arguments
            )
            state = initial_workflow_state(graph)
            frame = _LiveGraphFrame(
                project_root=project_root,
                project_path=project_path,
                frame_id=str(uuid.uuid4()) if frame_id is None else frame_id,
                definition=definition_reference,
                graph=graph,
                scope=scope,
                entry_input=value,
                params=params,
                value=value,
                state=state,
                control=_continuation.Terminal(outcome="failure"),
                graph_output=graph_output,
                resources=resources,
                check_output_transport=check_output_transport,
            )
            parent: _LiveGraphFrame | None = None
            if parent_call is not None:
                if (
                    len(self._continuation_frames) < 2
                    or self._continuation_frames[-1] is not parent_call
                    or not isinstance(
                        self._continuation_frames[-2], _LiveGraphFrame
                    )
                ):
                    raise RuntimeError(
                        "child activation has no parent call frame"
                    )
                parent = self._continuation_frames[-2]
                parent_call.child_graph_frame_id = frame.frame_id
            self._continuation_frames.append(frame)
            if parent is not None and parent_call is not None:
                _configuration.register_invocation(
                    output_root,
                    graph_output.report_relative,
                    parent.graph_output.relative,
                    pathlib.Path(parent_call.visit_path),
                    str(graph.id),
                    parent_report=parent.graph_output.report_relative,
                )
            self._enter(
                graph.enter.id,
                run_id,
                graph_output,
                (
                    *entry_values,
                    _configuration.ConfigurationValue("params", params),
                ),
            )
            self._route(
                graph,
                graph.enter.id,
                value,
                state,
                state,
                run_id,
                budget,
                graph_output,
                project_path,
            )
            return frame
        except BaseException as error:
            try:
                if (
                    isinstance(error, Exception)
                    and record_failure
                    and graph_output is not None
                ):
                    self._record_failure(
                        graph,
                        error,
                        run_id,
                        graph_output,
                        project_path,
                    )
            finally:
                if (
                    frame is not None
                    and self._continuation_frames
                    and self._continuation_frames[-1] is frame
                ):
                    self._continuation_frames.pop()
                statistics.write()
            raise

    def _ready_site(
        self, frame: _LiveGraphFrame, /
    ) -> tuple[Edge, FeatureNode | declarations.NodeDefinition[Any, Any]]:
        control = frame.control
        if not isinstance(control, _continuation.Ready):
            raise RuntimeError("graph activation is not ready")
        edge = next(
            (
                candidate
                for candidate in frame.graph.edges
                if candidate.id == control.incoming_edge_id
            ),
            None,
        )
        if edge is None or edge.target != control.target_node_id:
            raise ValueError(
                "checkpoint continuation edge does not match the graph"
            )
        target = self._nodes(frame.graph).get(edge.target)
        if target is None:
            raise ValueError("checkpoint continuation target does not exist")
        return edge, target

    @staticmethod
    def _call_node_type(
        operation: declarations.SubroutineCall | declarations.WorkflowCall, /
    ) -> str:
        return (
            "subroutine_call"
            if isinstance(operation, declarations.SubroutineCall)
            else "workflow_call"
        )

    @staticmethod
    def _call_note(
        parent: _LiveGraphFrame,
        error: Exception,
        /,
        *,
        node_id: ids.NodeId,
        edge_id: ids.EdgeId,
        visit_path: str,
        restored: bool = False,
    ) -> None:
        prefix = (
            "Verdog resumed durable call node"
            if restored
            else "Verdog durable call node"
        )
        error.add_note(
            f"{prefix}: project={parent.project_path} "
            f"graph={parent.graph.id} node={node_id} "
            f"edge={edge_id} output={visit_path}"
        )

    def _resolve_local_call(
        self,
        operation: declarations.SubroutineCall,
        parent: _LiveGraphFrame,
        node_id: ids.NodeId,
        params_registry: _ParameterRegistry,
        /,
    ) -> _ResolvedLocalCall:
        owner_path, project_root = _calls.resolve_call_project(
            parent.project_root, operation.project_path, node_id
        )
        if owner_path == ".":
            definition, scope = _calls.local_subroutine(
                project_root, parent.scope, operation
            )
        else:
            scope, definition = _calls.subroutine_scope(project_root, operation)
        project_path = _process.compose_project_path(
            parent.project_path, owner_path
        )
        return _ResolvedLocalCall(
            definition=definition,
            scope=scope,
            project_root=project_root,
            project_path=project_path,
            params=params_registry.value(project_path, definition.graph.id),
        )

    def _begin_call_activation(
        self,
        parent: _LiveGraphFrame,
        node: declarations.NodeDefinition[Any, Any],
        edge: Edge,
        run_id: ids.RunId,
        budget: _calls.Budget,
        params_registry: _ParameterRegistry,
        /,
    ) -> None:
        visit = edge.visit
        operation = node.operation
        if not isinstance(
            visit, declarations.CallVisitDefinition
        ) or not isinstance(
            operation, (declarations.SubroutineCall, declarations.WorkflowCall)
        ):
            raise RuntimeError(
                "durable call activation has an invalid call site"
            )
        previous_state = parent.state.get(node)
        output_dir, timing = parent.graph_output.start_node(
            node.id,
            self._call_node_type(operation),
        )
        visit_path = output_dir.relative_to(parent.graph_output.root).as_posix()
        node_context = declarations.NodeContext(
            run_id=run_id,
            graph_id=parent.graph.id,
            node_id=node.id,
            edge_id=edge.id,
            output_dir=output_dir,
            params=parent.params,
        )
        try:
            self._emit_node(
                parent.graph.id,
                node.id,
                run_id,
                ExecutionStatus.RUNNING,
                previous_state,
                parent.graph_output.project_path,
            )
        except BaseException as error:
            timing.finish(
                "cancelled"
                if isinstance(error, cancellation_module.ExecutionCancelled)
                else "failed"
            )
            raise
        call: _LiveCallFrame | None = None
        try:
            target: _ResolvedLocalCall | None
            child_params: object
            if isinstance(operation, declarations.SubroutineCall):
                target = self._resolve_local_call(
                    operation,
                    parent,
                    node.id,
                    params_registry,
                )
                child_params = target.params
            else:
                target = None
                child_params = _OMITTED_CHILD_PARAMS
            execution_request = self._capture_call_request(
                visit,
                parent.value,
                previous_state,
                node_context,
                child_params,
            )
            request = self._snapshot_call_request(execution_request)
            if self._checkpointing is run_store.CheckpointPolicy.REQUIRED:
                try:
                    cloudpickle.dumps(request)
                except Exception as error:
                    raise RuntimeError(
                        "child call request is not serializable "
                        "[checkpoint_required]"
                    ) from error
            reference = _continuation.DefinitionReference(
                kind=(
                    "subroutine"
                    if isinstance(operation, declarations.SubroutineCall)
                    else "workflow"
                ),
                id=operation.definition_id,
                module=operation.definition_module,
                project_path=operation.project_path,
            )
            call = _LiveCallFrame(
                frame_id=str(uuid.uuid4()),
                parent_graph_frame_id=parent.frame_id,
                node_id=node.id,
                incoming_edge_id=edge.id,
                visit_path=visit_path,
                adapter_run_id=run_id,
                operation=reference,
                input=parent.value,
                prior_state=previous_state,
                request=request,
                phase="child_pending",
                execution_request=execution_request,
                target=target,
                timing=timing,
                running_emitted=True,
            )
            parent.control = _continuation.WaitingForChild(
                call_frame_id=call.frame_id
            )
            self._continuation_frames.append(call)
            self._checkpoint(
                run_id,
                budget,
                run_store.CheckpointKind.CHILD_START,
                None,
                self._boundary(
                    parent.graph_output,
                    node.id,
                    parent.graph_output.visits[node.id],
                ),
            )
        except BaseException as error:
            if call is not None and self._continuation_frames[-1] is call:
                raise
            try:
                if isinstance(error, Exception):
                    self._call_note(
                        parent,
                        error,
                        node_id=node.id,
                        edge_id=edge.id,
                        visit_path=visit_path,
                    )
                    self._emit_node(
                        parent.graph.id,
                        node.id,
                        run_id,
                        ExecutionStatus.FAILED,
                        previous_state,
                        parent.graph_output.project_path,
                    )
            finally:
                timing.finish(
                    "cancelled"
                    if isinstance(error, cancellation_module.ExecutionCancelled)
                    else "failed"
                )
            raise

    @staticmethod
    def _call_node_context(
        parent: _LiveGraphFrame,
        call: _LiveCallFrame,
        /,
    ) -> declarations.NodeContext[object]:
        return declarations.NodeContext(
            run_id=call.adapter_run_id,
            graph_id=parent.graph.id,
            node_id=call.node_id,
            edge_id=call.incoming_edge_id,
            output_dir=_contained(
                parent.graph_output.root, pathlib.Path(call.visit_path)
            ),
            params=parent.params,
        )

    def _accept_child_return(
        self,
        call: _LiveCallFrame,
        child_output: object,
        child_error: Exception | None,
        run_id: ids.RunId,
        budget: _calls.Budget,
        /,
    ) -> None:
        if (
            not self._continuation_frames
            or self._continuation_frames[-1] is not call
        ):
            raise RuntimeError("child return has no active call frame")
        if len(self._continuation_frames) < 2 or not isinstance(
            self._continuation_frames[-2], _LiveGraphFrame
        ):
            raise RuntimeError("child return has no parent graph frame")
        parent = self._continuation_frames[-2]
        call.child_output = child_output
        call.child_error = child_error
        call.phase = "child_returned"
        parent.control = _continuation.ChildReturned(
            call_frame_id=call.frame_id
        )
        self._checkpoint(
            run_id,
            budget,
            run_store.CheckpointKind.CHILD_RETURN,
            None,
            self._boundary(
                parent.graph_output,
                call.node_id,
                parent.graph_output.visits[call.node_id],
            ),
        )

    def _child_activation_parent(self, call: _LiveCallFrame, /) -> pathlib.Path:
        raw = call.child_call_path
        if raw is None:
            # Older local checkpoints identify their flat directory only by ID.
            if call.operation.kind == "subroutine" and call.child_activation_id:
                return (
                    self._activation_root
                    / "activations"
                    / _encoded_id(call.child_activation_id)
                )
            raise RuntimeError("child call has no child path")
        path = pathlib.Path(raw)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != raw:
            raise ValueError("child call has an invalid child path")
        visit = pathlib.Path(call.visit_path)
        if path == visit:
            return path
        if path.parent == visit and _attempt_number(path.name) is not None:
            return path
        # Older isolated workflow attempts were siblings of the call visit.
        if (
            call.operation.kind == "workflow"
            and path.parent == visit.parent
            and path.name.startswith(f"{visit.name}-child-")
            and path.name != f"{visit.name}-child-"
        ):
            return path
        raise ValueError("checkpoint child output is outside its call visit")

    def _register_workflow_activation(
        self,
        parent: _LiveGraphFrame,
        call: _LiveCallFrame,
        /,
    ) -> None:
        activation_parent = self._child_activation_parent(call)
        _configuration.register_invocation(
            parent.graph_output.root,
            activation_parent,
            parent.graph_output.relative,
            pathlib.Path(call.visit_path),
            None,
            parent_report=parent.graph_output.report_relative,
        )

    def _push_local_child(
        self,
        parent: _LiveGraphFrame,
        call: _LiveCallFrame,
        operation: declarations.SubroutineCall,
        target: _ResolvedLocalCall,
        run_id: ids.RunId,
        budget: _calls.Budget,
        params_registry: _ParameterRegistry,
        /,
    ) -> None:
        arguments = _agents.child_resource_arguments(
            operation,
            target.definition.graph,
            parent.resources,
        )
        request = call.execution_request or call.request
        activation_id = call.child_activation_id
        if activation_id is None:
            raise RuntimeError("local child call has no activation id")
        try:
            self._start_graph_activation(
                target.definition.graph,
                target.scope,
                request.input,
                run_id,
                budget,
                parent.graph_output.root,
                self._child_activation_parent(call),
                target.project_root,
                target.project_path,
                params_registry,
                arguments,
                definition_reference=_continuation.DefinitionReference(
                    kind="subroutine",
                    id=target.definition.graph.id,
                    module=operation.definition_module,
                    project_path=target.project_path,
                ),
                params_override=request.params,
                parent_call=call,
                frame_id=activation_id,
            )
        except Exception as error:
            self._accept_child_return(call, None, error, run_id, budget)

    def _finish_call_activation(
        self,
        parent: _LiveGraphFrame,
        call: _LiveCallFrame,
        node: declarations.NodeDefinition[Any, Any],
        edge: Edge,
        visit: declarations.CallVisitDefinition[
            Any, Any, Any, Any, Any, Any, Any
        ],
        operation: declarations.SubroutineCall | declarations.WorkflowCall,
        run_id: ids.RunId,
        budget: _calls.Budget,
        params_registry: _ParameterRegistry,
        /,
    ) -> None:
        if isinstance(operation, declarations.SubroutineCall):
            target = call.target or self._resolve_local_call(
                operation,
                parent,
                node.id,
                params_registry,
            )
            call.target = target
            child_params: object = target.params
        else:
            child_params = _OMITTED_CHILD_PARAMS
        context = dataclasses.replace(
            self._call_node_context(parent, call), run_id=call.adapter_run_id
        )
        success = self._replay_call_visit(
            visit,
            call.input,
            call.prior_state,
            context,
            child_params,
            call.request,
            call.child_output,
            call.child_error,
        )
        candidate = self._ordinary_successor(node, success.state, parent.state)
        self._emit_node(
            parent.graph.id,
            node.id,
            run_id,
            ExecutionStatus.SUCCEEDED,
            candidate.get(node),
            parent.graph_output.project_path,
        )
        if call.timing is not None:
            call.timing.finish()
        popped = self._continuation_frames.pop()
        if popped is not call:
            raise RuntimeError("call activation stack did not unwind")
        self._route(
            parent.graph,
            node.id,
            success.output,
            candidate,
            parent.state,
            run_id,
            budget,
            parent.graph_output,
            parent.graph_output.project_path,
        )

    def _step_call_activation(
        self,
        call: _LiveCallFrame,
        run_id: ids.RunId,
        budget: _calls.Budget,
        params_registry: _ParameterRegistry,
        /,
    ) -> None:
        if len(self._continuation_frames) < 2 or not isinstance(
            self._continuation_frames[-2], _LiveGraphFrame
        ):
            raise RuntimeError("call activation has no parent graph frame")
        parent = self._continuation_frames[-2]
        node, edge, visit, operation = self._call_site(parent, call)
        if not call.running_emitted:
            self._emit_node(
                parent.graph.id,
                node.id,
                run_id,
                ExecutionStatus.RUNNING,
                call.prior_state,
                parent.graph_output.project_path,
            )
            call.running_emitted = True
        if call.phase in {"child_pending", "child_active"}:
            pending = call.phase == "child_pending"
            if pending:
                if isinstance(operation, declarations.SubroutineCall):
                    call.child_activation_id = str(uuid.uuid4())
                call.child_call_path = call.visit_path
                call.phase = "child_active"
            if isinstance(operation, declarations.WorkflowCall):
                output: object = None
                child_error: Exception | None = None
                request = call.execution_request or call.request
                try:
                    if pending:
                        self._register_workflow_activation(parent, call)
                    resume_payload = (
                        None if pending else self._remote_resume_payload(call)
                    )
                    output = self._run_durable_workflow(
                        operation,
                        call,
                        request,
                        parent,
                        run_id,
                        budget,
                        resume_continuation=resume_payload,
                    )
                except Exception as error:
                    child_error = error
                self._accept_child_return(
                    call,
                    output,
                    child_error,
                    run_id,
                    budget,
                )
                return
            if not pending:
                raise ValueError(
                    "active local call has no child graph activation"
                )
            target = call.target or self._resolve_local_call(
                operation,
                parent,
                node.id,
                params_registry,
            )
            call.target = target
            self._push_local_child(
                parent,
                call,
                operation,
                target,
                run_id,
                budget,
                params_registry,
            )
            return
        self._finish_call_activation(
            parent,
            call,
            node,
            edge,
            visit,
            operation,
            run_id,
            budget,
            params_registry,
        )

    def _finish_graph_activation(
        self,
        frame: _LiveGraphFrame,
        run_id: ids.RunId,
        budget: _calls.Budget,
        /,
    ) -> declarations.Success[object, declarations.WorkflowState[Any]] | None:
        result = self._finish_success(
            frame.graph,
            _Terminal(frame.value, frame.state),
            run_id,
            frame.graph_output,
            frame.check_output_transport,
        )
        popped = self._continuation_frames.pop()
        if popped is not frame:
            raise RuntimeError("graph activation stack did not unwind")
        try:
            frame.graph_output.statistics.write()
        except Exception as error:
            if self._continuation_frames and isinstance(
                self._continuation_frames[-1], _LiveCallFrame
            ):
                self._accept_child_return(
                    self._continuation_frames[-1],
                    None,
                    error,
                    run_id,
                    budget,
                )
                return None
            raise
        if not self._continuation_frames:
            return result
        call = self._continuation_frames[-1]
        if not isinstance(call, _LiveCallFrame):
            raise RuntimeError("completed child graph has no call activation")
        self._accept_child_return(call, result.output, None, run_id, budget)
        return None

    def _step_graph_activation(
        self,
        frame: _LiveGraphFrame,
        run_id: ids.RunId,
        budget: _calls.Budget,
        params_registry: _ParameterRegistry,
        /,
    ) -> declarations.Success[object, declarations.WorkflowState[Any]] | None:
        if isinstance(frame.control, _continuation.Ready):
            edge, node = self._ready_site(frame)
            if (
                isinstance(node, declarations.NodeDefinition)
                and isinstance(
                    node.operation,
                    (declarations.SubroutineCall, declarations.WorkflowCall),
                )
                and isinstance(edge.visit, declarations.CallVisitDefinition)
            ):
                self._begin_call_activation(
                    frame,
                    node,
                    edge,
                    run_id,
                    budget,
                    params_registry,
                )
                return None
            self._walk(
                frame.graph,
                frame.scope,
                node,
                edge,
                frame.value,
                frame.state,
                run_id,
                budget,
                frame.graph_output,
                frame.resources,
                frame.graph_output.project_path,
                frame.params,
                params_registry,
            )
            return None
        if (
            isinstance(frame.control, _continuation.Terminal)
            and frame.control.outcome == "success"
        ):
            return self._finish_graph_activation(frame, run_id, budget)
        raise ValueError("checkpoint is not a resumable continuation")

    def _fail_call_activation(
        self,
        call: _LiveCallFrame,
        error: Exception,
        run_id: ids.RunId,
        /,
    ) -> None:
        if len(self._continuation_frames) < 2 or not isinstance(
            self._continuation_frames[-2], _LiveGraphFrame
        ):
            raise RuntimeError("failed call has no parent graph frame")
        parent = self._continuation_frames[-2]
        self._call_note(
            parent,
            error,
            node_id=call.node_id,
            edge_id=call.incoming_edge_id,
            visit_path=call.visit_path,
            restored=call.restored,
        )
        try:
            if call.running_emitted:
                self._emit_node(
                    parent.graph.id,
                    call.node_id,
                    run_id,
                    ExecutionStatus.FAILED,
                    call.prior_state,
                    parent.graph_output.project_path,
                )
        finally:
            if call.timing is not None:
                call.timing.finish("failed")
            popped = self._continuation_frames.pop()
            if popped is not call:
                raise RuntimeError("failed call activation did not unwind")

    def _recover_activation_error(
        self,
        error: Exception,
        run_id: ids.RunId,
        budget: _calls.Budget,
        /,
    ) -> bool:
        active_error = error
        while self._continuation_frames:
            top = self._continuation_frames[-1]
            if isinstance(top, _LiveCallFrame):
                self._fail_call_activation(top, active_error, run_id)
                continue
            frame = top
            try:
                self._record_failure(
                    frame.graph,
                    active_error,
                    run_id,
                    frame.graph_output,
                    frame.graph_output.project_path,
                )
            except Exception as cleanup_error:
                active_error = cleanup_error
            finally:
                popped = self._continuation_frames.pop()
                if popped is not frame:
                    raise RuntimeError("failed graph activation did not unwind")
                try:
                    frame.graph_output.statistics.write()
                except Exception as cleanup_error:
                    active_error = cleanup_error
            if not self._continuation_frames:
                if active_error is not error:
                    raise active_error
                return False
            call = self._continuation_frames[-1]
            if not isinstance(call, _LiveCallFrame):
                raise RuntimeError("failed child graph has no call activation")
            self._accept_child_return(
                call,
                None,
                active_error,
                run_id,
                budget,
            )
            return True
        return False

    def _abort_activations(
        self, status: statistics_module.TimingStatus, /
    ) -> None:
        while self._continuation_frames:
            frame = self._continuation_frames.pop()
            try:
                if isinstance(frame, _LiveCallFrame):
                    if frame.timing is not None:
                        frame.timing.finish(status)
                else:
                    frame.graph_output.statistics.write()
            except Exception:
                continue

    def _drive_activations(
        self,
        run_id: ids.RunId,
        budget: _calls.Budget,
        params_registry: _ParameterRegistry,
        /,
    ) -> declarations.Success[object, declarations.WorkflowState[Any]]:
        while self._continuation_frames:
            try:
                self._cancellation.raise_if_cancelled()
                top = self._continuation_frames[-1]
                if isinstance(top, _LiveCallFrame):
                    self._step_call_activation(
                        top,
                        run_id,
                        budget,
                        params_registry,
                    )
                    continue
                result = self._step_graph_activation(
                    top,
                    run_id,
                    budget,
                    params_registry,
                )
                if result is not None:
                    return result
            except Exception as error:
                pending = error
                while True:
                    try:
                        recovered = self._recover_activation_error(
                            pending,
                            run_id,
                            budget,
                        )
                    except Exception as recovery_error:
                        pending = recovery_error
                        if not self._continuation_frames:
                            raise
                        continue
                    if not recovered:
                        if pending is error:
                            raise
                        raise pending from error
                    break
            except BaseException as error:
                self._abort_activations(
                    "cancelled"
                    if isinstance(error, cancellation_module.ExecutionCancelled)
                    else "failed"
                )
                raise
        raise RuntimeError("workflow activation stack ended without a result")

    def _run_graph(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
        scope: _calls.CallScope,
        value: object,
        run_id: ids.RunId,
        budget: _calls.Budget,
        output_root: pathlib.Path,
        call_path: pathlib.Path,
        project_path: str,
        params_registry: _ParameterRegistry,
        resource_arguments: _agents.Resources,
        *,
        definition_reference: _continuation.DefinitionReference,
        params_override: object = _USE_REGISTERED_PARAMS,
        check_output_transport: Callable[[object], bool] | None = None,
        entry_values: tuple[_configuration.ConfigurationValue, ...] = (),
    ) -> declarations.Success[object, declarations.WorkflowState[ScopeT]]:
        if self._continuation_frames:
            raise RuntimeError(
                "root graph execution requires an empty activation stack"
            )
        self._activation_root = call_path
        self._start_graph_activation(
            graph,
            scope,
            value,
            run_id,
            budget,
            output_root,
            call_path,
            self._project_root,
            project_path,
            params_registry,
            resource_arguments,
            definition_reference=definition_reference,
            params_override=params_override,
            check_output_transport=check_output_transport,
            entry_values=entry_values,
        )
        return cast(
            declarations.Success[object, declarations.WorkflowState[ScopeT]],
            self._drive_activations(run_id, budget, params_registry),
        )

    def _finish_success(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
        terminal: _Terminal[ScopeT],
        run_id: ids.RunId,
        graph_output: _GraphOutput,
        check_transport: Callable[[object], bool] | None,
    ) -> declarations.Success[object, declarations.WorkflowState[ScopeT]]:
        with graph_output.node(graph.exit.id, "exit"):
            self._emit_node(
                graph.id,
                graph.exit.id,
                run_id,
                ExecutionStatus.RUNNING,
                None,
                graph_output.project_path,
            )
            if check_transport is not None:
                self._run_check(
                    check_transport,
                    terminal.output,
                    graph.exit.id,
                    "child_output_not_transportable",
                )
            self._emit_node(
                graph.id,
                graph.exit.id,
                run_id,
                ExecutionStatus.SUCCEEDED,
                None,
                graph_output.project_path,
            )
            return declarations.Success(
                output=terminal.output, state=terminal.state
            )

    def _record_failure(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, Any],
        error: Exception,
        run_id: ids.RunId,
        graph_output: _GraphOutput,
        project_path: str,
    ) -> None:
        with graph_output.node(
            graph.failure.id, "failure", status="failed"
        ) as output_dir:
            self._emit_node(
                graph.id,
                graph.failure.id,
                run_id,
                ExecutionStatus.RUNNING,
                None,
                project_path,
            )
            output_path = output_dir.relative_to(graph_output.root).as_posix()
            error.add_note(
                "Verdog failure boundary: "
                f"project={project_path} graph={graph.id} "
                f"failure={graph.failure.id} output={output_path}"
            )
            self._write_stacktrace(output_dir, error)
            self._emit_node(
                graph.id,
                graph.failure.id,
                run_id,
                ExecutionStatus.FAILED,
                None,
                project_path,
            )

    def _enter(
        self,
        node_id: ids.NodeId,
        run_id: ids.RunId,
        graph_output: _GraphOutput,
        configuration: tuple[_configuration.ConfigurationValue, ...],
    ) -> None:
        with graph_output.node(node_id, "enter"):
            self._emit_node(
                graph_output.graph_id,
                node_id,
                run_id,
                ExecutionStatus.RUNNING,
                None,
                graph_output.project_path,
            )
            _configuration.write_configuration(
                _contained(graph_output.root, graph_output.report_relative),
                configuration,
            )
            self._emit_node(
                graph_output.graph_id,
                node_id,
                run_id,
                ExecutionStatus.SUCCEEDED,
                None,
                graph_output.project_path,
            )

    def _walk(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
        scope: _calls.CallScope,
        entity: FeatureNode | declarations.NodeDefinition[Any, ScopeT],
        incoming_edge: Edge,
        value: object,
        state: declarations.WorkflowState[ScopeT],
        run_id: ids.RunId,
        budget: _calls.Budget,
        graph_output: _GraphOutput,
        resources: _agents.Resources,
        project_path: str,
        params: object,
        params_registry: _ParameterRegistry,
    ) -> None:
        if isinstance(entity, declarations.FeatureNodeDefinition):
            output, candidate = self._execute_feature_node(
                graph,
                entity,
                incoming_edge,
                value,
                state,
                run_id,
                graph_output,
                params,
            )
        elif isinstance(
            entity.operation,
            (declarations.SubroutineCall, declarations.WorkflowCall),
        ) and isinstance(incoming_edge.visit, declarations.CallVisitDefinition):
            raise RuntimeError("durable call reached the ordinary graph walker")
        else:
            output, candidate = self._execute_ordinary_entity(
                entity,
                incoming_edge,
                value,
                state,
                run_id,
                scope,
                budget,
                graph_output,
                resources,
                params,
                params_registry,
            )
        self._route(
            graph,
            entity.id,
            output,
            candidate,
            state,
            run_id,
            budget,
            graph_output,
            project_path,
        )

    @contextlib.contextmanager
    def _visit(
        self,
        node_id: ids.NodeId,
        node_type: str,
        incoming_edge: Edge,
        previous_state: object,
        run_id: ids.RunId,
        graph_output: _GraphOutput,
        params: object,
    ) -> Generator[
        tuple[
            graph_declarations.VisitImplementation,
            declarations.NodeContext[object],
        ]
    ]:
        with graph_output.node(node_id, node_type) as output_dir:
            try:
                visit = incoming_edge.visit
                if (
                    not isinstance(visit, declarations.VisitDefinition)
                    or visit.implementation is None
                ):
                    _errors.fault(
                        node_id,
                        "missing_visit",
                        (
                            "an executable node must be entered "
                            "through an implemented visit"
                        ),
                    )
                context = declarations.NodeContext(
                    run_id=run_id,
                    graph_id=graph_output.graph_id,
                    node_id=node_id,
                    edge_id=incoming_edge.id,
                    output_dir=output_dir,
                    params=params,
                )
                self._emit_node(
                    context.graph_id,
                    node_id,
                    run_id,
                    ExecutionStatus.RUNNING,
                    previous_state,
                    graph_output.project_path,
                )
                yield visit.implementation, context
            except Exception as error:
                error.add_note(
                    "Verdog node: "
                    f"project={graph_output.project_path} "
                    f"graph={graph_output.graph_id} "
                    f"node={node_id} edge={incoming_edge.id} "
                    f"output={output_dir.relative_to(graph_output.root).as_posix()}"
                )
                self._emit_node(
                    graph_output.graph_id,
                    node_id,
                    run_id,
                    ExecutionStatus.FAILED,
                    previous_state,
                    graph_output.project_path,
                )
                raise

    def _execute_feature_node(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
        node: FeatureNode,
        incoming_edge: Edge,
        value: object,
        state: declarations.WorkflowState[ScopeT],
        run_id: ids.RunId,
        graph_output: _GraphOutput,
        params: object,
    ) -> tuple[object, declarations.WorkflowState[ScopeT]]:
        with self._visit(
            node.id,
            "feature",
            incoming_edge,
            None,
            run_id,
            graph_output,
            params,
        ) as (implementation, context):
            result = feature.execute(implementation, value, state, context)
            if not isinstance(result, declarations.FeatureSuccess):
                _errors.fault(
                    node.id,
                    "invalid_result",
                    "feature node did not return FeatureSuccess",
                )
            success = cast(declarations.FeatureSuccess[ScopeT], result)
            candidate = self._validate_feature_successor(
                graph, node, success.state, state
            )
            self._emit_node(
                graph.id,
                node.id,
                run_id,
                ExecutionStatus.SUCCEEDED,
                None,
                graph_output.project_path,
            )
            return value, candidate

    def _run_durable_workflow(
        self,
        operation: declarations.WorkflowCall,
        call: _LiveCallFrame,
        request: _CallRequest,
        parent: _LiveGraphFrame,
        run_id: ids.RunId,
        budget: _calls.Budget,
        /,
        *,
        resume_continuation: bytes | None = None,
    ) -> object:
        owner_path, _ = _calls.resolve_call_project(
            parent.project_root, operation.project_path, call.node_id
        )
        if owner_path == ".":
            _calls.require_local_workflow(parent.scope, operation)
        graph_output = parent.graph_output

        def handle_event(event: _protocol.EventFrame) -> None:
            self._forward_child_event(event, budget)

        def handle_checkpoint(frame: _protocol.CheckpointFrame) -> None:
            self._accept_remote_checkpoint(call, frame, run_id, budget)

        arguments: dict[str, object] = {
            "project_root": parent.project_root,
            "project_path": operation.project_path,
            "definition_id": operation.definition_id,
            "definition_module": operation.definition_module,
            "input": request.input,
            "run_id": run_id,
            "transitions_remaining": budget.remaining,
            "output_dir": graph_output.root,
            "call_path": self._child_activation_parent(call),
            "owner_project_path": parent.project_path,
            "event_handler": handle_event,
            "checkpoint_handler": handle_checkpoint,
            "started_at": graph_output.statistics.started_at,
            "timing_handler": graph_output.statistics.forward,
            "checkpointing": self._checkpointing,
            "resume_continuation": resume_continuation,
            "retry_incomplete": self._retry_incomplete,
            "check_cancelled": self._cancellation.raise_if_cancelled,
            "deadline": self._cancellation.deadline,
        }
        if request.params_override:
            arguments["params_override"] = request.params
        output, budget.remaining = child_module.invoke(**arguments)  # type: ignore[arg-type]
        return output

    @staticmethod
    def _call_request(
        child_input: object, child_params: object, /
    ) -> _CallRequest:
        if child_params is _OMITTED_CHILD_PARAMS:
            return _CallRequest(
                input=child_input,
                params=None,
                params_override=False,
            )
        return _CallRequest(
            input=child_input,
            params=child_params,
            params_override=True,
        )

    @staticmethod
    def _snapshot_call_request(request: _CallRequest, /) -> _CallRequest:
        try:
            copied = cloudpickle.loads(cloudpickle.dumps(request))
        except Exception:
            try:
                copied = copy.deepcopy(request)
            except Exception:
                return request
        if not isinstance(copied, _CallRequest):  # pragma: no cover - invariant
            raise TypeError("copied child call request has the wrong type")
        return copied

    @staticmethod
    def _same_call_value(expected: object, actual: object, /) -> bool:
        if expected is actual:
            return True
        if type(expected) is not type(actual):
            return False
        try:
            equal = expected == actual
            if equal if type(equal) is bool else bool(equal):
                return True
        except Exception:
            pass
        try:
            return cloudpickle.dumps(expected) == cloudpickle.dumps(actual)
        except Exception:
            return False

    @classmethod
    def _same_call_request(
        cls,
        expected: _CallRequest,
        actual: _CallRequest,
        /,
    ) -> bool:
        return (
            expected.params_override == actual.params_override
            and cls._same_call_value(expected.input, actual.input)
            and cls._same_call_value(expected.params, actual.params)
        )

    @staticmethod
    def _close_awaitable(value: Awaitable[Any], /) -> None:
        close = getattr(value, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _call_context(
        node_context: declarations.NodeContext[object],
        child_params: object,
        invoke: Callable[[object, object], object],
        /,
    ) -> declarations.CallContext[object, object, object, object]:
        # Pylint misses the dataclass fields inherited from NodeContext.
        # pylint: disable-next=unexpected-keyword-arg
        return declarations.CallContext(
            run_id=node_context.run_id,
            graph_id=node_context.graph_id,
            node_id=node_context.node_id,
            edge_id=node_context.edge_id,
            output_dir=node_context.output_dir,
            params=node_context.params,
            child_params=child_params,
            _invoke=invoke,
        )

    def _invoke_call_visit(
        self,
        visit: declarations.CallVisitDefinition[
            Any, Any, Any, Any, Any, Any, Any
        ],
        value: object,
        previous_state: object,
        node_context: declarations.NodeContext[object],
        child_params: object,
        invoke: Callable[[object, object], object],
        /,
    ) -> object:
        implementation = cast(Callable[..., object], visit.implementation)
        result = implementation(
            value,
            previous_state,
            self._call_context(node_context, child_params, invoke),
        )
        if isinstance(result, Awaitable):
            self._close_awaitable(cast(Awaitable[Any], result))
            _errors.fault(
                node_context.node_id,
                "async_call_visit",
                "call visit implementation must be synchronous",
            )
        return result

    def _capture_call_request(
        self,
        visit: declarations.CallVisitDefinition[
            Any, Any, Any, Any, Any, Any, Any
        ],
        value: object,
        previous_state: object,
        node_context: declarations.NodeContext[object],
        child_params: object,
        /,
    ) -> _CallRequest:
        invocations = 0

        def capture(child_input: object, selected_params: object) -> object:
            nonlocal invocations
            invocations += 1
            if invocations != 1:
                _errors.fault(
                    node_context.node_id,
                    "call_invocation_count",
                    "a call node must invoke its child exactly once",
                )
            raise _ChildCallRequested(
                self._call_request(child_input, selected_params)
            )

        try:
            self._invoke_call_visit(
                visit,
                value,
                previous_state,
                node_context,
                child_params,
                capture,
            )
        except _ChildCallRequested as requested:
            return requested.request
        if invocations:
            _errors.fault(
                node_context.node_id,
                "call_control_intercepted",
                "call visit intercepted its durable child invocation",
            )
        _errors.fault(
            node_context.node_id,
            "call_invocation_count",
            "a call node must invoke its child exactly once",
        )

    def _replay_call_visit(
        self,
        visit: declarations.CallVisitDefinition[
            Any, Any, Any, Any, Any, Any, Any
        ],
        value: object,
        previous_state: object,
        node_context: declarations.NodeContext[object],
        child_params: object,
        request: _CallRequest,
        child_output: object,
        child_error: Exception | None,
        /,
    ) -> declarations.Success[object, object]:
        replay = _CallReplayController(
            node_id=node_context.node_id,
            request=request,
            child_output=child_output,
            child_error=child_error,
            make_request=self._call_request,
            requests_match=self._same_call_request,
        )
        try:
            result = self._invoke_call_visit(
                visit,
                value,
                previous_state,
                node_context,
                child_params,
                replay.invoke,
            )
        finally:
            replay.enforce()
        if replay.invocations != 1:
            _errors.fault(
                node_context.node_id,
                "call_invocation_count",
                "a call node must invoke its child exactly once",
            )
        if not isinstance(result, declarations.Success):
            _errors.fault(
                node_context.node_id,
                "invalid_result",
                "call visit did not return Success",
            )
        return cast(declarations.Success[object, object], result)

    def _execute_ordinary_entity(
        self,
        entity: declarations.NodeDefinition[Any, ScopeT],
        incoming_edge: Edge,
        value: object,
        state: declarations.WorkflowState[ScopeT],
        run_id: ids.RunId,
        scope: _calls.CallScope,
        budget: _calls.Budget,
        graph_output: _GraphOutput,
        resources: _agents.Resources,
        params: object,
        params_registry: _ParameterRegistry,
    ) -> tuple[object, declarations.WorkflowState[ScopeT]]:
        previous_state: object = state.get(entity)
        if isinstance(
            entity.operation,
            (declarations.SubroutineCall, declarations.WorkflowCall),
        ):
            _errors.fault(
                entity.id,
                "durable_call_visit_required",
                "call nodes require a synchronous CallVisitDefinition",
            )
        handler = self._node_handler(
            entity.operation,
            entity.id,
            value,
            previous_state,
            budget,
            resources,
        )
        with self._visit(
            entity.id,
            handler.kind,
            incoming_edge,
            previous_state,
            run_id,
            graph_output,
            params,
        ) as (implementation, context):
            result = handler.execute(implementation, context)
            if not isinstance(result, declarations.Success):
                _errors.fault(
                    entity.id,
                    "invalid_result",
                    "entity did not return Success",
                )
            success = cast(declarations.Success[object, object], result)
            candidate = self._ordinary_successor(entity, success.state, state)
            self._emit_node(
                context.graph_id,
                entity.id,
                run_id,
                ExecutionStatus.SUCCEEDED,
                candidate.get(entity),
                graph_output.project_path,
            )
            return success.output, candidate

    def _node_handler(
        self,
        operation: operations.Operation,
        node_id: ids.NodeId,
        value: object,
        previous_state: object,
        budget: _calls.Budget,
        resources: _agents.Resources,
    ) -> _NodeHandler:
        if isinstance(operation, declarations.Agent):
            invocation_resources = _agents.Resources(
                resources.profiles,
                resources.sessions,
                resources.invocation_journal,
                budget.remaining,
            )
            return _NodeHandler(
                "agent",
                lambda implementation, context: agent.execute(
                    operation,
                    implementation,
                    value,
                    previous_state,
                    context,
                    invocation_resources,
                    self._cancellation,
                ),
            )
        if isinstance(operation, declarations.Python):
            return _NodeHandler(
                "python",
                lambda implementation, context: python.execute(
                    implementation, value, previous_state, context
                ),
            )

        def unsupported(
            implementation: graph_declarations.VisitImplementation,
            context: declarations.NodeContext[object],
        ) -> object:
            _errors.fault(
                node_id,
                "unsupported_operation",
                (
                    f"{type(operation).__name__} is not an operation "
                    f"this runtime executes"
                ),
            )

        return _NodeHandler("unknown", unsupported)

    def _forward_child_event(
        self,
        event: _protocol.EventFrame,
        budget: _calls.Budget,
    ) -> None:
        budget.remaining = event.transitions_remaining
        if event.kind == "node":
            self._emit(
                NodeExecution(
                    run_id=event.run_id,
                    graph_id=event.graph_id,
                    node_id=ids.NodeId(event.entity_id),
                    status=ExecutionStatus(event.status),
                    state=None,
                    remote=True,
                    project_path=event.project_path,
                )
            )
        else:
            self._emit(
                EdgeExecution(
                    run_id=event.run_id,
                    graph_id=event.graph_id,
                    edge_id=ids.EdgeId(event.entity_id),
                    status=ExecutionStatus(event.status),
                    state=None,
                    remote=True,
                    project_path=event.project_path,
                )
            )

    def _route(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
        source: ids.NodeId,
        value: object,
        candidate_state: declarations.WorkflowState[ScopeT],
        rollback_state: declarations.WorkflowState[ScopeT],
        run_id: ids.RunId,
        budget: _calls.Budget,
        graph_output: _GraphOutput,
        project_path: str,
    ) -> None:
        outgoing = self._outgoing(graph, source)
        compatible: list[tuple[Edge, declarations.WorkflowState[ScopeT]]] = []
        for edge in outgoing:
            edge_state = self._edge_candidate(
                graph, edge, rollback_state, candidate_state
            )
            if edge_state is not None:
                compatible.append((edge, edge_state))
        if len(compatible) != 1:
            _errors.fault(
                source,
                "routing_failed",
                (
                    f"entity {source} has {len(compatible)} "
                    f"compatible outgoing edges; expected 1"
                ),
            )
        edge, next_state = compatible[0]
        budget.consume(edge.id)
        self._emit_edge(graph.id, edge.id, run_id, next_state, project_path)
        frame = self._continuation_frames[-1]
        if not isinstance(frame, _LiveGraphFrame) or frame.graph is not graph:
            raise RuntimeError(
                "workflow continuation stack does not match the graph"
            )
        frame.value = value
        frame.state = next_state
        completed = self._boundary(
            graph_output,
            source,
            graph_output.visits.get(source, 0),
        )
        if edge.target == graph.failure.id:
            frame.control = _continuation.Terminal(outcome="failure")
            self._checkpoint(
                run_id,
                budget,
                run_store.CheckpointKind.TERMINAL,
                completed,
                None,
                restorable=False,
                unavailable_code="checkpoint.failure_port",
                unavailable_reason=(
                    "the completed node routed to the failure port"
                ),
            )
            error = RuntimeError(
                f"subroutine {graph.id} reached its failure port"
            )
            source_note = (
                f"project={project_path} graph={graph.id} source={source}"
            )
            source_output = graph_output.latest(source)
            if source_output is not None:
                source_note += (
                    " output="
                    + source_output.relative_to(graph_output.root).as_posix()
                )
            error.add_note("Verdog failure source: " + source_note)
            error.add_note(
                "Verdog failure edge: "
                f"project={project_path} graph={graph.id} edge={edge.id}"
            )
            raise error
        if edge.target == graph.exit.id:
            frame.control = _continuation.Terminal(outcome="success")
            self._checkpoint(
                run_id,
                budget,
                run_store.CheckpointKind.TERMINAL,
                completed,
                None,
            )
            return
        frame.control = _continuation.Ready(
            incoming_edge_id=edge.id,
            target_node_id=edge.target,
        )
        next_boundary = self._boundary(
            graph_output,
            edge.target,
            graph_output.visits.get(edge.target, 0) + 1,
        )
        self._checkpoint(
            run_id,
            budget,
            run_store.CheckpointKind.ENTRY
            if source == graph.enter.id
            else run_store.CheckpointKind.NODE,
            completed,
            next_boundary,
        )

    def _edge_candidate(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
        edge: Edge,
        source_state: declarations.WorkflowState[ScopeT],
        successor_state: declarations.WorkflowState[ScopeT],
    ) -> declarations.WorkflowState[ScopeT] | None:
        feature_by_id: dict[
            ids.FeatureId, declarations.FeatureDefinition[Any, ScopeT]
        ] = {feature.id: feature for feature in graph.features}
        try:
            if not feature_semantics.evaluate_conditions(
                edge.conditions, source_state, feature_by_id
            ):
                return None
        except ValueError as error:
            error.add_note(f"Verdog edge condition: {edge.id}")
            raise
        effects = (
            *edge.effects,
            *feature_semantics.analyze_effects(
                feature_by_id, edge.effects
            ).inferred,
        )
        try:
            if not feature_semantics.effects_satisfied(
                effects,
                source_state,
                successor_state,
                feature_by_id,
            ):
                return None
        except ValueError as error:
            error.add_note(f"Verdog edge effect: {edge.id}")
            raise
        return successor_state

    def _run_check(
        self,
        check: Callable[[object], object],
        value: object,
        entity_id: ids.NodeId,
        code: str,
    ) -> None:
        try:
            accepted = check(value)
        except Exception as error:
            error.add_note(f"Verdog check: entity={entity_id} phase={code}")
            raise
        if accepted is not True:
            _errors.fault(
                entity_id,
                code,
                f"{getattr(check, '__name__', 'check')} returned false",
            )

    def _ordinary_successor(
        self,
        entity: declarations.NodeDefinition[Any, ScopeT],
        next_state: object,
        source: declarations.WorkflowState[ScopeT],
    ) -> declarations.WorkflowState[ScopeT]:
        if type(next_state) is not entity.state_type:
            _errors.fault(
                entity.id,
                "invalid_state",
                "entity state has the wrong type",
            )
        try:
            validation.require_immutable_state(next_state)
        except TypeError as error:
            error.add_note(f"Verdog state returned by node {entity.id}")
            raise
        return source._replace(  # pyright: ignore[reportPrivateUsage]
            entity, next_state
        )

    def _validate_feature_successor(
        self,
        graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
        node: FeatureNode,
        successor: object,
        source: declarations.WorkflowState[ScopeT],
    ) -> declarations.WorkflowState[ScopeT]:
        if not isinstance(successor, declarations.FeatureState):
            _errors.fault(
                node.id,
                "invalid_feature_state",
                "FeatureSuccess.state must be FeatureState",
            )
        feature_state = cast(declarations.FeatureState[ScopeT], successor)
        candidate = feature_state._as_workflow_state()  # pyright: ignore[reportPrivateUsage]
        try:
            changed = source._changes(  # pyright: ignore[reportPrivateUsage]
                candidate
            )
        except ValueError as error:
            error.add_note(f"Verdog feature state returned by node {node.id}")
            raise
        feature_by_identity: dict[
            int, declarations.FeatureDefinition[Any, ScopeT]
        ] = {id(feature): feature for feature in graph.features}
        for key in changed:
            feature = feature_by_identity.get(id(key))
            if feature is None or feature is not key:
                _errors.fault(
                    node.id,
                    "unauthorized_state_change",
                    f"feature node {node.id} changed state {key.state_key}",
                )
            value = candidate.get(feature)
            if value is None:
                _errors.fault(
                    node.id,
                    "invalid_feature_value",
                    f"feature node {node.id} deinitialized "
                    f"feature {feature.id}",
                )
            try:
                feature_semantics.validate_feature_value(feature, value)
            except ValueError as error:
                error.add_note(
                    f"Verdog feature value returned by node "
                    f"{node.id}: {feature.id}"
                )
                raise
        return candidate

    def _emit(self, event: ExecutionEvent, /) -> None:
        if self._execution_handler is not None:
            self._execution_handler(event)

    def _emit_node(
        self,
        graph_id: ids.GraphId,
        node_id: ids.NodeId,
        run_id: ids.RunId,
        status: ExecutionStatus,
        state: object,
        project_path: str,
        /,
    ) -> None:
        self._emit(
            NodeExecution(
                run_id=run_id,
                graph_id=graph_id,
                node_id=node_id,
                status=status,
                state=state,
                project_path=project_path,
            )
        )

    def _emit_edge(
        self,
        graph_id: ids.GraphId,
        edge_id: ids.EdgeId,
        run_id: ids.RunId,
        state: declarations.WorkflowState[Any],
        project_path: str,
        /,
    ) -> None:
        self._emit(
            EdgeExecution(
                run_id=run_id,
                graph_id=graph_id,
                edge_id=edge_id,
                status=ExecutionStatus.SUCCEEDED,
                state=state,
                project_path=project_path,
            )
        )

    @staticmethod
    def _write_stacktrace(
        output_dir: pathlib.Path,
        error: Exception,
    ) -> None:
        (output_dir / "stacktrace.txt").write_text(
            "".join(traceback.format_exception(error)), "utf-8"
        )

    @staticmethod
    def _nodes(
        graph: declarations.GraphDefinition[Any, Any, Any, ScopeT],
    ) -> dict[
        ids.NodeId, FeatureNode | declarations.NodeDefinition[Any, ScopeT]
    ]:
        return {node.id: node for node in graph.nodes}

    @staticmethod
    def _outgoing(
        graph: declarations.GraphDefinition[Any, Any, Any, Any],
        source: ids.NodeId,
    ) -> tuple[Edge, ...]:
        return tuple(edge for edge in graph.edges if edge.source == source)
