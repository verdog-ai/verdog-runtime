"""Value objects and constants for durable local workflow runs."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import pathlib
import types
from collections.abc import Mapping

# The registry and legacy run manifests retain their original schema. Public
# run-history documents and current storage manifests evolve independently
# because they have different compatibility lifetimes.
SCHEMA_VERSION = 1
RUN_HISTORY_SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA_VERSION = 2
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 3
CONTROL_DIRECTORY = ".verdog"
RUN_MANIFEST = "run.json"
CHECKPOINT_DIRECTORY = "checkpoints"
REGISTRY_PATH = pathlib.Path(".verdog/run-registry.json")
REGISTRY_LOCK = pathlib.Path(".verdog/locks/run-registry.lock")
EMPTY_COMPATIBILITY: Mapping[str, str] = types.MappingProxyType({})
EMPTY_SHARDS: Mapping[str, bytes] = types.MappingProxyType({})


class RunStoreError(ValueError):
    """A stable local-run error which a CLI may expose to a machine."""

    def __init__(
        self, message: str, *, code: str = "run.invalid", details: object = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class CheckpointPolicy(enum.StrEnum):
    OFF = "off"
    AUTO = "auto"
    REQUIRED = "required"


class CheckpointKind(enum.StrEnum):
    ENTRY = "entry"
    NODE = "node"
    CHILD_START = "child-start"
    CHILD_RETURN = "child-return"
    TERMINAL = "terminal"


@dataclasses.dataclass(frozen=True, slots=True)
class WorkflowIdentity:
    id: str
    definition_id: str
    module: str

    def as_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "definition_id": self.definition_id,
            "module": self.module,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class LaunchRecord:
    workflow_arguments: tuple[str, ...]
    checkpointing: CheckpointPolicy

    def as_json(self) -> dict[str, object]:
        return {
            "workflow_arguments": list(self.workflow_arguments),
            "checkpointing": self.checkpointing.value,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ParentRun:
    run_id: str
    operation: str
    checkpoint: int | None
    arguments: str

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "operation": self.operation,
            "checkpoint": self.checkpoint,
            "arguments": self.arguments,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointState:
    count: int = 0
    latest_completed: int | None = None
    latest_restorable: int | None = None
    resume_available: bool = False
    unavailable_code: str | None = "checkpoint.none"
    unavailable_reason: str | None = "this run has no committed checkpoint"

    def as_json(self) -> dict[str, object]:
        return {
            "count": self.count,
            "latest_completed": self.latest_completed,
            "latest_restorable": self.latest_restorable,
            "resume_available": self.resume_available,
            "unavailable_code": self.unavailable_code,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SessionIssue:
    address: str
    provider: str
    code: str
    message: str

    def as_json(self) -> dict[str, object]:
        return {
            "address": self.address,
            "provider": self.provider,
            "code": self.code,
            "message": self.message,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SessionState:
    persistent: int = 0
    model: str = "copy-on-write"
    branch_available: bool = True
    issues: tuple[SessionIssue, ...] = ()

    def as_json(self) -> dict[str, object]:
        return {
            "persistent": self.persistent,
            "model": self.model,
            "branch_available": self.branch_available,
            "issues": [issue.as_json() for issue in self.issues],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class RunManifest:
    id: str
    project_root: str
    directory_name: str
    workflow: WorkflowIdentity
    status: RunStatus
    started_at: str
    updated_at: str
    output_dir: str
    launch: LaunchRecord
    parent: ParentRun | None = None
    checkpoints: CheckpointState = CheckpointState()
    sessions: SessionState = SessionState()
    compatibility: Mapping[str, str] = EMPTY_COMPATIBILITY
    storage_schema_version: int = RUN_MANIFEST_SCHEMA_VERSION

    def as_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.storage_schema_version,
            "id": self.id,
            "project_root": self.project_root,
            "workflow": self.workflow.as_json(),
            "status": self.status.value,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "output_dir": self.output_dir,
            "launch": self.launch.as_json(),
            "parent": None if self.parent is None else self.parent.as_json(),
            "compatibility": dict(self.compatibility),
        }
        # Schema v1 cached directory-derived facts. Preserve that shape when an
        # existing v1 run is updated, but never write those caches for new v2
        # runs. Reads always recompute checkpoint truth from committed
        # checkpoint directories for both versions.
        if self.storage_schema_version == 1:
            value["directory_name"] = self.directory_name
            value["checkpoints"] = self.checkpoints.as_json()
            value["sessions"] = self.sessions.as_json()
        return value

    def as_summary(
        self, *, status: RunStatus | None = None
    ) -> dict[str, object]:
        value = self.as_json()
        value.pop("schema_version")
        value.pop("project_root")
        value.pop("compatibility")
        value["directory_name"] = self.directory_name
        value["checkpoints"] = self.checkpoints.as_json()
        value["sessions"] = self.sessions.as_json()
        if status is not None:
            value["status"] = status.value
        return value


@dataclasses.dataclass(frozen=True, slots=True)
class Boundary:
    project_path: str
    graph: str
    node: str
    visit: int
    call_path: str

    def as_json(self) -> dict[str, object]:
        return {
            "project_path": self.project_path,
            "graph": self.graph,
            "node": self.node,
            "visit": self.visit,
            "call_path": self.call_path,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointSummary:
    sequence: int
    created_at: str
    kind: CheckpointKind
    completed: Boundary | None
    next: Boundary | None
    restore_available: bool
    fork_with_branch_available: bool
    fork_with_fresh_available: bool
    unavailable_code: str | None = None
    unavailable_reason: str | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "kind": self.kind.value,
            "completed": None
            if self.completed is None
            else self.completed.as_json(),
            "next": None if self.next is None else self.next.as_json(),
            "restore_available": self.restore_available,
            "fork_with_branch_available": self.fork_with_branch_available,
            "fork_with_fresh_available": self.fork_with_fresh_available,
            "unavailable_code": self.unavailable_code,
            "unavailable_reason": self.unavailable_reason,
        }

    def as_summary(self) -> dict[str, object]:
        value = self.as_json()
        value.pop("schema_version")
        return value


def utc_now() -> str:
    """A canonical RFC 3339 UTC timestamp."""
    return (
        datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
    )
