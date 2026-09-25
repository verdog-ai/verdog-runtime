from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

from .._definitions import load_definition
from .._process import normalize_project_path, resolve_project_root
from ..declarations import (
    SubroutineCall,
    SubroutineDefinition,
    WorkflowCall,
    WorkflowDefinition,
)
from ..declarations.ids import EdgeId, GraphId, NodeId, is_valid_definition_id

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ParamsT = TypeVar("ParamsT")
ScopeT = TypeVar("ScopeT")


class Budget:
    def __init__(self, limit: int) -> None:
        self.remaining = limit

    def consume(self, edge_id: EdgeId) -> None:
        if self.remaining <= 0:
            error = RuntimeError(
                "workflow transition limit exceeded [transition_limit]"
            )
            error.add_note(f"Verdog edge: {edge_id}")
            raise error
        self.remaining -= 1


@dataclass(frozen=True, slots=True)
class CallScope:
    current: GraphId
    root: GraphId
    root_workflow_id: GraphId | None = None


def _definition_path(definition_id: GraphId, /) -> tuple[str, str]:
    package, separator, local = str(definition_id).rpartition(".")
    if not is_valid_definition_id(local):
        raise ValueError(f"invalid definition id: {definition_id}")
    return (package if separator else ""), local


def _root_definition_id(definition_id: GraphId, /) -> GraphId:
    package, local = _definition_path(definition_id)
    root = local.split("__", 1)[0]
    return GraphId(f"{package}.{root}" if package else root)


def _visible(scope: CallScope, definition_id: GraphId, /) -> bool:
    target_package, target = _definition_path(definition_id)
    current_package, current = _definition_path(scope.current)
    if target_package != current_package:
        return False
    owner, separator, _ = target.rpartition("__")
    if not separator:
        return definition_id == scope.root
    while True:
        if owner == current:
            return True
        current, separator, _ = current.rpartition("__")
        if not separator:
            return False


def local_subroutine(
    project_root: Path, scope: CallScope, operation: SubroutineCall, /
) -> tuple[SubroutineDefinition[object, object, object, object], CallScope]:
    if not _visible(scope, operation.definition_id):
        raise LookupError(
            f"subroutine call target is not lexically visible: {operation.definition_id}"
        )
    child = _load_subroutine(project_root, operation)
    return child, replace(scope, current=child.graph.id)


def require_local_workflow(scope: CallScope, operation: WorkflowCall, /) -> None:
    if operation.definition_id != scope.root_workflow_id and not _visible(
        scope, operation.definition_id
    ):
        raise LookupError(
            f"workflow call target is not lexically visible: {operation.definition_id}"
        )


def subroutine_scope(
    project_root: Path, operation: SubroutineCall, /
) -> tuple[CallScope, SubroutineDefinition[object, object, object, object]]:
    target = _load_subroutine(project_root, operation)
    return CallScope(target.graph.id, _root_definition_id(target.graph.id)), target


def _load_subroutine(
    project_root: Path,
    operation: SubroutineCall,
    /,
) -> SubroutineDefinition[object, object, object, object]:
    definition_type: type[SubroutineDefinition[object, object, object, object]] = (
        SubroutineDefinition
    )
    target, _ = load_definition(
        project_root,
        operation.definition_module,
        operation.definition_id,
        definition_type,
        lambda definition: definition.graph.id,
    )
    return target


def workflow_scope(
    definition: WorkflowDefinition[InputT, OutputT, ParamsT, ScopeT],
    /,
) -> CallScope:
    root = definition.entry.definition_id
    return CallScope(root, _root_definition_id(root), definition.id)


def workflow_subroutine(
    project_root: Path,
    definition: WorkflowDefinition[InputT, OutputT, ParamsT, ScopeT],
    /,
) -> tuple[SubroutineDefinition[object, object, object, object], CallScope]:
    if normalize_project_path(definition.entry.project_path) != ".":
        raise ValueError("workflow entry subroutine must be local")
    return local_subroutine(project_root, workflow_scope(definition), definition.entry)


def resolve_call_project(
    project_root: Path,
    project_path: str,
    node_id: NodeId,
    /,
) -> tuple[str, Path]:
    try:
        return (
            normalize_project_path(project_path),
            resolve_project_root(project_root, project_path),
        )
    except ValueError as error:
        error.add_note(f"Verdog child project for call node {node_id} is invalid")
        raise
