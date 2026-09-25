"""Resolve child definitions and track lexical scope and transition budgets."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TypeVar

from verdog_runtime import _definitions, _process, declarations
from verdog_runtime.declarations import ids

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ParamsT = TypeVar("ParamsT")
ScopeT = TypeVar("ScopeT")


class Budget:
    def __init__(self, limit: int) -> None:
        self.remaining = limit

    def consume(self, edge_id: ids.EdgeId) -> None:
        if self.remaining <= 0:
            error = RuntimeError(
                "workflow transition limit exceeded [transition_limit]"
            )
            error.add_note(f"Verdog edge: {edge_id}")
            raise error
        self.remaining -= 1


@dataclasses.dataclass(frozen=True, slots=True)
class CallScope:
    current: ids.GraphId
    root: ids.GraphId
    root_workflow_id: ids.GraphId | None = None


def _definition_path(definition_id: ids.GraphId, /) -> tuple[str, str]:
    package, separator, local = str(definition_id).rpartition(".")
    if not ids.is_valid_definition_id(local):
        raise ValueError(f"invalid definition id: {definition_id}")
    return (package if separator else ""), local


def _root_definition_id(definition_id: ids.GraphId, /) -> ids.GraphId:
    package, local = _definition_path(definition_id)
    root = local.split("__", 1)[0]
    return ids.GraphId(f"{package}.{root}" if package else root)


def _visible(scope: CallScope, definition_id: ids.GraphId, /) -> bool:
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
    project_root: pathlib.Path,
    scope: CallScope,
    operation: declarations.SubroutineCall,
    /,
) -> tuple[
    declarations.SubroutineDefinition[object, object, object, object], CallScope
]:
    if not _visible(scope, operation.definition_id):
        raise LookupError(
            f"subroutine call target is not lexically visible: "
            f"{operation.definition_id}"
        )
    child = _load_subroutine(project_root, operation)
    return child, dataclasses.replace(scope, current=child.graph.id)


def require_local_workflow(
    scope: CallScope, operation: declarations.WorkflowCall, /
) -> None:
    if operation.definition_id != scope.root_workflow_id and not _visible(
        scope, operation.definition_id
    ):
        raise LookupError(
            f"workflow call target is not lexically visible: "
            f"{operation.definition_id}"
        )


def subroutine_scope(
    project_root: pathlib.Path, operation: declarations.SubroutineCall, /
) -> tuple[
    CallScope, declarations.SubroutineDefinition[object, object, object, object]
]:
    target = _load_subroutine(project_root, operation)
    return CallScope(
        target.graph.id, _root_definition_id(target.graph.id)
    ), target


def _load_subroutine(
    project_root: pathlib.Path,
    operation: declarations.SubroutineCall,
    /,
) -> declarations.SubroutineDefinition[object, object, object, object]:
    definition_type: type[
        declarations.SubroutineDefinition[object, object, object, object]
    ] = declarations.SubroutineDefinition
    target, _ = _definitions.load_definition(
        project_root,
        operation.definition_module,
        operation.definition_id,
        definition_type,
        lambda definition: definition.graph.id,
    )
    return target


def workflow_scope(
    definition: declarations.WorkflowDefinition[
        InputT, OutputT, ParamsT, ScopeT
    ],
    /,
) -> CallScope:
    root = definition.entry.definition_id
    return CallScope(root, _root_definition_id(root), definition.id)


def workflow_subroutine(
    project_root: pathlib.Path,
    definition: declarations.WorkflowDefinition[
        InputT, OutputT, ParamsT, ScopeT
    ],
    /,
) -> tuple[
    declarations.SubroutineDefinition[object, object, object, object], CallScope
]:
    if _process.normalize_project_path(definition.entry.project_path) != ".":
        raise ValueError("workflow entry subroutine must be local")
    return local_subroutine(
        project_root, workflow_scope(definition), definition.entry
    )


def resolve_call_project(
    project_root: pathlib.Path,
    project_path: str,
    node_id: ids.NodeId,
    /,
) -> tuple[str, pathlib.Path]:
    try:
        return (
            _process.normalize_project_path(project_path),
            _process.resolve_project_root(project_root, project_path),
        )
    except ValueError as error:
        error.add_note(
            f"Verdog child project for call node {node_id} is invalid"
        )
        raise
