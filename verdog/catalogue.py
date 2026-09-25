"""Canonical, read-only catalogue metadata for one local workflow.

Publication and inspection must describe a workflow identically. Keeping that
projection here gives the CLI one serializer for both operations; importantly, it
reads manifests and requirements but neither imports authored modules nor writes a
generated file.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, cast

from .local import (
    SCHEMA_VERSION,
    Clone,
    LocalDefinition,
    WorkspaceError,
    dependency_clones,
)
from .requirements import canonical_python_specifier, canonical_requirements


@dataclass(frozen=True, slots=True)
class EnvironmentDescription:
    """The declarative inputs needed to prepare one workflow environment."""

    python: str
    schema_version: int
    requirements: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "schema_version": self.schema_version,
            "requirements": list(self.requirements),
        }


@dataclass(frozen=True, slots=True)
class WorkflowDescription:
    """The publisher claim for one workflow at one immutable source revision."""

    package: str
    workflow_id: str
    display_name: str
    preview: dict[str, Any]
    closure: tuple[dict[str, Any], ...]
    environment: EnvironmentDescription

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "package": self.package,
            "workflow_id": self.workflow_id,
            "display_name": self.display_name,
            "preview": self.preview,
            "closure": list(self.closure),
            "environment": self.environment.as_json(),
        }


def describe_workflow(
    clone: Clone, workflow_id: str | None = None
) -> WorkflowDescription:
    """Describe one saved workflow without service access, generation, or execution."""

    workflow = clone.workflow_definition(workflow_id)
    display_name, preview = _preview(clone, workflow)
    package = clone.project.get("package")
    if not isinstance(package, str):
        raise WorkspaceError("project.json has no package")
    try:
        python = canonical_python_specifier(_python_requirement())
        requirements = canonical_requirements(clone.workflow_requirements(workflow))
    except ValueError as error:
        raise WorkspaceError(str(error)) from error
    return WorkflowDescription(
        package=package,
        workflow_id=workflow.local_id,
        display_name=display_name,
        preview=preview,
        closure=dependency_closure(clone),
        environment=EnvironmentDescription(
            python=python,
            schema_version=SCHEMA_VERSION,
            requirements=requirements,
        ),
    )


def preview_for(clone: Clone, workflow_id: str) -> dict[str, Any]:
    """Only the stable graph preview, retained for callers that need no environment."""

    _, preview = _preview(clone, clone.workflow_definition(workflow_id))
    return preview


def _preview(clone: Clone, workflow: LocalDefinition) -> tuple[str, dict[str, Any]]:
    subroutine = clone.subroutine_for(workflow).body
    display_name = str(subroutine.get("name", workflow.local_id))
    preview: dict[str, Any] = {
        "name": display_name,
        "ports": _object(subroutine.get("ports")),
        "nodes": _objects(subroutine.get("nodes")),
        "edges": _objects(subroutine.get("edges")),
        "features": _objects(subroutine.get("features")),
    }
    return display_name, preview


def _python_requirement() -> str:
    """Use the installed runtime's constraint instead of repeating it here."""

    try:
        requirement = metadata.metadata("verdog-runtime").get("Requires-Python")
    except metadata.PackageNotFoundError as error:
        raise WorkspaceError(
            "verdog_runtime is not installed; reinstall verdog-runtime"
        ) from error
    if not requirement:
        raise WorkspaceError(
            "verdog_runtime has no Python compatibility metadata; reinstall verdog-runtime"
        )
    return requirement


def dependency_closure(clone: Clone) -> tuple[dict[str, Any], ...]:
    """The unique immutable repository releases reachable from this project."""

    found: list[dict[str, Any]] = []
    walked: set[tuple[str, str, str]] = set()
    for pin, _ in dependency_clones(clone)[1:]:
        assert pin is not None
        owner = str(pin.get("owner", ""))
        name = str(pin.get("name", ""))
        commit = str(pin.get("commit", ""))
        release = (owner, name, commit)
        if release in walked:
            continue
        walked.add(release)
        identifier = pin.get("repository_id")
        if isinstance(identifier, int):
            found.append(
                {
                    "repository_id": identifier,
                    "owner": owner,
                    "name": name,
                    "commit": commit,
                }
            )
    return tuple(found)


def _object(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        cast(dict[str, Any], item)
        for item in cast(list[object], value)
        if isinstance(item, dict)
    ]


__all__ = [
    "EnvironmentDescription",
    "WorkflowDescription",
    "dependency_closure",
    "describe_workflow",
    "preview_for",
]
