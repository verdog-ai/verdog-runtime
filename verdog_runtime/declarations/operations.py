"""Operation declarations for Python, features, agents, and child calls."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import TypeAlias, cast

from typing_extensions import TypeForm

from verdog_runtime.declarations import ids

ParameterType: TypeAlias = TypeForm[object] | types.UnionType


def _arguments(value: object, name: str, /) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, str] = {}
    for target, source in cast(Mapping[object, object], value).items():
        if (
            not isinstance(target, str)
            or not isinstance(source, str)
            or not ids.is_valid_entity_id(target)
            or not ids.is_valid_entity_id(source)
        ):
            raise ValueError(f"{name} must map valid identifiers")
        result[target] = source
    return types.MappingProxyType(result)


def _module(value: object, /) -> str:
    if not isinstance(value, str):
        raise TypeError("definition_module must be a string")
    if not value or any(
        not ids.is_valid_entity_id(part) for part in value.split(".")
    ):
        raise ValueError("definition_module must be an absolute dotted module")
    return value


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Agent:
    """Invoke an agent using the specified profile and session."""

    profile: ids.AgentProfileId
    session: ids.AgentSessionId


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Python:
    """Execute an ordinary Python visit."""

    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Feature:
    """Execute a visit that can replace feature values."""

    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SubroutineCall:
    """A child subroutine invoked in the process running its caller.

    The definition is resolved lazily in the caller's process and environment.
    ``project_path`` retains its owning clone so nested calls resolve relative
    to it.
    """

    definition_id: ids.GraphId
    definition_module: str
    params_types: Mapping[ids.ParameterAddress, ParameterType]
    profile_arguments: Mapping[ids.AgentProfileId, ids.AgentProfileId]
    session_arguments: Mapping[ids.AgentSessionId, ids.AgentSessionId]
    project_path: str = "."

    def __post_init__(self) -> None:
        """Validate module and freeze profile and session argument maps."""
        object.__setattr__(
            self, "definition_module", _module(self.definition_module)
        )
        object.__setattr__(
            self,
            "profile_arguments",
            _arguments(self.profile_arguments, "profile_arguments"),
        )
        object.__setattr__(
            self,
            "session_arguments",
            _arguments(self.session_arguments, "session_arguments"),
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowCall:
    """A child workflow invoked in a process of its own."""

    definition_id: ids.GraphId
    definition_module: str
    project_path: str = "."

    def __post_init__(self) -> None:
        """Validate absolute module path for child workflow declaration."""
        object.__setattr__(
            self, "definition_module", _module(self.definition_module)
        )


Operation: TypeAlias = Agent | Feature | Python | SubroutineCall | WorkflowCall
