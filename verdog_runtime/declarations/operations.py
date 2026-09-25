from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, UnionType
from typing import TypeAlias, cast

from typing_extensions import TypeForm

from .ids import (
    AgentProfileId,
    AgentSessionId,
    GraphId,
    ParameterAddress,
    is_valid_entity_id,
)


ParameterType: TypeAlias = TypeForm[object] | UnionType


def _arguments(value: object, name: str, /) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, str] = {}
    for target, source in cast(Mapping[object, object], value).items():
        if (
            not isinstance(target, str)
            or not isinstance(source, str)
            or not is_valid_entity_id(target)
            or not is_valid_entity_id(source)
        ):
            raise ValueError(f"{name} must map valid identifiers")
        result[target] = source
    return MappingProxyType(result)


def _module(value: object, /) -> str:
    if not isinstance(value, str):
        raise TypeError("definition_module must be a string")
    if not value or any(not is_valid_entity_id(part) for part in value.split(".")):
        raise ValueError("definition_module must be an absolute dotted module")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class Agent:
    profile: AgentProfileId
    session: AgentSessionId


@dataclass(frozen=True, slots=True, kw_only=True)
class Python:
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Feature:
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class SubroutineCall:
    """A child subroutine invoked in the process running its caller.

    The definition is resolved lazily in the caller's process and environment.
    ``project_path`` retains its owning clone so nested calls resolve relative to it.
    """

    definition_id: GraphId
    definition_module: str
    params_types: Mapping[ParameterAddress, ParameterType]
    profile_arguments: Mapping[AgentProfileId, AgentProfileId]
    session_arguments: Mapping[AgentSessionId, AgentSessionId]
    project_path: str = "."

    def __post_init__(self) -> None:
        object.__setattr__(self, "definition_module", _module(self.definition_module))
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


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowCall:
    """A child workflow invoked in a process of its own."""

    definition_id: GraphId
    definition_module: str
    project_path: str = "."

    def __post_init__(self) -> None:
        object.__setattr__(self, "definition_module", _module(self.definition_module))


Operation: TypeAlias = Agent | Feature | Python | SubroutineCall | WorkflowCall
