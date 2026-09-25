from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType, UnionType
from typing import Any, Generic, TypeAlias, TypeVar, override

from typing_extensions import TypeForm

from .calls import CallVisitDefinition
from .configuration import WorkflowConfiguration
from .ids import (
    AgentProfileId,
    AgentSessionId,
    EdgeId,
    FeatureId,
    GraphId,
    NodeId,
    ParameterAddress,
)
from .interfaces import AgentProfileInitializer
from .keys import StateAddress, StateKey
from .operations import Feature, Operation, ParameterType, SubroutineCall


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
StateT = TypeVar("StateT")
ScopeT = TypeVar("ScopeT")
FeatureValue: TypeAlias = bool | int | float | str
FeatureValueT = TypeVar("FeatureValueT", bound=FeatureValue)
ParamsT = TypeVar("ParamsT")
_ContractType: TypeAlias = TypeForm[InputT] | UnionType


class FeatureKind(StrEnum):
    BOOLEAN = "boolean"
    ENUM = "enum"
    INTEGER = "integer"
    FLOAT = "float"


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentProfileDefinition(Generic[InputT, ParamsT]):
    id: AgentProfileId
    name: str
    implementation: AgentProfileInitializer[InputT, ParamsT]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSessionDefinition:
    id: AgentSessionId
    name: str
    persistent: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentProfileParameter:
    id: AgentProfileId
    name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSessionParameter:
    id: AgentSessionId
    name: str


class BooleanConditionObservation(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class NumericalConditionObservation(StrEnum):
    EQUAL_ZERO = "equal_zero"
    GREATER_ZERO = "greater_zero"


class EnumConditionObservation(StrEnum):
    EQUAL = "equal"


class BooleanEffectObservation(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCHANGED = "unchanged"
    UNCONSTRAINED = "unconstrained"


class NumericalEffectObservation(StrEnum):
    INCREASES = "increases"
    DECREASES = "decreases"
    UNCHANGED = "unchanged"
    UNCONSTRAINED = "unconstrained"


class EnumEffectObservation(StrEnum):
    EQUAL = "equal"
    UNCHANGED = "unchanged"
    UNCONSTRAINED = "unconstrained"


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureDefinition(
    StateKey[FeatureValueT | None, ScopeT],
    Generic[FeatureValueT, ScopeT],
):
    id: FeatureId
    label: str
    description: str
    kind: FeatureKind
    values: tuple[str, ...] = ()

    @property
    @override
    def state_key(self) -> StateAddress:
        return ("feature", str(self.id))


@dataclass(frozen=True, slots=True, kw_only=True)
class BooleanFeatureCondition:
    feature_id: FeatureId
    observation: BooleanConditionObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericalFeatureCondition:
    feature_id: FeatureId
    observation: NumericalConditionObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class EnumFeatureCondition:
    feature_id: FeatureId
    observation: EnumConditionObservation
    value: str


FeatureCondition: TypeAlias = (
    BooleanFeatureCondition | EnumFeatureCondition | NumericalFeatureCondition
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BooleanFeatureEffect:
    feature_id: FeatureId
    observation: BooleanEffectObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericalFeatureEffect:
    feature_id: FeatureId
    observation: NumericalEffectObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class EnumFeatureEffect:
    feature_id: FeatureId
    observation: EnumEffectObservation
    value: str | None = None


FeatureEffect: TypeAlias = (
    BooleanFeatureEffect | EnumFeatureEffect | NumericalFeatureEffect
)


VisitImplementation: TypeAlias = Callable[..., object]


@dataclass(frozen=True, slots=True, kw_only=True)
class VisitDefinition:
    implementation: VisitImplementation | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeDefinition(StateKey[StateT, ScopeT], Generic[StateT, ScopeT]):
    id: NodeId
    name: str
    operation: Operation
    state_type: type[StateT]

    @property
    @override
    def state_key(self) -> StateAddress:
        return ("node", str(self.id))


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureNodeDefinition:
    id: NodeId
    name: str
    operation: Feature


@dataclass(frozen=True, slots=True, kw_only=True)
class PortDefinition:
    id: NodeId


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeDefinition:
    id: EdgeId
    source: NodeId
    target: NodeId
    name: str = ""
    conditions: tuple[FeatureCondition, ...] = ()
    effects: tuple[FeatureEffect, ...] = ()
    visit: (
        VisitDefinition | CallVisitDefinition[Any, Any, Any, Any, Any, Any, Any] | None
    ) = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphDefinition(Generic[InputT, OutputT, ParamsT, ScopeT]):
    id: GraphId
    params_type: _ContractType[ParamsT]
    enter: PortDefinition
    exit: PortDefinition
    failure: PortDefinition
    nodes: tuple[NodeDefinition[Any, ScopeT] | FeatureNodeDefinition, ...]
    edges: tuple[EdgeDefinition, ...]
    features: tuple[
        FeatureDefinition[bool, ScopeT]
        | FeatureDefinition[int, ScopeT]
        | FeatureDefinition[float, ScopeT]
        | FeatureDefinition[str, ScopeT],
        ...,
    ] = ()
    profiles: tuple[AgentProfileDefinition[InputT, ParamsT], ...] = ()
    profile_parameters: tuple[AgentProfileParameter, ...] = ()
    sessions: tuple[AgentSessionDefinition, ...] = ()
    session_parameters: tuple[AgentSessionParameter, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowDefinition(Generic[InputT, OutputT, ParamsT, ScopeT]):
    id: GraphId
    input_type: _ContractType[InputT]
    entry: SubroutineCall
    configuration: WorkflowConfiguration
    sessions: tuple[AgentSessionDefinition, ...] = ()
    params_types: Mapping[ParameterAddress, ParameterType] = field(init=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("workflow definition id must not be empty")
        object.__setattr__(
            self,
            "params_types",
            MappingProxyType(dict(self.entry.params_types)),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SubroutineDefinition(Generic[InputT, OutputT, ParamsT, ScopeT]):
    graph: GraphDefinition[InputT, OutputT, ParamsT, ScopeT]

    def __post_init__(self) -> None:
        if not self.graph.id:
            raise ValueError("subroutine definition id must not be empty")
