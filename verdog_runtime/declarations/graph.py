"""Typed graph declarations and qualitative feature observations."""

from __future__ import annotations

import dataclasses
import enum
import types
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeAlias, TypeVar, override

from typing_extensions import TypeForm

from verdog_runtime.declarations import calls as call_declarations
from verdog_runtime.declarations import (
    configuration as configuration_declarations,
)
from verdog_runtime.declarations import ids, interfaces, keys, operations

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
StateT = TypeVar("StateT")
ScopeT = TypeVar("ScopeT")
FeatureValue: TypeAlias = bool | int | float | str
FeatureValueT = TypeVar("FeatureValueT", bound=FeatureValue)
ParamsT = TypeVar("ParamsT")
_ContractType: TypeAlias = TypeForm[InputT] | types.UnionType


class FeatureKind(enum.StrEnum):
    """The value domain of a declared qualitative feature."""

    BOOLEAN = "boolean"
    ENUM = "enum"
    INTEGER = "integer"
    FLOAT = "float"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentProfileDefinition(Generic[InputT, ParamsT]):
    """A local profile initialized from graph input and parameters."""

    id: ids.AgentProfileId
    name: str
    implementation: interfaces.AgentProfileInitializer[InputT, ParamsT]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentSessionDefinition:
    """A named session whose persistence is fixed by the declaration."""

    id: ids.AgentSessionId
    name: str
    persistent: bool


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentProfileParameter:
    """A profile slot supplied by the caller of a graph."""

    id: ids.AgentProfileId
    name: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentSessionParameter:
    """A session slot supplied by the caller of a graph."""

    id: ids.AgentSessionId
    name: str


class BooleanConditionObservation(enum.StrEnum):
    """A required Boolean value before an edge is taken."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class NumericalConditionObservation(enum.StrEnum):
    """A required zero or positive value before an edge is taken."""

    EQUAL_ZERO = "equal_zero"
    GREATER_ZERO = "greater_zero"


class EnumConditionObservation(enum.StrEnum):
    """An equality condition on an enumerated feature."""

    EQUAL = "equal"


class BooleanEffectObservation(enum.StrEnum):
    """A required Boolean value or change across a transition."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCHANGED = "unchanged"
    UNCONSTRAINED = "unconstrained"


class NumericalEffectObservation(enum.StrEnum):
    """A required numerical change across a transition."""

    INCREASES = "increases"
    DECREASES = "decreases"
    UNCHANGED = "unchanged"
    UNCONSTRAINED = "unconstrained"


class EnumEffectObservation(enum.StrEnum):
    """A required enum value or change across a transition."""

    EQUAL = "equal"
    UNCHANGED = "unchanged"
    UNCONSTRAINED = "unconstrained"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FeatureDefinition(
    keys.StateKey[FeatureValueT | None, ScopeT],
    Generic[FeatureValueT, ScopeT],
):
    """A typed state slot with its label, domain, and stable declaration ID."""

    id: ids.FeatureId
    label: str
    description: str
    kind: FeatureKind
    values: tuple[str, ...] = ()

    @property
    @override
    def state_key(self) -> keys.StateAddress:
        return ("feature", str(self.id))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BooleanFeatureCondition:
    """A Boolean feature and its required source observation."""

    feature_id: ids.FeatureId
    observation: BooleanConditionObservation


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NumericalFeatureCondition:
    """A numerical feature and its required source observation."""

    feature_id: ids.FeatureId
    observation: NumericalConditionObservation


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EnumFeatureCondition:
    """An enum feature and its required source value."""

    feature_id: ids.FeatureId
    observation: EnumConditionObservation
    value: str


FeatureCondition: TypeAlias = (
    BooleanFeatureCondition | EnumFeatureCondition | NumericalFeatureCondition
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BooleanFeatureEffect:
    """A Boolean feature and its required transition observation."""

    feature_id: ids.FeatureId
    observation: BooleanEffectObservation


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NumericalFeatureEffect:
    """A numerical feature and its required transition observation."""

    feature_id: ids.FeatureId
    observation: NumericalEffectObservation


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EnumFeatureEffect:
    """An enum feature and its required observation or target value."""

    feature_id: ids.FeatureId
    observation: EnumEffectObservation
    value: str | None = None


FeatureEffect: TypeAlias = (
    BooleanFeatureEffect | EnumFeatureEffect | NumericalFeatureEffect
)


VisitImplementation: TypeAlias = Callable[..., object]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class VisitDefinition:
    """The optional implementation attached to an ordinary edge visit."""

    implementation: VisitImplementation | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NodeDefinition(keys.StateKey[StateT, ScopeT], Generic[StateT, ScopeT]):
    """An operation and its per-node state contract."""

    id: ids.NodeId
    name: str
    operation: operations.Operation
    state_type: type[StateT]

    @property
    @override
    def state_key(self) -> keys.StateAddress:
        return ("node", str(self.id))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FeatureNodeDefinition:
    """A node whose operation may update the feature state."""

    id: ids.NodeId
    name: str
    operation: operations.Feature


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PortDefinition:
    """An entry, success, or failure boundary identified within a graph."""

    id: ids.NodeId


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EdgeDefinition:
    """A directed transition with source conditions, effects, and a visit."""

    id: ids.EdgeId
    source: ids.NodeId
    target: ids.NodeId
    name: str = ""
    conditions: tuple[FeatureCondition, ...] = ()
    effects: tuple[FeatureEffect, ...] = ()
    visit: (
        VisitDefinition
        | call_declarations.CallVisitDefinition[
            Any, Any, Any, Any, Any, Any, Any
        ]
        | None
    ) = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class GraphDefinition(Generic[InputT, OutputT, ParamsT, ScopeT]):
    """A subroutine graph and its parameter, state, and resource contracts."""

    id: ids.GraphId
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowDefinition(Generic[InputT, OutputT, ParamsT, ScopeT]):
    """An entry subroutine with workflow resources and an input type."""

    id: ids.GraphId
    input_type: _ContractType[InputT]
    entry: operations.SubroutineCall
    configuration: configuration_declarations.WorkflowConfiguration
    sessions: tuple[AgentSessionDefinition, ...] = ()
    params_types: Mapping[ids.ParameterAddress, operations.ParameterType] = (
        dataclasses.field(init=False)
    )

    def __post_init__(self) -> None:
        """Validate the workflow ID and freeze the entry parameter types."""
        if not self.id:
            raise ValueError("workflow definition id must not be empty")
        object.__setattr__(
            self,
            "params_types",
            types.MappingProxyType(dict(self.entry.params_types)),
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SubroutineDefinition(Generic[InputT, OutputT, ParamsT, ScopeT]):
    """A callable graph evaluated within its caller process."""

    graph: GraphDefinition[InputT, OutputT, ParamsT, ScopeT]

    def __post_init__(self) -> None:
        """Validate the graph ID used to address this subroutine."""
        if not self.graph.id:
            raise ValueError("subroutine definition id must not be empty")
