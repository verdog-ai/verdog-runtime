from __future__ import annotations

from dataclasses import MISSING, Field, fields, is_dataclass
from enum import Enum
from inspect import signature
from pathlib import PurePath
from typing import Any, cast

from ..declarations import (
    Agent,
    BooleanFeatureCondition,
    BooleanFeatureEffect,
    CallVisitDefinition,
    EdgeDefinition,
    EnumConditionObservation,
    EnumEffectObservation,
    EnumFeatureCondition,
    EnumFeatureEffect,
    Feature,
    FeatureDefinition,
    FeatureKind,
    FeatureNodeDefinition,
    GraphDefinition,
    NumericalFeatureCondition,
    NumericalFeatureEffect,
    NodeDefinition,
    PortDefinition,
    SubroutineCall,
    WorkflowCall,
    WorkflowState,
    VisitDefinition,
)
from ..declarations.graph import (
    FeatureCondition,
    FeatureEffect,
)
from ..declarations.ids import (
    AgentProfileId,
    AgentSessionId,
    EdgeId,
    FeatureId,
    NodeId,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_instance(value: object, expected: type[object], message: str) -> None:
    _require(isinstance(value, expected), message)


def require_immutable_state(
    value: object,
    /,
    *,
    _seen: set[int] | None = None,
    _label: str = "workflow state",
) -> None:
    """Reject state values which can be mutated through a workflow snapshot."""

    if value is None or isinstance(
        value, (bool, int, float, complex, str, bytes, Enum, PurePath)
    ):
        return
    if isinstance(value, WorkflowState):
        return
    seen: set[int] = _seen if _seen is not None else set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (tuple, frozenset)):
        for item in cast(tuple[object, ...] | frozenset[object], value):
            require_immutable_state(item, _seen=seen, _label=_label)
        return
    if is_dataclass(value) and getattr(type(value), "__dataclass_params__").frozen:
        for field in fields(value):
            require_immutable_state(
                getattr(value, field.name), _seen=seen, _label=_label
            )
        return
    raise TypeError(f"{_label} must be immutable, got {type(value).__name__}")


def _validate_node_state_type(state_type: type[object], node_id: NodeId, /) -> None:
    label = f"entity {node_id} state type"
    if not is_dataclass(state_type):
        raise TypeError(f"{label} must be a dataclass record")
    if state_type.__bases__ != (object,):
        raise TypeError(f"{label} must not inherit fields")

    parameters = getattr(state_type, "__dataclass_params__")
    if not parameters.frozen:
        raise TypeError(f"{label} must be frozen")
    if not parameters.init:
        raise TypeError(f"{label} must use the generated initializer")

    declared = tuple(getattr(state_type, "__annotations__", ()))
    all_fields = tuple(getattr(state_type, "__dataclass_fields__", ()))
    record_fields = fields(state_type)
    field_names = tuple(field.name for field in record_fields)
    if declared != all_fields or declared != field_names:
        raise TypeError(
            f"{label} must declare only direct instance fields; "
            "inherited fields, InitVar, and ClassVar are unsupported"
        )

    slot_names = _state_slot_names(state_type)
    if "__dict__" in state_type.__dict__ or set(slot_names) - {"__weakref__"} != set(
        field_names
    ):
        raise TypeError(f"{label} must use slots")

    for record_field in record_fields:
        _validate_record_field(record_field, label)

    try:
        signature(state_type).bind()
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be default-constructible") from error


def _state_slot_names(state_type: type[object], /) -> tuple[str, ...]:
    slots = cast(object, state_type.__dict__.get("__slots__"))
    if isinstance(slots, str):
        return (slots,)
    if isinstance(slots, tuple):
        raw_slots = cast(tuple[object, ...], slots)
        return (
            cast(tuple[str, ...], raw_slots)
            if all(isinstance(slot, str) for slot in raw_slots)
            else ()
        )
    return ()


def _validate_record_field(record_field: Field[Any], label: str, /) -> None:
    field_label = f"{label} field {record_field.name}"
    if (
        not record_field.init
        or not record_field.repr
        or not record_field.compare
        or record_field.hash is not None
        or record_field.metadata
    ):
        raise TypeError(f"{field_label} must not customize field()")
    if record_field.default_factory is not MISSING:
        raise TypeError(f"{field_label} must use a direct default")
    if record_field.default is MISSING:
        raise TypeError(f"{field_label} must have a default")
    require_immutable_state(record_field.default, _label=f"{field_label} default")


def _validate_enum_condition(
    edge_id: EdgeId,
    reference: EnumFeatureCondition,
    values: tuple[str, ...],
) -> None:
    _require(
        reference.observation is EnumConditionObservation.EQUAL,
        f"edge {edge_id} enum feature {reference.feature_id} has an invalid observation",
    )
    _require(
        reference.value in values,
        f"edge {edge_id} enum feature {reference.feature_id} has an unknown value",
    )


def _validate_enum_effect(
    edge_id: EdgeId,
    reference: EnumFeatureEffect,
    values: tuple[str, ...],
) -> None:
    require_instance(
        reference.observation,
        EnumEffectObservation,
        f"edge {edge_id} enum feature {reference.feature_id} has an invalid observation",
    )
    _require(
        (reference.value is not None)
        == (reference.observation is EnumEffectObservation.EQUAL),
        f"edge {edge_id} enum feature {reference.feature_id} has an invalid value",
    )
    if reference.value is not None:
        _require(
            reference.value in values,
            f"edge {edge_id} enum feature {reference.feature_id} has an unknown value",
        )


def _validate_feature_reference(
    edge_id: EdgeId,
    reference: FeatureCondition | FeatureEffect,
    features: dict[FeatureId, FeatureDefinition[Any, Any]],
) -> None:
    _require(
        reference.feature_id in features,
        f"edge {edge_id} references unknown feature: {reference.feature_id}",
    )
    feature = features[reference.feature_id]
    wrong_kind = (
        f"edge {edge_id} feature {reference.feature_id} has the wrong observation type"
    )
    if isinstance(reference, EnumFeatureCondition):
        _require(feature.kind is FeatureKind.ENUM, wrong_kind)
        _validate_enum_condition(edge_id, reference, feature.values)
    elif isinstance(reference, EnumFeatureEffect):
        _require(feature.kind is FeatureKind.ENUM, wrong_kind)
        _validate_enum_effect(edge_id, reference, feature.values)
    else:
        _require(
            (
                feature.kind is FeatureKind.BOOLEAN
                and isinstance(
                    reference, (BooleanFeatureCondition, BooleanFeatureEffect)
                )
            )
            or (
                feature.kind in (FeatureKind.INTEGER, FeatureKind.FLOAT)
                and isinstance(
                    reference, (NumericalFeatureCondition, NumericalFeatureEffect)
                )
            ),
            wrong_kind,
        )


def _validate_session(
    session_id: AgentSessionId, name: str, sessions: set[AgentSessionId], /
) -> None:
    _require(bool(session_id), "agent session id must not be empty")
    _require(bool(name), f"agent session {session_id} name must not be empty")
    _require(session_id not in sessions, f"duplicate agent session id: {session_id}")


def _validate_edge_visit(
    edge: EdgeDefinition,
    target: FeatureNodeDefinition | NodeDefinition[Any, Any] | PortDefinition,
    source: FeatureNodeDefinition | NodeDefinition[Any, Any] | PortDefinition,
    enter_id: NodeId,
    /,
) -> None:
    if isinstance(target, PortDefinition):
        _require(
            edge.visit is None,
            f"edge {edge.id} targeting a port must not declare a visit",
        )
        return
    visit = edge.visit
    if isinstance(visit, CallVisitDefinition):
        _require(
            isinstance(target, NodeDefinition)
            and isinstance(target.operation, (SubroutineCall, WorkflowCall)),
            f"call visit {edge.id} may only target a subroutine or workflow call node",
        )
        _require(
            callable(visit.implementation),
            f"call visit {edge.id} implementation must be callable",
        )
    elif isinstance(visit, VisitDefinition):
        _require(
            not (
                isinstance(target, NodeDefinition)
                and isinstance(target.operation, (SubroutineCall, WorkflowCall))
            ),
            f"edge {edge.id} targeting a call node must declare CallVisitDefinition",
        )
        _require(
            callable(visit.implementation),
            f"visit {edge.id} implementation must be callable",
        )
    else:
        raise ValueError(
            f"edge {edge.id} targeting node {target.id} must declare a visit"
        )
    if isinstance(source, PortDefinition):
        _require(
            source.id == enter_id,
            f"edge {edge.id} has a terminal port source: {source.id}",
        )


def _graph_entities(
    graph: GraphDefinition[Any, Any, Any, Any], /
) -> dict[NodeId, FeatureNodeDefinition | NodeDefinition[Any, Any] | PortDefinition]:
    entities: tuple[
        tuple[
            str,
            FeatureNodeDefinition | NodeDefinition[Any, Any] | PortDefinition,
        ],
        ...,
    ] = (
        ("enter port", graph.enter),
        ("exit port", graph.exit),
        ("failure port", graph.failure),
        *(("node", node) for node in graph.nodes),
    )
    found: dict[
        NodeId, FeatureNodeDefinition | NodeDefinition[Any, Any] | PortDefinition
    ] = {}
    for kind, entity in entities:
        if kind.endswith("port"):
            _require(
                isinstance(entity, PortDefinition),
                f"{kind} must be a PortDefinition",
            )
        _require(bool(entity.id), f"{kind} id must not be empty")
        _require(entity.id not in found, f"duplicate graph entity id: {entity.id}")
        found[entity.id] = entity
        if isinstance(entity, PortDefinition):
            _require(
                kind.endswith("port"),
                f"graph nodes must not contain a PortDefinition: {entity.id}",
            )
            continue
        if isinstance(entity, FeatureNodeDefinition):
            operation = cast(object, entity.operation)
            _require(
                isinstance(operation, Feature),
                f"feature node {entity.id} must declare a Feature operation",
            )
            continue
        _validate_node_state_type(entity.state_type, entity.id)
        operation = entity.operation
        _require(
            not isinstance(operation, Feature),
            f"node {entity.id} must use FeatureNodeDefinition for a Feature operation",
        )
        if isinstance(operation, (SubroutineCall, WorkflowCall)):
            _require(
                bool(operation.definition_id),
                f"node {entity.id} child definition id must not be empty",
            )
            _require(
                bool(operation.project_path),
                f"node {entity.id} child project path must not be empty",
            )
    return found


def _graph_features(
    graph: GraphDefinition[Any, Any, Any, Any], /
) -> dict[FeatureId, FeatureDefinition[Any, Any]]:
    found: dict[FeatureId, FeatureDefinition[Any, Any]] = {}
    for feature in graph.features:
        _require(bool(feature.id), "feature id must not be empty")
        _require(bool(feature.label), f"feature {feature.id} label must not be empty")
        _require(feature.id not in found, f"duplicate feature id: {feature.id}")
        if feature.kind is FeatureKind.ENUM:
            _require(
                bool(feature.values)
                and all(type(value) is str and bool(value) for value in feature.values)
                and len(feature.values) == len(set(feature.values)),
                f"enum feature {feature.id} must have a non-empty unique string domain",
            )
        else:
            _require(
                not feature.values,
                f"non-enum feature {feature.id} must not declare enum values",
            )
        found[feature.id] = feature
    return found


def _validate_graph_edges(
    graph: GraphDefinition[Any, Any, Any, Any],
    entities: dict[
        NodeId, FeatureNodeDefinition | NodeDefinition[Any, Any] | PortDefinition
    ],
    features: dict[FeatureId, FeatureDefinition[Any, Any]],
    /,
) -> None:
    edge_by_id: dict[EdgeId, EdgeDefinition] = {}
    for edge in graph.edges:
        _require(bool(edge.id), "edge id must not be empty")
        _require(edge.id not in edge_by_id, f"duplicate edge id: {edge.id}")
        edge_by_id[edge.id] = edge
    for edge in graph.edges:
        _require(
            edge.source in entities, f"edge {edge.id} has unknown source: {edge.source}"
        )
        _require(
            edge.target in entities, f"edge {edge.id} has unknown target: {edge.target}"
        )
        _validate_edge_visit(
            edge,
            entities[edge.target],
            entities[edge.source],
            graph.enter.id,
        )
        if edge.effects:
            _require(
                isinstance(entities[edge.source], FeatureNodeDefinition),
                f"edge {edge.id} has effects but its source is not a feature node",
            )
        condition_ids = [condition.feature_id for condition in edge.conditions]
        effect_ids = [effect.feature_id for effect in edge.effects]
        _require(
            len(condition_ids) == len(set(condition_ids)),
            f"edge {edge.id} has duplicate feature conditions",
        )
        _require(
            len(effect_ids) == len(set(effect_ids)),
            f"edge {edge.id} has duplicate feature effects",
        )
        for reference in (*edge.conditions, *edge.effects):
            _validate_feature_reference(edge.id, reference, features)


def _validate_port_routes(graph: GraphDefinition[Any, Any, Any, Any], /) -> None:
    _require(
        any(edge.source == graph.enter.id for edge in graph.edges),
        "enter port must have an outgoing edge",
    )
    _require(
        not any(edge.target == graph.enter.id for edge in graph.edges),
        "enter port must not have an incoming edge",
    )
    _require(
        not any(edge.source == graph.exit.id for edge in graph.edges),
        "exit port must not have an outgoing edge",
    )
    _require(
        not any(edge.source == graph.failure.id for edge in graph.edges),
        "failure port must not have an outgoing edge",
    )


def _validate_graph_resources(graph: GraphDefinition[Any, Any, Any, Any], /) -> None:
    profiles: set[AgentProfileId] = set()
    for definition in graph.profiles:
        _require(bool(definition.id), "agent profile id must not be empty")
        _require(
            bool(definition.name),
            f"agent profile {definition.id} name must not be empty",
        )
        _require(
            definition.id not in profiles,
            f"duplicate agent profile id: {definition.id}",
        )
        _require(
            callable(definition.implementation),
            f"agent profile {definition.id} implementation must be callable",
        )
        profiles.add(definition.id)
    for parameter in graph.profile_parameters:
        _require(bool(parameter.id), "agent profile id must not be empty")
        _require(
            bool(parameter.name),
            f"agent profile {parameter.id} name must not be empty",
        )
        _require(
            parameter.id not in profiles,
            f"duplicate agent profile id: {parameter.id}",
        )
        profiles.add(parameter.id)

    sessions: set[AgentSessionId] = set()
    for definition in graph.sessions:
        _validate_session(definition.id, definition.name, sessions)
        require_instance(
            definition.persistent,
            bool,
            f"agent session {definition.id} persistent must be boolean",
        )
        sessions.add(definition.id)
    for parameter in graph.session_parameters:
        _validate_session(parameter.id, parameter.name, sessions)
        sessions.add(parameter.id)

    for node in graph.nodes:
        if not isinstance(node.operation, Agent):
            continue
        operation = node.operation
        _require(
            operation.profile in profiles,
            f"agent node {node.id} references unknown profile: {operation.profile}",
        )
        _require(
            operation.session in sessions,
            f"agent node {node.id} references unknown session: {operation.session}",
        )


def validate_graph(graph: GraphDefinition[Any, Any, Any, Any]) -> None:
    """Check one graph's structure. Its boundary types are not this function's business.

    `Any` rather than type variables, and that is the point rather than laziness. Nothing below
    reads `InputT` or `OutputT` -- only `enter`, `exit`, `failure` and `nodes` -- so the
    parameters are decoration. The dispatcher validates one concrete subroutine definition at
    a time, so its workflow envelope and boundary configuration are irrelevant here.
    """

    entities = _graph_entities(graph)
    features = _graph_features(graph)
    _validate_graph_edges(graph, entities, features)
    _validate_port_routes(graph)
    _validate_graph_resources(graph)
