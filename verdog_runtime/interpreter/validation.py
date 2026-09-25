"""Validate graph contracts and workflow state records."""

from __future__ import annotations

import dataclasses
import enum
import inspect
import pathlib
from typing import Any, cast

from verdog_runtime import declarations
from verdog_runtime.declarations import graph as graph_declarations
from verdog_runtime.declarations import ids


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_instance(
    value: object, expected: type[object], message: str
) -> None:
    """Raise ValueError with the supplied message unless the type matches."""
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
        value,
        (bool, int, float, complex, str, bytes, enum.Enum, pathlib.PurePath),
    ):
        return
    if isinstance(value, declarations.WorkflowState):
        return
    seen: set[int] = _seen if _seen is not None else set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (tuple, frozenset)):
        for item in cast(tuple[object, ...] | frozenset[object], value):
            require_immutable_state(item, _seen=seen, _label=_label)
        return
    if (
        dataclasses.is_dataclass(value)
        and getattr(type(value), "__dataclass_params__").frozen  # noqa: B009 - generated attribute absent from typeshed.
    ):
        for field in dataclasses.fields(value):
            require_immutable_state(
                getattr(value, field.name), _seen=seen, _label=_label
            )
        return
    raise TypeError(f"{_label} must be immutable, got {type(value).__name__}")


def _validate_node_state_type(
    state_type: type[object], node_id: ids.NodeId, /
) -> None:
    label = f"entity {node_id} state type"
    if not dataclasses.is_dataclass(state_type):
        raise TypeError(f"{label} must be a dataclass record")
    if state_type.__bases__ != (object,):
        raise TypeError(f"{label} must not inherit fields")

    parameters = getattr(state_type, "__dataclass_params__")  # noqa: B009 - generated attribute absent from typeshed.
    if not parameters.frozen:
        raise TypeError(f"{label} must be frozen")
    if not parameters.init:
        raise TypeError(f"{label} must use the generated initializer")

    declared = tuple(getattr(state_type, "__annotations__", ()))
    all_fields = tuple(getattr(state_type, "__dataclass_fields__", ()))
    record_fields = dataclasses.fields(state_type)
    field_names = tuple(field.name for field in record_fields)
    if declared != all_fields or declared != field_names:
        raise TypeError(
            f"{label} must declare only direct instance fields; "
            "inherited fields, InitVar, and ClassVar are unsupported"
        )

    slot_names = _state_slot_names(state_type)
    if "__dict__" in state_type.__dict__ or set(slot_names) - {
        "__weakref__"
    } != set(field_names):
        raise TypeError(f"{label} must use slots")

    for record_field in record_fields:
        _validate_record_field(record_field, label)

    try:
        inspect.signature(state_type).bind()
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


def _validate_record_field(
    record_field: dataclasses.Field[Any], label: str, /
) -> None:
    field_label = f"{label} field {record_field.name}"
    if (
        not record_field.init
        or not record_field.repr
        or not record_field.compare
        or record_field.hash is not None
        or record_field.metadata
    ):
        raise TypeError(f"{field_label} must not customize field()")
    if record_field.default_factory is not dataclasses.MISSING:
        raise TypeError(f"{field_label} must use a direct default")
    if record_field.default is dataclasses.MISSING:
        raise TypeError(f"{field_label} must have a default")
    require_immutable_state(
        record_field.default, _label=f"{field_label} default"
    )


def _validate_enum_condition(
    edge_id: ids.EdgeId,
    reference: declarations.EnumFeatureCondition,
    values: tuple[str, ...],
) -> None:
    _require(
        reference.observation is declarations.EnumConditionObservation.EQUAL,
        (
            f"edge {edge_id} enum feature {reference.feature_id} has "
            f"an invalid observation"
        ),
    )
    _require(
        reference.value in values,
        (
            f"edge {edge_id} enum feature {reference.feature_id} has "
            f"an unknown value"
        ),
    )


def _validate_enum_effect(
    edge_id: ids.EdgeId,
    reference: declarations.EnumFeatureEffect,
    values: tuple[str, ...],
) -> None:
    require_instance(
        reference.observation,
        declarations.EnumEffectObservation,
        (
            f"edge {edge_id} enum feature {reference.feature_id} has "
            f"an invalid observation"
        ),
    )
    _require(
        (reference.value is not None)
        == (reference.observation is declarations.EnumEffectObservation.EQUAL),
        (
            f"edge {edge_id} enum feature {reference.feature_id} has "
            f"an invalid value"
        ),
    )
    if reference.value is not None:
        _require(
            reference.value in values,
            (
                f"edge {edge_id} enum feature {reference.feature_id} "
                f"has an unknown value"
            ),
        )


def _validate_feature_reference(
    edge_id: ids.EdgeId,
    reference: graph_declarations.FeatureCondition
    | graph_declarations.FeatureEffect,
    features: dict[ids.FeatureId, declarations.FeatureDefinition[Any, Any]],
) -> None:
    _require(
        reference.feature_id in features,
        f"edge {edge_id} references unknown feature: {reference.feature_id}",
    )
    feature = features[reference.feature_id]
    wrong_kind = (
        f"edge {edge_id} feature {reference.feature_id} "
        f"has the wrong observation type"
    )
    if isinstance(reference, declarations.EnumFeatureCondition):
        _require(feature.kind is declarations.FeatureKind.ENUM, wrong_kind)
        _validate_enum_condition(edge_id, reference, feature.values)
    elif isinstance(reference, declarations.EnumFeatureEffect):
        _require(feature.kind is declarations.FeatureKind.ENUM, wrong_kind)
        _validate_enum_effect(edge_id, reference, feature.values)
    else:
        _require(
            (
                feature.kind is declarations.FeatureKind.BOOLEAN
                and isinstance(
                    reference,
                    (
                        declarations.BooleanFeatureCondition,
                        declarations.BooleanFeatureEffect,
                    ),
                )
            )
            or (
                feature.kind
                in (
                    declarations.FeatureKind.INTEGER,
                    declarations.FeatureKind.FLOAT,
                )
                and isinstance(
                    reference,
                    (
                        declarations.NumericalFeatureCondition,
                        declarations.NumericalFeatureEffect,
                    ),
                )
            ),
            wrong_kind,
        )


def _validate_session(
    session_id: ids.AgentSessionId,
    name: str,
    sessions: set[ids.AgentSessionId],
    /,
) -> None:
    _require(bool(session_id), "agent session id must not be empty")
    _require(bool(name), f"agent session {session_id} name must not be empty")
    _require(
        session_id not in sessions, f"duplicate agent session id: {session_id}"
    )


def _validate_edge_visit(
    edge: declarations.EdgeDefinition,
    target: declarations.FeatureNodeDefinition
    | declarations.NodeDefinition[Any, Any]
    | declarations.PortDefinition,
    source: declarations.FeatureNodeDefinition
    | declarations.NodeDefinition[Any, Any]
    | declarations.PortDefinition,
    enter_id: ids.NodeId,
    /,
) -> None:
    if isinstance(target, declarations.PortDefinition):
        _require(
            edge.visit is None,
            f"edge {edge.id} targeting a port must not declare a visit",
        )
        return
    visit = edge.visit
    if isinstance(visit, declarations.CallVisitDefinition):
        _require(
            isinstance(target, declarations.NodeDefinition)
            and isinstance(
                target.operation,
                (declarations.SubroutineCall, declarations.WorkflowCall),
            ),
            (
                f"call visit {edge.id} may only target a subroutine "
                f"or workflow call node"
            ),
        )
        _require(
            callable(visit.implementation),
            f"call visit {edge.id} implementation must be callable",
        )
    elif isinstance(visit, declarations.VisitDefinition):
        _require(
            not (
                isinstance(target, declarations.NodeDefinition)
                and isinstance(
                    target.operation,
                    (declarations.SubroutineCall, declarations.WorkflowCall),
                )
            ),
            (
                f"edge {edge.id} targeting a call node must declare "
                f"CallVisitDefinition"
            ),
        )
        _require(
            callable(visit.implementation),
            f"visit {edge.id} implementation must be callable",
        )
    else:
        raise ValueError(
            f"edge {edge.id} targeting node {target.id} must declare a visit"
        )
    if isinstance(source, declarations.PortDefinition):
        _require(
            source.id == enter_id,
            f"edge {edge.id} has a terminal port source: {source.id}",
        )


def _graph_entities(
    graph: declarations.GraphDefinition[Any, Any, Any, Any], /
) -> dict[
    ids.NodeId,
    declarations.FeatureNodeDefinition
    | declarations.NodeDefinition[Any, Any]
    | declarations.PortDefinition,
]:
    entities: tuple[
        tuple[
            str,
            declarations.FeatureNodeDefinition
            | declarations.NodeDefinition[Any, Any]
            | declarations.PortDefinition,
        ],
        ...,
    ] = (
        ("enter port", graph.enter),
        ("exit port", graph.exit),
        ("failure port", graph.failure),
        *(("node", node) for node in graph.nodes),
    )
    found: dict[
        ids.NodeId,
        declarations.FeatureNodeDefinition
        | declarations.NodeDefinition[Any, Any]
        | declarations.PortDefinition,
    ] = {}
    for kind, entity in entities:
        if kind.endswith("port"):
            _require(
                isinstance(entity, declarations.PortDefinition),
                f"{kind} must be a PortDefinition",
            )
        _require(bool(entity.id), f"{kind} id must not be empty")
        _require(
            entity.id not in found, f"duplicate graph entity id: {entity.id}"
        )
        found[entity.id] = entity
        if isinstance(entity, declarations.PortDefinition):
            _require(
                kind.endswith("port"),
                f"graph nodes must not contain a PortDefinition: {entity.id}",
            )
            continue
        if isinstance(entity, declarations.FeatureNodeDefinition):
            operation = cast(object, entity.operation)
            _require(
                isinstance(operation, declarations.Feature),
                f"feature node {entity.id} must declare a Feature operation",
            )
            continue
        _validate_node_state_type(entity.state_type, entity.id)
        operation = entity.operation
        _require(
            not isinstance(operation, declarations.Feature),
            (
                f"node {entity.id} must use FeatureNodeDefinition for "
                f"a Feature operation"
            ),
        )
        if isinstance(
            operation, (declarations.SubroutineCall, declarations.WorkflowCall)
        ):
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
    graph: declarations.GraphDefinition[Any, Any, Any, Any], /
) -> dict[ids.FeatureId, declarations.FeatureDefinition[Any, Any]]:
    found: dict[ids.FeatureId, declarations.FeatureDefinition[Any, Any]] = {}
    for feature in graph.features:
        _require(bool(feature.id), "feature id must not be empty")
        _require(
            bool(feature.label), f"feature {feature.id} label must not be empty"
        )
        _require(feature.id not in found, f"duplicate feature id: {feature.id}")
        if feature.kind is declarations.FeatureKind.ENUM:
            _require(
                bool(feature.values)
                and all(
                    type(value) is str and bool(value)
                    for value in feature.values
                )
                and len(feature.values) == len(set(feature.values)),
                (
                    f"enum feature {feature.id} must have a non-empty "
                    f"unique string domain"
                ),
            )
        else:
            _require(
                not feature.values,
                f"non-enum feature {feature.id} must not declare enum values",
            )
        found[feature.id] = feature
    return found


def _validate_graph_edges(
    graph: declarations.GraphDefinition[Any, Any, Any, Any],
    entities: dict[
        ids.NodeId,
        declarations.FeatureNodeDefinition
        | declarations.NodeDefinition[Any, Any]
        | declarations.PortDefinition,
    ],
    features: dict[ids.FeatureId, declarations.FeatureDefinition[Any, Any]],
    /,
) -> None:
    edge_by_id: dict[ids.EdgeId, declarations.EdgeDefinition] = {}
    for edge in graph.edges:
        _require(bool(edge.id), "edge id must not be empty")
        _require(edge.id not in edge_by_id, f"duplicate edge id: {edge.id}")
        edge_by_id[edge.id] = edge
    for edge in graph.edges:
        _require(
            edge.source in entities,
            f"edge {edge.id} has unknown source: {edge.source}",
        )
        _require(
            edge.target in entities,
            f"edge {edge.id} has unknown target: {edge.target}",
        )
        _validate_edge_visit(
            edge,
            entities[edge.target],
            entities[edge.source],
            graph.enter.id,
        )
        if edge.effects:
            _require(
                isinstance(
                    entities[edge.source], declarations.FeatureNodeDefinition
                ),
                (
                    f"edge {edge.id} has effects but its source is "
                    f"not a feature node"
                ),
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


def _validate_port_routes(
    graph: declarations.GraphDefinition[Any, Any, Any, Any], /
) -> None:
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


def _validate_graph_resources(
    graph: declarations.GraphDefinition[Any, Any, Any, Any], /
) -> None:
    profiles: set[ids.AgentProfileId] = set()
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

    sessions: set[ids.AgentSessionId] = set()
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
        if not isinstance(node.operation, declarations.Agent):
            continue
        operation = node.operation
        _require(
            operation.profile in profiles,
            (
                f"agent node {node.id} references unknown profile: "
                f"{operation.profile}"
            ),
        )
        _require(
            operation.session in sessions,
            (
                f"agent node {node.id} references unknown session: "
                f"{operation.session}"
            ),
        )


def validate_graph(
    graph: declarations.GraphDefinition[Any, Any, Any, Any],
) -> None:
    """Validate graph structure, visits, features, and resource bindings.

    Boundary input and output values are checked during execution.
    """
    entities = _graph_entities(graph)
    features = _graph_features(graph)
    _validate_graph_edges(graph, entities, features)
    _validate_port_routes(graph)
    _validate_graph_resources(graph)
