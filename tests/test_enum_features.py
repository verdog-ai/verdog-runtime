from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from verdog_runtime.declarations import (
    AgentSessionDefinition,
    AgentSessionParameter,
    EdgeDefinition,
    EnumConditionObservation,
    EnumEffectObservation,
    EnumFeatureCondition,
    EnumFeatureEffect,
    Feature,
    FeatureNodeDefinition,
    FeatureState,
    FeatureSuccess,
    GraphDefinition,
    PortDefinition,
    VisitDefinition,
)
from verdog_runtime.declarations.graph import (
    FeatureCondition,
    FeatureDefinition,
    FeatureEffect,
    FeatureKind,
)
from verdog_runtime.declarations.ids import (
    AgentSessionId,
    EdgeId,
    FeatureId,
    GraphId,
    NodeId,
)
from verdog_runtime.interpreter import (
    analyze_effects,
    effects_satisfied,
    evaluate_conditions,
    initial_workflow_state,
    validate_feature_value,
    validate_graph,
)


def _feature(
    values: tuple[str, ...] = ("retry", "done"),
) -> FeatureDefinition[str, object]:
    return FeatureDefinition(
        id=FeatureId("route"),
        label="Route",
        description="The selected route",
        kind=FeatureKind.ENUM,
        values=values,
    )


def _pass_feature(
    input: object, state: FeatureState[object], context: object, /
) -> FeatureSuccess[object]:
    return FeatureSuccess(state=state)


def _graph(
    feature: FeatureDefinition[Any, Any],
    *,
    conditions: tuple[FeatureCondition, ...] = (),
    effects: tuple[FeatureEffect, ...] = (),
) -> GraphDefinition[object, object, None, object]:
    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    update = FeatureNodeDefinition(
        id=NodeId("update"),
        name="Update feature",
        operation=Feature(),
    )
    return GraphDefinition(
        id=GraphId("enum_graph"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(update,),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter_update"),
                source=enter.id,
                target=update.id,
                visit=VisitDefinition(
                    implementation=_pass_feature,
                ),
            ),
            EdgeDefinition(
                id=EdgeId("update_exit"),
                source=update.id,
                target=exit_.id,
                conditions=conditions,
                effects=effects,
            ),
        ),
        features=(feature,),
    )


def test_enum_feature_equality_unconstrained_and_implicit_unchanged() -> None:
    feature = _feature()
    graph = _graph(feature)
    initial = initial_workflow_state(graph)
    assert initial.get(feature) is None
    assert effects_satisfied(
        analyze_effects({feature.id: feature}, ()).inferred,
        initial,
        initial,
        {feature.id: feature},
    )
    with pytest.raises(ValueError, match="feature is uninitialized"):
        evaluate_conditions(
            (
                EnumFeatureCondition(
                    feature_id=feature.id,
                    observation=EnumConditionObservation.EQUAL,
                    value="retry",
                ),
            ),
            initial,
            {feature.id: feature},
        )
    source = initial._replace(feature, "retry")  # pyright: ignore[reportPrivateUsage]
    successor = source._replace(  # pyright: ignore[reportPrivateUsage]
        feature, "done"
    )
    features = {feature.id: feature}

    assert evaluate_conditions(
        (
            EnumFeatureCondition(
                feature_id=feature.id,
                observation=EnumConditionObservation.EQUAL,
                value="retry",
            ),
        ),
        source,
        features,
    )
    assert not evaluate_conditions(
        (
            EnumFeatureCondition(
                feature_id=feature.id,
                observation=EnumConditionObservation.EQUAL,
                value="done",
            ),
        ),
        source,
        features,
    )
    assert effects_satisfied(
        (
            EnumFeatureEffect(
                feature_id=feature.id,
                observation=EnumEffectObservation.EQUAL,
                value="done",
            ),
        ),
        source,
        successor,
        features,
    )
    assert effects_satisfied(
        (
            EnumFeatureEffect(
                feature_id=feature.id,
                observation=EnumEffectObservation.UNCONSTRAINED,
            ),
        ),
        source,
        successor,
        features,
    )
    with pytest.raises(ValueError, match="must be one of"):
        effects_satisfied(
            (
                EnumFeatureEffect(
                    feature_id=feature.id,
                    observation=EnumEffectObservation.UNCONSTRAINED,
                ),
            ),
            source,
            source._replace(  # pyright: ignore[reportPrivateUsage]
                feature, "missing"
            ),
            features,
        )

    inferred = analyze_effects(features, ()).inferred
    assert inferred == (
        EnumFeatureEffect(
            feature_id=feature.id,
            observation=EnumEffectObservation.UNCHANGED,
        ),
    )
    assert effects_satisfied(inferred, source, source, features)
    assert not effects_satisfied(inferred, source, successor, features)

    assert validate_feature_value(feature, "retry") == "retry"
    with pytest.raises(ValueError, match="must be one of"):
        validate_feature_value(feature, "missing")


def test_graph_controls_must_be_port_definitions() -> None:
    graph = _graph(_feature())
    not_a_port = cast(PortDefinition, graph.nodes[0])

    for invalid in (
        replace(graph, enter=not_a_port),
        replace(graph, exit=not_a_port),
        replace(graph, failure=not_a_port),
    ):
        with pytest.raises(ValueError, match="port must be a PortDefinition"):
            validate_graph(invalid)


def test_graph_nodes_must_not_contain_ports() -> None:
    graph = _graph(_feature())
    extra = PortDefinition(id=NodeId("extra"))

    with pytest.raises(ValueError, match="must not contain a PortDefinition"):
        validate_graph(replace(graph, nodes=cast(Any, (extra,))))


def test_enter_port_must_not_have_an_incoming_edge() -> None:
    graph = _graph(_feature())
    update = graph.nodes[0]
    incoming = EdgeDefinition(
        id=EdgeId("update_enter"),
        source=update.id,
        target=graph.enter.id,
    )

    with pytest.raises(ValueError, match="enter port must not have an incoming edge"):
        validate_graph(replace(graph, edges=(*graph.edges, incoming)))


def test_feature_payload_annotations_are_static_only() -> None:
    graph = _graph(_feature())
    update = graph.nodes[0]
    loop = EdgeDefinition(
        id=EdgeId("update_again"),
        source=update.id,
        target=update.id,
        visit=VisitDefinition(
            implementation=_pass_feature,
        ),
    )

    validate_graph(replace(graph, edges=(*graph.edges, loop)))


def test_enum_feature_declarations_validate_domains_and_observations() -> None:
    feature = _feature()
    valid = _graph(
        feature,
        conditions=(
            EnumFeatureCondition(
                feature_id=feature.id,
                observation=EnumConditionObservation.EQUAL,
                value="retry",
            ),
        ),
        effects=(
            EnumFeatureEffect(
                feature_id=feature.id,
                observation=EnumEffectObservation.EQUAL,
                value="retry",
            ),
        ),
    )
    validate_graph(valid)

    for values in (
        (),
        ("",),
        ("retry", "retry"),
        cast(tuple[str, ...], ("retry", 1)),
    ):
        with pytest.raises(ValueError, match="non-empty unique string domain"):
            validate_graph(_graph(_feature(values)))

    integer = FeatureDefinition[int, object](
        id=FeatureId("count"),
        label="Count",
        description="A count",
        kind=FeatureKind.INTEGER,
        values=("not", "an", "enum"),
    )
    with pytest.raises(ValueError, match="must not declare enum values"):
        validate_graph(_graph(integer))

    invalid_references = (
        _graph(
            feature,
            conditions=(
                EnumFeatureCondition(
                    feature_id=feature.id,
                    observation=EnumConditionObservation.EQUAL,
                    value="missing",
                ),
            ),
        ),
        _graph(
            feature,
            effects=(
                EnumFeatureEffect(
                    feature_id=feature.id,
                    observation=EnumEffectObservation.EQUAL,
                ),
            ),
        ),
        _graph(
            feature,
            effects=(
                EnumFeatureEffect(
                    feature_id=feature.id,
                    observation=EnumEffectObservation.UNCONSTRAINED,
                    value="retry",
                ),
            ),
        ),
        _graph(
            feature,
            effects=(
                EnumFeatureEffect(
                    feature_id=feature.id,
                    observation=cast(EnumEffectObservation, "unconstrained"),
                ),
            ),
        ),
    )
    for graph in invalid_references:
        with pytest.raises(ValueError, match="enum feature"):
            validate_graph(graph)


@pytest.mark.parametrize(
    ("persistent", "parameter_name", "message"),
    [
        (None, "", "persistent must be boolean"),
        (True, "", "name must not be empty"),
        (True, "shared", "duplicate agent session id"),
    ],
)
def test_session_definitions_are_validated_before_parameters(
    persistent: object, parameter_name: str, message: str
) -> None:
    session = AgentSessionDefinition(
        id=AgentSessionId("shared"), name="shared", persistent=True
    )
    object.__setattr__(session, "persistent", persistent)
    graph = replace(
        _graph(_feature()),
        sessions=(session,),
        session_parameters=(
            AgentSessionParameter(id=session.id, name=parameter_name),
        ),
    )
    with pytest.raises(ValueError, match=message):
        validate_graph(graph)
