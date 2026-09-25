"""Evaluate qualitative feature semantics over workflow-state snapshots."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

from verdog_runtime.declarations import graph as graph_declarations
from verdog_runtime.declarations import ids
from verdog_runtime.declarations import state as state_declarations

FeatureValueT = TypeVar("FeatureValueT", bound=graph_declarations.FeatureValue)
ScopeT = TypeVar("ScopeT")


@dataclasses.dataclass(frozen=True, slots=True)
class EffectAnalysis:
    """Frame effects inferred for features omitted from explicit effects."""

    inferred: tuple[graph_declarations.FeatureEffect, ...]


def validate_feature_value(
    feature: graph_declarations.FeatureDefinition[FeatureValueT, Any],
    value: object,
    /,
) -> FeatureValueT:
    """Validate a feature value and return it with the declared value type.

    Raises:
        ValueError: If the value is outside the feature domain.
    """
    if feature.kind is graph_declarations.FeatureKind.BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError(f"boolean feature {feature.id} must be bool")
    elif feature.kind is graph_declarations.FeatureKind.ENUM:
        if type(value) is not str or value not in feature.values:
            raise ValueError(
                f"enum feature {feature.id} must be one of {feature.values!r}"
            )
    elif feature.kind is graph_declarations.FeatureKind.INTEGER:
        if type(value) is not int or value < 0:
            raise ValueError(
                f"integer feature {feature.id} must be a non-negative int"
            )
    elif type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError(
            f"float feature {feature.id} must be a finite non-negative float"
        )
    return cast(FeatureValueT, value)


def _raw(
    feature: graph_declarations.FeatureDefinition[FeatureValueT, ScopeT],
    state: state_declarations.WorkflowState[ScopeT],
    /,
) -> FeatureValueT | None:
    value = state.get(feature)
    return None if value is None else validate_feature_value(feature, value)


def _initialized(
    feature: graph_declarations.FeatureDefinition[FeatureValueT, ScopeT],
    state: state_declarations.WorkflowState[ScopeT],
    /,
) -> FeatureValueT:
    value = _raw(feature, state)
    if value is None:
        raise ValueError(f"feature is uninitialized: {feature.id}")
    return value


def evaluate_conditions(
    conditions: Sequence[graph_declarations.FeatureCondition],
    state: state_declarations.WorkflowState[ScopeT],
    features: Mapping[
        ids.FeatureId, graph_declarations.FeatureDefinition[Any, ScopeT]
    ],
    /,
) -> bool:
    """Return the conjunction of an edge's source feature conditions."""
    for condition in conditions:
        feature = features[condition.feature_id]
        value = _initialized(feature, state)
        if isinstance(condition, graph_declarations.EnumFeatureCondition):
            if (
                condition.observation
                is not graph_declarations.EnumConditionObservation.EQUAL
            ):
                raise ValueError(
                    f"unknown enum condition: {condition.observation}"
                )
            if value != condition.value:
                return False
        elif isinstance(condition, graph_declarations.BooleanFeatureCondition):
            expected = (
                condition.observation
                is graph_declarations.BooleanConditionObservation.POSITIVE
            )
            if value is not expected:
                return False
        elif (
            condition.observation
            is graph_declarations.NumericalConditionObservation.EQUAL_ZERO
        ):
            if value != 0:
                return False
        elif cast(int | float, value) <= 0:
            return False
    return True


def analyze_effects(
    features: Mapping[
        ids.FeatureId, graph_declarations.FeatureDefinition[Any, ScopeT]
    ],
    explicit: Sequence[graph_declarations.FeatureEffect],
    /,
) -> EffectAnalysis:
    """Complete an edge's effect frame with UNCHANGED observations."""
    explicit_ids = {effect.feature_id for effect in explicit}
    inferred: list[graph_declarations.FeatureEffect] = []
    for feature_id, feature in features.items():
        if feature_id in explicit_ids:
            continue
        if feature.kind is graph_declarations.FeatureKind.BOOLEAN:
            inferred.append(
                graph_declarations.BooleanFeatureEffect(
                    feature_id=feature.id,
                    observation=graph_declarations.BooleanEffectObservation.UNCHANGED,
                )
            )
        elif feature.kind is graph_declarations.FeatureKind.ENUM:
            inferred.append(
                graph_declarations.EnumFeatureEffect(
                    feature_id=feature.id,
                    observation=graph_declarations.EnumEffectObservation.UNCHANGED,
                )
            )
        else:
            inferred.append(
                graph_declarations.NumericalFeatureEffect(
                    feature_id=feature.id,
                    observation=graph_declarations.NumericalEffectObservation.UNCHANGED,
                )
            )
    return EffectAnalysis(inferred=tuple(inferred))


def effects_satisfied(
    effects: Sequence[graph_declarations.FeatureEffect],
    source: state_declarations.WorkflowState[ScopeT],
    successor: state_declarations.WorkflowState[ScopeT],
    features: Mapping[
        ids.FeatureId, graph_declarations.FeatureDefinition[Any, ScopeT]
    ],
    /,
) -> bool:
    """Return whether every effect holds across the source and successor."""
    for effect in effects:
        feature = features[effect.feature_id]
        if isinstance(effect, graph_declarations.EnumFeatureEffect):
            if (
                effect.observation
                is graph_declarations.EnumEffectObservation.UNCONSTRAINED
            ):
                _initialized(feature, successor)
                continue
            if (
                effect.observation
                is graph_declarations.EnumEffectObservation.UNCHANGED
            ):
                if _raw(feature, successor) != _raw(feature, source):
                    return False
                continue
            if (
                effect.observation
                is graph_declarations.EnumEffectObservation.EQUAL
            ):
                if _initialized(feature, successor) != effect.value:
                    return False
                continue
            raise ValueError(f"unknown enum effect: {effect.observation}")
        if isinstance(effect, graph_declarations.BooleanFeatureEffect):
            if (
                effect.observation
                is graph_declarations.BooleanEffectObservation.UNCONSTRAINED
            ):
                _initialized(feature, successor)
                continue
            if (
                effect.observation
                is graph_declarations.BooleanEffectObservation.UNCHANGED
            ):
                if _raw(feature, successor) != _raw(feature, source):
                    return False
                continue
            target = _initialized(feature, successor)
            expected = (
                effect.observation
                is graph_declarations.BooleanEffectObservation.POSITIVE
            )
            if target is not expected:
                return False
            continue

        if (
            effect.observation
            is graph_declarations.NumericalEffectObservation.UNCONSTRAINED
        ):
            _initialized(feature, successor)
            continue
        if (
            effect.observation
            is graph_declarations.NumericalEffectObservation.UNCHANGED
        ):
            if _raw(feature, successor) != _raw(feature, source):
                return False
            continue
        before = cast(int | float, _initialized(feature, source))
        after = cast(int | float, _initialized(feature, successor))
        if (
            effect.observation
            is graph_declarations.NumericalEffectObservation.INCREASES
        ):
            if after <= before:
                return False
        elif after >= before:
            return False
    return True
