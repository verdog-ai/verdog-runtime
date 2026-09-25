"""Evaluate qualitative feature semantics over workflow-state snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, TypeVar, cast

from ..declarations.graph import (
    BooleanConditionObservation,
    BooleanEffectObservation,
    BooleanFeatureCondition,
    BooleanFeatureEffect,
    EnumConditionObservation,
    EnumEffectObservation,
    EnumFeatureCondition,
    EnumFeatureEffect,
    FeatureCondition,
    FeatureDefinition,
    FeatureEffect,
    FeatureKind,
    FeatureValue,
    NumericalConditionObservation,
    NumericalEffectObservation,
    NumericalFeatureEffect,
)
from ..declarations.ids import FeatureId
from ..declarations.state import WorkflowState


FeatureValueT = TypeVar("FeatureValueT", bound=FeatureValue)
ScopeT = TypeVar("ScopeT")


@dataclass(frozen=True, slots=True)
class EffectAnalysis:
    inferred: tuple[FeatureEffect, ...]


def validate_feature_value(
    feature: FeatureDefinition[FeatureValueT, Any],
    value: object,
    /,
) -> FeatureValueT:
    if feature.kind is FeatureKind.BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError(f"boolean feature {feature.id} must be bool")
    elif feature.kind is FeatureKind.ENUM:
        if type(value) is not str or value not in feature.values:
            raise ValueError(
                f"enum feature {feature.id} must be one of {feature.values!r}"
            )
    elif feature.kind is FeatureKind.INTEGER:
        if type(value) is not int or value < 0:
            raise ValueError(f"integer feature {feature.id} must be a non-negative int")
    elif type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError(
            f"float feature {feature.id} must be a finite non-negative float"
        )
    return cast(FeatureValueT, value)


def _raw(
    feature: FeatureDefinition[FeatureValueT, ScopeT],
    state: WorkflowState[ScopeT],
    /,
) -> FeatureValueT | None:
    value = state.get(feature)
    return None if value is None else validate_feature_value(feature, value)


def _initialized(
    feature: FeatureDefinition[FeatureValueT, ScopeT],
    state: WorkflowState[ScopeT],
    /,
) -> FeatureValueT:
    value = _raw(feature, state)
    if value is None:
        raise ValueError(f"feature is uninitialized: {feature.id}")
    return value


def evaluate_conditions(
    conditions: Sequence[FeatureCondition],
    state: WorkflowState[ScopeT],
    features: Mapping[FeatureId, FeatureDefinition[Any, ScopeT]],
    /,
) -> bool:
    """Return the conjunction of an edge's source feature conditions."""

    for condition in conditions:
        feature = features[condition.feature_id]
        value = _initialized(feature, state)
        if isinstance(condition, EnumFeatureCondition):
            if condition.observation is not EnumConditionObservation.EQUAL:
                raise ValueError(f"unknown enum condition: {condition.observation}")
            if value != condition.value:
                return False
        elif isinstance(condition, BooleanFeatureCondition):
            expected = condition.observation is BooleanConditionObservation.POSITIVE
            if value is not expected:
                return False
        elif condition.observation is NumericalConditionObservation.EQUAL_ZERO:
            if value != 0:
                return False
        elif cast(int | float, value) <= 0:
            return False
    return True


def analyze_effects(
    features: Mapping[FeatureId, FeatureDefinition[Any, ScopeT]],
    explicit: Sequence[FeatureEffect],
    /,
) -> EffectAnalysis:
    """Complete an edge's effect frame with UNCHANGED observations."""

    explicit_ids = {effect.feature_id for effect in explicit}
    inferred: list[FeatureEffect] = []
    for feature_id, feature in features.items():
        if feature_id in explicit_ids:
            continue
        if feature.kind is FeatureKind.BOOLEAN:
            inferred.append(
                BooleanFeatureEffect(
                    feature_id=feature.id,
                    observation=BooleanEffectObservation.UNCHANGED,
                )
            )
        elif feature.kind is FeatureKind.ENUM:
            inferred.append(
                EnumFeatureEffect(
                    feature_id=feature.id,
                    observation=EnumEffectObservation.UNCHANGED,
                )
            )
        else:
            inferred.append(
                NumericalFeatureEffect(
                    feature_id=feature.id,
                    observation=NumericalEffectObservation.UNCHANGED,
                )
            )
    return EffectAnalysis(inferred=tuple(inferred))


def effects_satisfied(
    effects: Sequence[FeatureEffect],
    source: WorkflowState[ScopeT],
    successor: WorkflowState[ScopeT],
    features: Mapping[FeatureId, FeatureDefinition[Any, ScopeT]],
    /,
) -> bool:
    for effect in effects:
        feature = features[effect.feature_id]
        if isinstance(effect, EnumFeatureEffect):
            if effect.observation is EnumEffectObservation.UNCONSTRAINED:
                _initialized(feature, successor)
                continue
            if effect.observation is EnumEffectObservation.UNCHANGED:
                if _raw(feature, successor) != _raw(feature, source):
                    return False
                continue
            if effect.observation is EnumEffectObservation.EQUAL:
                if _initialized(feature, successor) != effect.value:
                    return False
                continue
            raise ValueError(f"unknown enum effect: {effect.observation}")
        if isinstance(effect, BooleanFeatureEffect):
            if effect.observation is BooleanEffectObservation.UNCONSTRAINED:
                _initialized(feature, successor)
                continue
            if effect.observation is BooleanEffectObservation.UNCHANGED:
                if _raw(feature, successor) != _raw(feature, source):
                    return False
                continue
            target = _initialized(feature, successor)
            expected = effect.observation is BooleanEffectObservation.POSITIVE
            if target is not expected:
                return False
            continue

        if effect.observation is NumericalEffectObservation.UNCONSTRAINED:
            _initialized(feature, successor)
            continue
        if effect.observation is NumericalEffectObservation.UNCHANGED:
            if _raw(feature, successor) != _raw(feature, source):
                return False
            continue
        before = cast(int | float, _initialized(feature, source))
        after = cast(int | float, _initialized(feature, successor))
        if effect.observation is NumericalEffectObservation.INCREASES:
            if after <= before:
                return False
        elif after >= before:
            return False
    return True
