"""Immutable per-run workflow state and execution boundary values."""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping
from typing import Any, Generic, TypeVar, cast, override

from verdog_runtime.declarations import graph as graph_declarations
from verdog_runtime.declarations import keys

StateT = TypeVar("StateT")
ScopeT = TypeVar("ScopeT")
FeatureValueT = TypeVar("FeatureValueT", bound=bool | int | float | str)


def _require_feature(key: object, /) -> None:
    if not isinstance(key, graph_declarations.FeatureDefinition):
        raise TypeError("FeatureState.replace key must be FeatureDefinition")


class WorkflowState(Generic[ScopeT]):
    """An immutable, structurally shared view of every entity's state."""

    __slots__ = ("_states", "_universe")
    _states: Mapping[int, tuple[keys.StateKey[Any, ScopeT], object]]
    _universe: object

    def __init__(self) -> None:
        """Create an empty universe with no declaration keys."""
        object.__setattr__(
            self,
            "_states",
            types.MappingProxyType({}),
        )
        object.__setattr__(self, "_universe", object())

    @override
    def __setattr__(self, name: str, value: object, /) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    @classmethod
    def _from_states(
        cls,
        states: Mapping[int, tuple[keys.StateKey[Any, ScopeT], object]],
        universe: object,
        /,
    ) -> WorkflowState[ScopeT]:
        value = cls()
        object.__setattr__(
            value, "_states", types.MappingProxyType(dict(states))
        )
        object.__setattr__(value, "_universe", universe)
        return value

    @classmethod
    def _initial(
        cls,
        values: Iterable[tuple[keys.StateKey[Any, ScopeT], object]],
        /,
    ) -> WorkflowState[ScopeT]:
        states: dict[int, tuple[keys.StateKey[Any, ScopeT], object]] = {}
        for owner, value in values:
            key = id(owner)
            if key in states:
                raise ValueError(
                    f"duplicate workflow state owner: {owner.state_key}"
                )
            states[key] = owner, value
        return cls._from_states(states, object())

    def get(self, key: keys.StateKey[StateT, ScopeT], /) -> StateT:
        """Return value for this exact declaration key, or raise KeyError."""
        try:
            stored_key, value = self._states[id(key)]
            if stored_key is not key:
                raise KeyError
            return cast(StateT, value)
        except KeyError as error:
            raise KeyError(
                f"unknown workflow state key: {key.state_key}"
            ) from error

    def _replace(
        self,
        key: keys.StateKey[StateT, ScopeT],
        value: StateT,
        /,
    ) -> WorkflowState[ScopeT]:
        identity = id(key)
        current = self._states.get(identity)
        if current is None or current[0] is not key:
            raise KeyError(f"unknown workflow state key: {key.state_key}")
        if current[1] is value:
            return self
        states = dict(self._states)
        states[identity] = key, value
        return WorkflowState[ScopeT]._from_states(states, self._universe)

    def _changes(
        self, other: WorkflowState[ScopeT], /
    ) -> tuple[keys.StateKey[Any, ScopeT], ...]:
        """Return changed keys, rejecting foreign state universes."""
        if self._universe is not other._universe:
            raise ValueError("workflow state universe changed")
        if set(self._states) != set(other._states):
            raise ValueError("workflow state keys changed")
        changed: list[keys.StateKey[Any, ScopeT]] = []
        for identity, (key, value) in self._states.items():
            other_key, other_value = other._states[identity]
            if other_key is not key:
                raise ValueError("workflow state keys changed")
            if not bool(value == other_value):
                changed.append(key)
        return tuple(changed)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkflowState):
            return False
        typed = cast(WorkflowState[Any], other)
        if set(self._states) != set(typed._states):
            return False
        for key, (owner, value) in self._states.items():
            other_owner, other_value = typed._states[key]
            if owner is not other_owner or not bool(value == other_value):
                return False
        return True

    @override
    def __repr__(self) -> str:
        values = {key.state_key: value for key, value in self._states.values()}
        return f"WorkflowState({values!r})"


class FeatureState(Generic[ScopeT]):
    """A state view that reads every slot and replaces only features."""

    __slots__ = ("_state",)
    _state: WorkflowState[ScopeT]

    def __init__(self) -> None:
        """Create an empty feature view with no registered declaration keys."""
        object.__setattr__(self, "_state", WorkflowState[ScopeT]())

    @override
    def __setattr__(self, name: str, value: object, /) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    @classmethod
    def _from_workflow_state(
        cls, state: WorkflowState[ScopeT], /
    ) -> FeatureState[ScopeT]:
        value = cls()
        object.__setattr__(value, "_state", state)
        return value

    def _as_workflow_state(self, /) -> WorkflowState[ScopeT]:
        return self._state

    def get(self, key: keys.StateKey[StateT, ScopeT], /) -> StateT:
        """Read a node or feature value from the underlying workflow state."""
        return self._state.get(key)

    def replace(
        self,
        key: graph_declarations.FeatureDefinition[FeatureValueT, ScopeT],
        value: FeatureValueT,
        /,
    ) -> FeatureState[ScopeT]:
        """Return a new view with one feature replaced, preserving original."""
        _require_feature(key)
        return FeatureState[ScopeT]._from_workflow_state(
            self._state._replace(  # pyright: ignore[reportPrivateUsage]
                key, value
            )
        )

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FeatureState)
            and self._state == cast(FeatureState[Any], other)._state
        )

    @override
    def __repr__(self) -> str:
        return f"FeatureState({self._state!r})"
