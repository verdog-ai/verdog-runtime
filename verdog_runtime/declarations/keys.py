"""Typed keys for immutable workflow state."""

from __future__ import annotations

from typing import Generic, TypeAlias, TypeVar


StateT = TypeVar("StateT")
ScopeT = TypeVar("ScopeT")
StateAddress: TypeAlias = tuple[str, str]


class StateKey(Generic[StateT, ScopeT]):
    """A declaration whose value occupies one workflow-state slot."""

    __slots__ = ()

    @property
    def state_key(self) -> StateAddress:
        raise NotImplementedError
