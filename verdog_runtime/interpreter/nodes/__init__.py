"""Shared callback types for the per-operation node handlers."""

from collections.abc import Callable
from typing import TypeAlias, TypeVar

InputT = TypeVar("InputT")
StateT = TypeVar("StateT")
ContextT = TypeVar("ContextT")
ResultT = TypeVar("ResultT")

Visit: TypeAlias = Callable[[InputT, StateT, ContextT], ResultT]
