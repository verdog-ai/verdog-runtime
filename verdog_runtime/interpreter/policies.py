"""Public execution policies for durable workflow operations."""

from enum import StrEnum


class SessionPolicy(StrEnum):
    """How a new run treats persistent provider conversations."""

    BRANCH = "branch"
    FRESH = "fresh"


__all__ = ["SessionPolicy"]
