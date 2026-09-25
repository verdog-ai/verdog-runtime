"""Public execution policies for durable workflow operations."""

import enum


class SessionPolicy(enum.StrEnum):
    """How a new run treats persistent provider conversations."""

    BRANCH = "branch"
    FRESH = "fresh"


__all__ = ["SessionPolicy"]
