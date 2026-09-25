"""Thread-safe cancellation and deadlines for workflow execution."""

from __future__ import annotations

import math
import threading
import time


class ExecutionCancelled(BaseException):
    """Execution stopped through cancellation or a deadline."""


class CancellationToken:
    """Thread-safe cancellation with an optional monotonic deadline."""

    def __init__(self, *, deadline: float | None = None) -> None:
        """Create a token with a deadline on ``time.monotonic()``."""
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("cancellation deadline must be finite")
        self._deadline = deadline
        self._cancelled = threading.Event()

    @classmethod
    def with_timeout(cls, timeout: float, /) -> CancellationToken:
        """Create a token that expires after the given non-negative seconds."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError(
                "cancellation timeout must be finite and non-negative"
            )
        return cls(deadline=time.monotonic() + timeout)

    @property
    def deadline(self) -> float | None:
        """The absolute time on the monotonic clock, if one was configured."""
        return self._deadline

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested or the deadline has elapsed."""
        return self._cancelled.is_set() or self.expired

    @property
    def expired(self) -> bool:
        """Whether the configured monotonic deadline has elapsed."""
        return self._deadline is not None and time.monotonic() >= self._deadline

    def cancel(self) -> None:
        """Request cancellation; blocked token waits are woken immediately."""
        self._cancelled.set()

    def remaining(self, maximum: float | None = None, /) -> float | None:
        """Bound a blocking wait by both its local maximum and this deadline."""
        if maximum is not None and (not math.isfinite(maximum) or maximum < 0):
            raise ValueError("wait maximum must be finite and non-negative")
        if self._deadline is None:
            return maximum
        remaining = max(0.0, self._deadline - time.monotonic())
        return remaining if maximum is None else min(remaining, maximum)

    def wait(self, timeout: float, /) -> bool:
        """Wait up to timeout seconds and report cancellation or expiry."""
        bounded = self.remaining(timeout)
        if bounded is None:
            raise ValueError("wait timeout must not be None")
        if self._cancelled.wait(bounded):
            return True
        return self.expired

    def raise_if_cancelled(self) -> None:
        """Raise ExecutionCancelled when cancellation or expiry is observed."""
        if self._cancelled.is_set():
            raise ExecutionCancelled("execution was cancelled")
        if self.expired:
            raise ExecutionCancelled("execution deadline was exceeded")
