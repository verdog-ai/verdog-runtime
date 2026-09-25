from __future__ import annotations

import math
import time
from threading import Event


class ExecutionCancelled(BaseException):
    """Execution stopped through an explicit cancellation request or deadline."""


class CancellationToken:
    """Thread-safe cooperative cancellation with an optional monotonic deadline."""

    def __init__(self, *, deadline: float | None = None) -> None:
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("cancellation deadline must be finite")
        self._deadline = deadline
        self._cancelled = Event()

    @classmethod
    def with_timeout(cls, timeout: float, /) -> CancellationToken:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("cancellation timeout must be finite and non-negative")
        return cls(deadline=time.monotonic() + timeout)

    @property
    def deadline(self) -> float | None:
        """The absolute time on the monotonic clock, if one was configured."""

        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set() or self.expired

    @property
    def expired(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def cancel(self) -> None:
        self._cancelled.set()

    def remaining(self, maximum: float | None = None, /) -> float | None:
        """Bound a blocking wait by both its local maximum and this deadline."""

        if maximum is not None and (
            not math.isfinite(maximum) or maximum < 0
        ):
            raise ValueError("wait maximum must be finite and non-negative")
        if self._deadline is None:
            return maximum
        remaining = max(0.0, self._deadline - time.monotonic())
        return remaining if maximum is None else min(remaining, maximum)

    def wait(self, timeout: float, /) -> bool:
        """Block for at most timeout seconds and report cancellation or expiry."""

        bounded = self.remaining(timeout)
        assert bounded is not None
        if self._cancelled.wait(bounded):
            return True
        return self.expired

    def raise_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise ExecutionCancelled("execution was cancelled")
        if self.expired:
            raise ExecutionCancelled("execution deadline was exceeded")
