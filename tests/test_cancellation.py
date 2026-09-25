from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from verdog_runtime import CancellationToken, ExecutionCancelled


@dataclass(slots=True)
class _Clock:
    value: float

    def monotonic(self) -> float:
        return self.value


def test_explicit_cancellation_is_sticky_and_wakes_waiters() -> None:
    token = CancellationToken()

    assert not token.cancelled
    assert not token.expired
    assert token.remaining() is None
    assert not token.wait(0.0)

    token.cancel()

    assert token.cancelled
    assert not token.expired
    assert token.wait(60.0)
    with pytest.raises(ExecutionCancelled, match="execution was cancelled"):
        token.raise_if_cancelled()


def test_timeout_uses_an_absolute_monotonic_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(100.0)
    monkeypatch.setattr("verdog_runtime.cancellation.time", clock)

    token = CancellationToken.with_timeout(2.0)

    assert token.deadline == 102.0
    assert token.remaining() == 2.0
    assert token.remaining(0.25) == 0.25
    assert not token.cancelled

    clock.value = 101.0
    assert token.remaining() == 1.0
    assert not token.expired

    clock.value = 102.0
    assert token.remaining() == 0.0
    assert token.expired
    assert token.cancelled
    assert token.wait(60.0)
    with pytest.raises(ExecutionCancelled, match="deadline was exceeded"):
        token.raise_if_cancelled()


@pytest.mark.parametrize("timeout", (-1.0, float("inf"), float("nan")))
def test_timeout_rejects_negative_and_non_finite_values(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        CancellationToken.with_timeout(timeout)


def test_wait_rejects_an_unbounded_timeout() -> None:
    token = CancellationToken()
    with pytest.raises(ValueError, match="wait timeout must not be None"):
        token.wait(cast(float, None))
