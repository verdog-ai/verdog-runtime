"""Successful results crossing execution boundaries."""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, override

from .state import FeatureState


ResultOutputT = TypeVar("ResultOutputT", covariant=True)
ResultStateT = TypeVar("ResultStateT", covariant=True)
ScopeT = TypeVar("ScopeT")


@dataclass(frozen=True, slots=True, kw_only=True)
class Success(Generic[ResultOutputT, ResultStateT]):
    output: ResultOutputT
    state: ResultStateT


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureSuccess(Generic[ScopeT]):
    state: FeatureState[ScopeT]


class RemoteWorkflowError(RuntimeError):
    """An exception reported by an isolated workflow process."""

    def __init__(
        self,
        exception_type: str,
        message: str,
        remote_traceback: str,
        /,
    ) -> None:
        super().__init__(f"{exception_type}: {message}")
        self.exception_type = exception_type
        self.message = message
        self.remote_traceback = remote_traceback
        self.add_note("Remote traceback:\n" + remote_traceback)

    @override
    def __reduce__(self) -> str | tuple[Any, ...]:
        return (
            _restore_remote_workflow_error,
            (
                self.exception_type,
                self.message,
                self.remote_traceback,
                tuple(getattr(self, "__notes__", ())),
            ),
        )


def _restore_remote_workflow_error(
    exception_type: str,
    message: str,
    remote_traceback: str,
    notes: tuple[str, ...],
    /,
) -> RemoteWorkflowError:
    error = RemoteWorkflowError(exception_type, message, remote_traceback)
    error.__notes__ = list(notes)
    return error
