"""Durable records around external agent invocations.

An interpreter checkpoint is taken at every graph boundary.  The transition
budget at that boundary is therefore a stable, run-local invocation epoch: it
is restored with the continuation, and it cannot repeat during an ordinary
run.  Combining that epoch with the node address and the invocation's slot in
the node lets a resumed visit find the request it was executing before the
process stopped without depending on attempt-specific artifact directories.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from .._run_metadata import atomic_json
from ..declarations.agents import AgentReply, AgentRequest
from ..declarations.ids import ProviderSessionId

_SCHEMA_VERSION = 1
_DIRECTORY = "invocations"
_NODE_OUTPUT_TOKEN = "${VERDOG_NODE_OUTPUT}"


class InvocationJournalError(RuntimeError):
    """A stable error raised while recovering an external invocation."""

    def __init__(self, message: str, *, code: str, details: object = None) -> None:
        super().__init__(f"{message} [{code}]")
        self.code = code
        self.details = details


@dataclass(frozen=True, slots=True)
class InvocationAddress:
    run_id: str
    transition_epoch: int
    graph_id: str
    node_id: str
    edge_id: str
    node_path: str
    slot: int

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "transition_epoch": self.transition_epoch,
            "graph_id": self.graph_id,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "node_path": self.node_path,
            "slot": self.slot,
        }


@dataclass(frozen=True, slots=True)
class InvocationRequestRecord:
    prompt_sha256: str
    profile_id: str
    session_id: str
    persistent: bool
    provider: str
    provider_session_id: str | None
    provider_session_action: str
    workspace: str
    access: str

    @classmethod
    def from_request(
        cls, request: AgentRequest, provider: str, /
    ) -> InvocationRequestRecord:
        node_output = request.node_context.output_dir.resolve()
        prompt = request.prompt.replace(str(node_output), _NODE_OUTPUT_TOKEN)
        workspace = request.workspace.resolve()
        try:
            workspace_relative = workspace.relative_to(node_output)
        except ValueError:
            workspace_record = str(workspace)
        else:
            workspace_record = _NODE_OUTPUT_TOKEN
            if workspace_relative.parts:
                workspace_record += "/" + workspace_relative.as_posix()
        return cls(
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            profile_id=str(request.profile_id),
            session_id=str(request.session_id),
            persistent=request.persistent,
            provider=provider,
            provider_session_id=(
                None
                if request.provider_session_id is None
                else str(request.provider_session_id)
            ),
            provider_session_action=request.provider_session_action.value,
            workspace=workspace_record,
            access=request.access.value,
        )

    def as_json(self) -> dict[str, object]:
        return {
            "prompt_sha256": self.prompt_sha256,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "persistent": self.persistent,
            "provider": self.provider,
            "provider_session_id": self.provider_session_id,
            "provider_session_action": self.provider_session_action,
            "workspace": self.workspace,
            "access": self.access,
        }


@dataclass(frozen=True, slots=True)
class JournaledReply:
    reply: AgentReply
    replayed: bool


def _object(path: Path) -> dict[str, object]:
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise OSError("not a regular file")
        value: object = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise InvocationJournalError(
            f"agent invocation journal record is unreadable: {path}",
            code="invocation.journal_unreadable",
            details={"path": str(path)},
        ) from error
    if not isinstance(value, dict):
        raise InvocationJournalError(
            f"agent invocation journal record is not an object: {path}",
            code="invocation.journal_invalid",
            details={"path": str(path)},
        )
    record = cast(dict[str, object], value)
    if record.get("schema_version") != _SCHEMA_VERSION:
        raise InvocationJournalError(
            f"agent invocation journal record has an unsupported schema: {path}",
            code="invocation.journal_invalid",
            details={"path": str(path)},
        )
    return record


class InvocationJournal:
    """Atomically distinguish incomplete and replayable provider calls."""

    def __init__(
        self,
        output_dir: Path,
        /,
        *,
        retry_incomplete: bool = False,
    ) -> None:
        self.output_dir = output_dir.resolve()
        control = self.output_dir / ".verdog"
        if not control.is_dir() or control.is_symlink():
            raise InvocationJournalError(
                f"run control directory is unavailable: {control}",
                code="invocation.journal_unavailable",
                details={"path": str(control)},
            )
        self.root = control / _DIRECTORY
        if self.root.is_symlink() or (self.root.exists() and not self.root.is_dir()):
            raise InvocationJournalError(
                f"agent invocation journal directory is unsafe: {self.root}",
                code="invocation.journal_unsafe",
                details={"path": str(self.root)},
            )
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.retry_incomplete = retry_incomplete

    def address(
        self,
        request: AgentRequest,
        /,
        *,
        transition_epoch: int,
        slot: int,
    ) -> InvocationAddress:
        if transition_epoch < 0:
            raise ValueError("invocation transition epoch must not be negative")
        if slot <= 0:
            raise ValueError("invocation slot must be positive")
        try:
            node_directory = request.node_context.output_dir.parent.resolve()
            relative = node_directory.relative_to(self.output_dir)
        except ValueError as error:
            raise InvocationJournalError(
                "agent invocation output is outside the run directory",
                code="invocation.address_invalid",
                details={"path": str(request.node_context.output_dir)},
            ) from error
        pure = PurePosixPath(relative.as_posix())
        if pure.is_absolute() or ".." in pure.parts:
            raise InvocationJournalError(
                "agent invocation output has an unsafe run-relative address",
                code="invocation.address_invalid",
                details={"path": relative.as_posix()},
            )
        return InvocationAddress(
            run_id=str(request.node_context.run_id),
            transition_epoch=transition_epoch,
            graph_id=str(request.node_context.graph_id),
            node_id=str(request.node_context.node_id),
            edge_id=str(request.node_context.edge_id),
            node_path=relative.as_posix(),
            slot=slot,
        )

    def prepare(
        self,
        address: InvocationAddress,
        request: AgentRequest,
        provider: str,
        /,
    ) -> JournaledReply | None:
        requested = InvocationRequestRecord.from_request(request, provider)
        path = self._path(address)
        if not path.exists():
            self._write_started(path, address, requested, attempt=1)
            return None

        record = _object(path)
        self._validate_identity(path, record, address, requested)
        status = record.get("status")
        attempt = self._attempt(path, record)
        if status == "started":
            if not self.retry_incomplete:
                raise InvocationJournalError(
                    "the previous agent request may have reached its provider; "
                    "resume with --retry-incomplete to authorize a fresh request",
                    code="invocation.ambiguous",
                    details={"path": str(path), "address": address.as_json()},
                )
            self._write_started(path, address, requested, attempt=attempt + 1)
            return None
        if status != "completed":
            self._invalid(path, "status")
        raw_reply = record.get("reply")
        if not isinstance(raw_reply, dict):
            self._invalid(path, "reply")
        reply = cast(dict[str, object], raw_reply)
        text = reply.get("text")
        provider_session_id = reply.get("provider_session_id")
        if not isinstance(text, str) or (
            provider_session_id is not None
            and (not isinstance(provider_session_id, str) or not provider_session_id)
        ):
            self._invalid(path, "reply")
        return JournaledReply(
            AgentReply(
                text=text,
                provider_session_id=(
                    None
                    if provider_session_id is None
                    else ProviderSessionId(provider_session_id)
                ),
            ),
            replayed=True,
        )

    def complete(
        self,
        address: InvocationAddress,
        request: AgentRequest,
        provider: str,
        reply: AgentReply,
        /,
    ) -> JournaledReply:
        requested = InvocationRequestRecord.from_request(request, provider)
        path = self._path(address)
        record = _object(path)
        self._validate_identity(path, record, address, requested)
        if record.get("status") != "started":
            raise InvocationJournalError(
                "agent invocation was not in progress when its reply was recorded",
                code="invocation.journal_conflict",
                details={"path": str(path)},
            )
        attempt = self._attempt(path, record)
        atomic_json(
            path,
            {
                "schema_version": _SCHEMA_VERSION,
                "status": "completed",
                "attempt": attempt,
                "address": address.as_json(),
                "request": requested.as_json(),
                "reply": {
                    "text": reply.text,
                    "provider_session_id": (
                        None
                        if reply.provider_session_id is None
                        else str(reply.provider_session_id)
                    ),
                },
            },
        )
        return JournaledReply(reply, replayed=False)

    def _path(self, address: InvocationAddress, /) -> Path:
        encoded = json.dumps(
            address.as_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return self.root / f"{hashlib.sha256(encoded).hexdigest()}.json"

    @staticmethod
    def _write_started(
        path: Path,
        address: InvocationAddress,
        request: InvocationRequestRecord,
        *,
        attempt: int,
    ) -> None:
        atomic_json(
            path,
            {
                "schema_version": _SCHEMA_VERSION,
                "status": "started",
                "attempt": attempt,
                "address": address.as_json(),
                "request": request.as_json(),
                "reply": None,
            },
        )

    @staticmethod
    def _validate_identity(
        path: Path,
        record: dict[str, object],
        address: InvocationAddress,
        request: InvocationRequestRecord,
    ) -> None:
        if record.get("address") != address.as_json():
            raise InvocationJournalError(
                "agent invocation journal address does not match the resumed request",
                code="invocation.identity_mismatch",
                details={"path": str(path), "field": "address"},
            )
        if record.get("request") != request.as_json():
            raise InvocationJournalError(
                "agent invocation journal request does not match the resumed request",
                code="invocation.identity_mismatch",
                details={"path": str(path), "field": "request"},
            )

    @classmethod
    def _attempt(cls, path: Path, record: dict[str, object], /) -> int:
        value = record.get("attempt")
        if type(value) is not int or value <= 0:
            cls._invalid(path, "attempt")
        return value

    @staticmethod
    def _invalid(path: Path, field: str) -> NoReturn:
        raise InvocationJournalError(
            f"agent invocation journal record has an invalid {field}: {path}",
            code="invocation.journal_invalid",
            details={"path": str(path), "field": field},
        )
