"""HTTP client for compiler and authenticated catalogue operations."""

from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, cast, override


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        # Tokens and uploaded source stay at the explicitly selected
        # destination.
        return None


urlopen = urllib.request.build_opener(_NoRedirect()).open


class ServiceError(Exception):
    """The service refused, or could not be reached."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "service.error",
        details: object = None,
        machine_message: str | None = None,
    ) -> None:
        """Retain the human-readable message and structured service error."""
        super().__init__(message)
        self.code = code
        self.details = details
        self.machine_message = machine_message or message


def format_error(message: str, code: object, details: object = None) -> str:
    """A compiler error for a terminal, including any affected paths."""
    lines = [f"{message} [{code}]" if isinstance(code, str) else message]
    paths = (
        cast(dict[str, object], details).get("paths")
        if isinstance(details, dict)
        else None
    )
    if isinstance(paths, list):
        lines.extend(
            f"  {path}"
            for path in cast(list[object], paths)
            if isinstance(path, str)
        )
    return "\n".join(lines)


@dataclasses.dataclass(frozen=True, slots=True)
class Service:
    """A service client with optional credentials and no redirects."""

    origin: str
    token: str | None = dataclasses.field(default=None, repr=False)
    timeout: float = 60.0

    def json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send JSON, returning an object or raising ServiceError."""
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - the origin comes from the local config
            f"{self.origin.rstrip('/')}{path}",
            data=payload,
            method=method,
            headers={
                "Accept": "application/json",
                **(
                    {"Authorization": f"Bearer {self.token}"}
                    if self.token
                    else {}
                ),
                **(
                    {"Content-Type": "application/json"}
                    if payload is not None
                    else {}
                ),
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                raw = cast(bytes, response.read())
        except urllib.error.HTTPError as error:
            raise _service_error(error) from error
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise ServiceError(
                f"{self.origin} could not be reached: {error}",
                code="service.unavailable",
            ) from error
        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except ValueError as error:
            raise ServiceError(
                f"{path} did not return valid JSON",
                code="service.invalid_response",
            ) from error
        if not isinstance(decoded, dict):
            raise ServiceError(
                f"{path} did not return a JSON object",
                code="service.invalid_response",
            )
        return cast(dict[str, Any], decoded)

    # the compiler

    def check(self, files: dict[str, str]) -> dict[str, Any]:
        """Validate project sources and request their generated projection."""
        return self.json("POST", "/api/v1/check", {"files": files})

    def analyze(self, files: dict[str, str]) -> dict[str, Any]:
        """Analyze graph manifests without executing project code."""
        return self.json("POST", "/api/v1/analyze", {"files": files})

    def rename(
        self,
        files: dict[str, str],
        kind: str,
        subroutine_id: str | None,
        old_id: str,
        new_id: str,
    ) -> dict[str, Any]:
        """Request an identity rename and its affected source projection."""
        body: dict[str, Any] = {
            "files": files,
            "kind": kind,
            "old_id": old_id,
            "new_id": new_id,
        }
        if kind not in {"workflow", "subroutine"}:
            body["subroutine_id"] = subroutine_id
        return self.json("POST", "/api/v1/entities/rename", body)

    # the account and the repository

    def me(self) -> dict[str, Any]:
        """Return the authenticated account and its access information."""
        return self.json("GET", "/api/v1/me")

    def access(self, owner: str, name: str) -> dict[str, Any]:
        """Return the current account's access to a GitHub repository."""
        return self.json("GET", f"/api/v1/access?repository={owner}/{name}")

    # the catalogue

    def catalogue(
        self,
        *,
        query: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List visible workflow offers with optional search and pagination."""
        parameters: dict[str, str | int] = {}
        if query:
            parameters["q"] = query
        if visibility and visibility != "all":
            parameters["visibility"] = visibility
        if limit is not None:
            parameters["limit"] = limit
        if cursor:
            parameters["cursor"] = cursor
        suffix = f"?{urllib.parse.urlencode(parameters)}" if parameters else ""
        return self.json("GET", f"/api/v1/catalogue{suffix}")

    def entry(self, entry_id: str) -> dict[str, Any]:
        """One offer, with the preview the listing leaves out."""
        return self.json("GET", f"/api/v1/catalogue/{entry_id}")

    def publish(
        self,
        repository: str,
        commit: str,
        workflow_id: str,
        package: str,
        description: str,
        preview: dict[str, Any],
        *,
        private: bool | None = None,
        closure: list[dict[str, Any]] | None = None,
        environment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish immutable workflow metadata for an accessible repository."""
        body: dict[str, Any] = {
            "repository": repository,
            "commit": commit,
            "workflow_id": workflow_id,
            "package": package,
            "description": description,
            "preview": preview,
        }
        if private is not None:
            body["private"] = private
        if closure:
            body["closure"] = closure
        if environment is not None:
            body["environment"] = environment
        return self.json("POST", "/api/v1/catalogue", body)

    def retract(self, entry_id: str) -> dict[str, Any]:
        """Remove an offer from the catalogue."""
        return self.json("DELETE", f"/api/v1/catalogue/{entry_id}")

    def preflight(
        self, repository: str, commit: str, workflow_id: str
    ) -> dict[str, Any]:
        """Check access and resolve the metadata for a proposed import."""
        return self.json(
            "POST",
            "/api/v1/catalogue/imports",
            {
                "repository": repository,
                "commit": commit,
                "workflow_id": workflow_id,
            },
        )

    # tokens

    def create_token(self, repository: str, name: str) -> dict[str, Any]:
        """Create a named, repository-scoped token."""
        return self.json(
            "POST", "/api/v1/tokens", {"repository": repository, "name": name}
        )

    def list_tokens(self, repository: str) -> dict[str, Any]:
        """List repository token metadata without exposing token values."""
        return self.json("GET", f"/api/v1/tokens?repository={repository}")

    def revoke_token(self, repository: str, token_id: str) -> dict[str, Any]:
        """Revoke a repository token by its identifier."""
        return self.json(
            "DELETE", f"/api/v1/tokens/{token_id}?repository={repository}"
        )


def _service_error(error: urllib.error.HTTPError) -> ServiceError:
    """Retain the service's structured refusal for machine-mode CLI callers."""
    message = f"{error.code} {error.reason}"
    try:
        payload = json.loads(error.read())
    except (ValueError, OSError):
        return ServiceError(message, code="service.http_error")
    if isinstance(payload, dict):
        detail = cast(dict[str, Any], payload).get("error")
        if not isinstance(detail, dict):
            return ServiceError(message, code="service.http_error")
        fields = cast(dict[str, Any], detail)
        text, code = fields.get("message"), fields.get("code")
        if isinstance(text, str):
            return ServiceError(
                format_error(text, code, fields.get("details")),
                code=code if isinstance(code, str) else "service.http_error",
                details=fields.get("details"),
                machine_message=text,
            )
    return ServiceError(message, code="service.http_error")


__all__ = ["Service", "ServiceError", "format_error"]
