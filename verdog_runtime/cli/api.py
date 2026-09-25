"""The service calls this machine makes.

Standard library only. Compiler calls can be anonymous. Other calls use a repository
token, which is scoped to one repository and lives in the clone, or a user session, which is
the account and lives in `~/.config/verdog`. Neither needs a CSRF header or an `Origin` --
those defend against a browser attaching a credential on its own, and nothing here is
attached automatically.

One client type now, not two. There used to be a `Service` bound to a project id and an
`Account` for everything outside one, because a project token could not create a project and
a session could not be used where a clone expected its own. Projects are GitHub repositories:
nothing is created here, the repository travels in the request, and the two credentials
differ only in what the *service* will let them do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast, override
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    @override
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        # Tokens and uploaded source stay at the explicitly selected destination.
        return None


urlopen = build_opener(_NoRedirect()).open


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
            f"  {path}" for path in cast(list[object], paths) if isinstance(path, str)
        )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Service:
    origin: str
    token: str | None = field(default=None, repr=False)
    timeout: float = 60.0

    def json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(  # noqa: S310 - the origin comes from the local config
            f"{self.origin.rstrip('/')}{path}",
            data=payload,
            method=method,
            headers={
                "Accept": "application/json",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
                **({"Content-Type": "application/json"} if payload is not None else {}),
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                raw = cast(bytes, response.read())
        except HTTPError as error:
            raise _service_error(error) from error
        except (URLError, OSError, TimeoutError) as error:
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
                f"{path} did not return valid JSON", code="service.invalid_response"
            ) from error
        if not isinstance(decoded, dict):
            raise ServiceError(
                f"{path} did not return a JSON object",
                code="service.invalid_response",
            )
        return cast(dict[str, Any], decoded)

    # -- the compiler -------------------------------------------------------------------

    def check(self, files: dict[str, str]) -> dict[str, Any]:
        return self.json("POST", "/api/v1/check", {"files": files})

    def analyze(self, files: dict[str, str]) -> dict[str, Any]:
        return self.json("POST", "/api/v1/analyze", {"files": files})

    def rename(
        self,
        files: dict[str, str],
        kind: str,
        subroutine_id: str | None,
        old_id: str,
        new_id: str,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "files": files,
            "kind": kind,
            "old_id": old_id,
            "new_id": new_id,
        }
        if kind not in {"workflow", "subroutine"}:
            body["subroutine_id"] = subroutine_id
        return self.json("POST", "/api/v1/entities/rename", body)

    # -- the account and the repository ------------------------------------------------

    def me(self) -> dict[str, Any]:
        return self.json("GET", "/api/v1/me")

    def access(self, owner: str, name: str) -> dict[str, Any]:
        return self.json("GET", f"/api/v1/access?repository={owner}/{name}")

    # -- the catalogue -----------------------------------------------------------------

    def catalogue(
        self,
        *,
        query: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        parameters: dict[str, str | int] = {}
        if query:
            parameters["q"] = query
        if visibility and visibility != "all":
            parameters["visibility"] = visibility
        if limit is not None:
            parameters["limit"] = limit
        if cursor:
            parameters["cursor"] = cursor
        suffix = f"?{urlencode(parameters)}" if parameters else ""
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
        return self.json("DELETE", f"/api/v1/catalogue/{entry_id}")

    def preflight(
        self, repository: str, commit: str, workflow_id: str
    ) -> dict[str, Any]:
        return self.json(
            "POST",
            "/api/v1/catalogue/imports",
            {"repository": repository, "commit": commit, "workflow_id": workflow_id},
        )

    # -- tokens ------------------------------------------------------------------------

    def create_token(self, repository: str, name: str) -> dict[str, Any]:
        return self.json(
            "POST", "/api/v1/tokens", {"repository": repository, "name": name}
        )

    def list_tokens(self, repository: str) -> dict[str, Any]:
        return self.json("GET", f"/api/v1/tokens?repository={repository}")

    def revoke_token(self, repository: str, token_id: str) -> dict[str, Any]:
        return self.json("DELETE", f"/api/v1/tokens/{token_id}?repository={repository}")


def _service_error(error: HTTPError) -> ServiceError:
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
