"""Exchange GitHub credentials for origin-bound Verdog sessions."""

from __future__ import annotations

import dataclasses
import datetime
import functools
import getpass
import json
import os
import pathlib
import sys
import urllib.parse
from typing import Any, Final, cast

from verdog_runtime.cli import api


def config_home() -> pathlib.Path:
    """Return the XDG configuration directory for Verdog."""
    return (
        pathlib.Path(
            os.environ.get("XDG_CONFIG_HOME", pathlib.Path.home() / ".config")
        )
        / "verdog"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Login:
    """A backend origin, service session token, and authenticated username."""

    origin: str
    token: str
    login: str


class SessionError(Exception):
    """No session, or one the service no longer honours."""


def _path() -> pathlib.Path:
    return config_home() / "session.json"


def store(login: Login) -> pathlib.Path:
    """Persist a validated login in the local session file."""
    if not login.login:
        raise SessionError("the service returned a session without a login")
    target = _path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written 0600 before the bytes go in, so the credential is never briefly
    # readable.
    handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as file:
        json.dump(
            {
                "origin": login.origin,
                "token": login.token,
                "login": login.login,
                "stored_at": datetime.datetime.now(datetime.UTC).isoformat(),
            },
            file,
            indent=2,
        )
        file.write("\n")
    return target


def load() -> Login:
    """Read the saved login or raise SessionError when absent or invalid."""
    try:
        decoded = json.loads(_path().read_text("utf-8"))
    except OSError as error:
        raise SessionError("not signed in; run `verdog login` first") from error
    except ValueError as error:
        raise SessionError(
            f"{_path()} is unreadable; run `verdog login`"
        ) from error
    if not isinstance(decoded, dict):
        raise SessionError(f"{_path()} is not an object; run `verdog login`")
    fields = cast(dict[str, Any], decoded)
    origin, token, login = (
        fields.get("origin"),
        fields.get("token"),
        fields.get("login"),
    )
    if not all(
        isinstance(value, str) and value for value in (origin, token, login)
    ):
        raise SessionError(f"{_path()} is incomplete; run `verdog login`")
    return Login(cast(str, origin), cast(str, token), cast(str, login))


def forget() -> None:
    """Remove the saved session if it exists."""
    _path().unlink(missing_ok=True)


def account() -> api.Service:
    """Return the authenticated account client."""
    origin = backend_origin()
    if origin is not None:
        if os.environ.get("VERDOG_SESSION_TOKEN_STDIN") != "1":
            raise SessionError(
                "not signed in to the configured backend; sign in from VS Code"
            )
        client = _stdin_account()
        if client.origin != origin:
            raise SessionError(
                "the configured backend changed after receiving its session"
            )
        return client
    login = load()
    return api.Service(login.origin, login.token)


@functools.cache
def _stdin_account() -> api.Service:
    """Read one ephemeral, origin-bound service session from stdin."""
    origin = backend_origin()
    if origin is None:
        raise SessionError("a backend origin is required for stdin credentials")
    try:
        token = sys.stdin.read(1025).strip()
    except (EOFError, OSError, UnicodeError) as error:
        raise SessionError(
            "no Verdog session received through stdin"
        ) from error
    if (
        not token
        or len(token) > 512
        or any(not "!" <= char <= "~" for char in token)
    ):
        raise SessionError("supply one Verdog session token through stdin")
    return api.Service(origin, token)


def credential(
    origin: str | None = None,
    token: str | None = None,
    /,
    *,
    anonymous: bool = False,
) -> api.Service:
    """Prefer editor credentials, then the saved session or clone token."""
    selected = backend_origin()
    if selected is not None:
        return api.Service(selected) if anonymous else account()
    try:
        return account()
    except SessionError:
        if origin is None or (token is None and not anonymous):
            raise
        return api.Service(origin, token)


DISCLOSURE: Final = """\
What Verdog shares:

  Signing in gives the configured Verdog service your GitHub account ID and
  username,   plus your name and email when returned by GitHub. The service
  stores this identity   and a GitHub token for permission checks. It does
  not fetch repository file contents   through GitHub. The CLI saves a local
  service session token.

  `verdog generate`, `verdog check`, and `verdog rename` upload project
  manifests and   declared source files, including pinned dependencies, to
  the configured service.   `verdog init` and dependency changes also
  generate through the service. `verdog publish`   checks before submitting
  catalogue metadata. `verdog analyze` uploads only manifests;   the VS Code
  extension can request analysis automatically in a trusted workspace.

  Running workflows can send prompts, code, and results to configured agent
  providers   or other services used by your code and tools. Consult the
  configured service and   providers for their privacy and retention
  practices.
"""
"Printed before accepting a token so the destination and data use are explicit."


def backend_origin() -> str | None:
    """Read the editor origin without reusing saved credentials."""
    origin = os.environ.get("VERDOG_BACKEND_ORIGIN")
    return validate_origin(origin) if origin else None


def validate_origin(origin: str) -> str:
    """Accept HTTPS origins and local HTTP development services."""
    try:
        parsed = urllib.parse.urlsplit(origin)
        secure = parsed.scheme == "https" or (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        )
        if (
            not secure
            or not parsed.hostname
            or parsed.username is not None
            or parsed.path not in {"", "/"}
            or "?" in origin
            or "#" in origin
            or parsed.port == 0
            or parsed.netloc.endswith(":")
            or any(not "!" <= char <= "~" for char in origin)
        ):
            raise ValueError("invalid service origin")
    except ValueError as error:
        raise SessionError(
            "use an HTTPS service origin (HTTP is allowed only on loopback), "
            "without credentials, a path, query, or fragment"
        ) from error
    return origin.rstrip("/")


def sign_in(origin: str, *, from_stdin: bool = False) -> Login:
    """Exchange a GitHub token without storing it locally."""
    origin = validate_origin(origin)
    print(DISCLOSURE)
    print(f"Signing in to {origin}")
    try:
        token = (
            sys.stdin.read(1025)
            if from_stdin
            else getpass.getpass("GitHub access token (input hidden): ")
        ).strip()
    except EOFError as error:
        raise SessionError(
            "no GitHub token received; sign in from VS Code or supply a token"
        ) from error
    if (
        not token
        or len(token) > 512
        or any(ord(char) < 33 or ord(char) > 126 for char in token)
    ):
        raise SessionError(
            "supply one GitHub access token; do not include spaces"
        )
    answer = api.Service(origin).json(
        "POST", "/api/v1/auth/github", {"access_token": token}
    )
    session_token = answer.get("session_token")
    user = answer.get("user")
    login = (
        cast(dict[str, Any], user).get("login")
        if isinstance(user, dict)
        else None
    )
    if (
        not isinstance(session_token, str)
        or not session_token
        or not isinstance(login, str)
        or not login
    ):
        raise SessionError(f"{origin} returned an unusable session")
    return Login(origin, session_token, login)
