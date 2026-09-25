"""Compiler calls need no login; catalogue credentials remain private."""

import json
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import Request

import pytest

from verdog_runtime.cli import main as cli
from verdog_runtime.cli.local import Clone, write_config
from verdog_runtime.cli.manage import blank_graph
from verdog_runtime.cli.session import (
    Login,
    SessionError,
    config_home,
    credential,
    forget,
    load,
    sign_in,
    store,
)


@pytest.mark.parametrize(
    "configured_origin",
    [
        None,
        "https://157.180.79.112",
        "http://127.0.0.1:18765",
    ],
)
@pytest.mark.parametrize(
    "arguments, endpoint",
    [
        (["generate"], "/api/v1/check"),
        (["check", "--json"], "/api/v1/check"),
        (["analyze", "--json"], "/api/v1/analyze"),
        (["rename", "workflow", "main", "renamed"], "/api/v1/entities/rename"),
    ],
)
def test_compiler_commands_need_no_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    endpoint: str,
    configured_origin: str | None,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "project.json").write_text(
        json.dumps(blank_graph("test.project", "Anonymous compiler")),
        encoding="utf-8",
    )
    write_config(tmp_path, "https://compiler.test")
    if configured_origin is not None:
        write_config(tmp_path, "https://old-project.test", "old-project-secret")
        store(Login("https://old-session.test", "old-session-secret", "ada"))
        monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", configured_origin)
        monkeypatch.setenv("VERDOG_SESSION_TOKEN_STDIN", "1")
        monkeypatch.setattr(
            "sys.stdin", object()
        )  # Compiler calls never consume a token.
    received: list[str] = []

    def respond(request: Request, *, timeout: float) -> BytesIO:
        assert timeout > 0
        assert request.get_header("Authorization") is None
        assert isinstance(request.data, bytes)
        assert "project.json" in json.loads(request.data)["files"]
        received.append(request.full_url)
        return BytesIO(b'{"files":{},"diagnostics":[],"definitions":{}}')

    def type_check(_clone: Clone) -> tuple[list[object], int]:
        return [], 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("verdog_runtime.cli.api.urlopen", respond)
    monkeypatch.setattr(cli, "_type_check_json", type_check)
    assert cli.main(arguments) == 0
    assert received == [
        f"{configured_origin or 'https://compiler.test'}{endpoint}"
    ]


def test_compiler_fallback_preserves_configured_service_and_catalogue_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert credential("https://project.test", anonymous=True).token is None
    store(Login("https://saved.test", "verdog-secret", "ada"))
    assert (
        credential("https://project.test", anonymous=True).origin
        == "https://saved.test"
    )
    forget()
    (tmp_path / ".git").mkdir()
    (tmp_path / "project.json").write_text(
        json.dumps(blank_graph("test.project", "Catalogue auth")),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def repository(_clone: Clone) -> tuple[str, str]:
        return "ada", "tools"

    monkeypatch.setattr(Clone, "require_repository", repository)
    for arguments in (["catalogue"], ["token", "list"], ["access"]):
        assert cli.main(arguments) == 1
        assert "not signed in" in capsys.readouterr().err


@pytest.mark.parametrize("from_stdin", [True, False])
@pytest.mark.parametrize(
    "origin",
    [
        "https://verdog.test",
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://[::1]:8765",
    ],
)
def test_login_exchanges_a_token_without_printing_or_storing_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    from_stdin: bool,
    origin: str,
) -> None:
    github_token = "github-secret-only-for-exchange"
    received: list[str] = []

    def respond(request: Request, *, timeout: float) -> BytesIO:
        assert timeout > 0
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") is None
        assert isinstance(request.data, bytes)
        assert json.loads(request.data) == {"access_token": github_token}
        received.append(request.full_url)
        return BytesIO(
            b'{"session_token":"verdog-session-secret",'
            b'"user":{"id":"1","login":"ada"}}'
        )

    def hidden_prompt(prompt: str) -> str:
        assert "input hidden" in prompt
        assert not from_stdin
        return github_token

    monkeypatch.setattr("verdog_runtime.cli.api.urlopen", respond)
    monkeypatch.setattr("sys.stdin", StringIO(f"{github_token}\n"))
    monkeypatch.setattr("getpass.getpass", hidden_prompt)
    arguments = ["login", origin]
    if from_stdin:
        arguments.append("--github-token-stdin")
    assert cli.main(arguments) == 0
    assert received == [f"{origin}/api/v1/auth/github"]
    saved = load()
    assert saved == Login(origin, "verdog-session-secret", "ada")
    path = config_home() / "session.json"
    assert github_token not in path.read_text("utf-8")
    assert path.stat().st_mode & 0o777 == 0o600
    captured = capsys.readouterr()
    assert "Signed in as ada" in captured.out
    assert github_token not in captured.out + captured.err
    assert saved.token not in captured.out + captured.err


@pytest.mark.parametrize("token", ["", "two tokens", "x" * 1025])
def test_login_rejects_invalid_stdin_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    token: str,
) -> None:
    monkeypatch.setattr("sys.stdin", StringIO(token))
    assert cli.main(["login", "--github-token-stdin"]) == 1
    captured = capsys.readouterr()
    assert "supply one GitHub access token" in captured.err
    if token:
        assert token not in captured.out + captured.err


@pytest.mark.parametrize(
    "origin",
    [
        "http://verdog.test",
        "http://localhost.evil",
        "http://127.0.0.1.evil",
        "https://user:password@verdog.test",
        "https://verdog.test/path",
        "https://verdog.test?secret",
        "https://verdog.test#fragment",
        "https://verdog.test:",
        "https://verdog.test:invalid",
        "https://verdog.test:65536",
        "https://verdog.test:0",
        "https:///missing-host",
        "file:///tmp/verdog",
        "https://verdog.test\n",
    ],
)
def test_login_rejects_unsafe_origins_before_reading_the_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    origin: str,
) -> None:
    monkeypatch.setattr(
        "sys.stdin", object()
    )  # Has no read method: it must not be touched.
    with pytest.raises(SessionError, match="HTTPS service origin"):
        sign_in(origin, from_stdin=True)
    monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", origin)
    with pytest.raises(SessionError, match="HTTPS service origin"):
        credential("https://old-project.test", "old-secret", anonymous=True)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
