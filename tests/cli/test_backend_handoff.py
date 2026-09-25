"""Editor sessions remain separate from terminal credentials."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO, StringIO
from pathlib import Path
from threading import Thread
from typing import override
from urllib.request import Request

import pytest

from verdog_runtime.cli import main as cli
from verdog_runtime.cli import manage
from verdog_runtime.cli.api import Service, ServiceError
from verdog_runtime.cli.api import urlopen as service_urlopen
from verdog_runtime.cli.local import DEFAULT_ORIGIN, Clone
from verdog_runtime.cli.session import (
    Login,
    SessionError,
    account,
    credential,
    load,
    store,
)


def test_ephemeral_session_is_read_once_and_bound_to_its_origin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = store(Login("https://old-session.test", "old-secret", "ada"))
    before = saved.read_bytes()
    monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", "https://157.180.79.112/")
    monkeypatch.setenv("VERDOG_SESSION_TOKEN_STDIN", "1")
    monkeypatch.setattr("sys.stdin", StringIO("ephemeral-verdog-session\n"))

    first = account()
    assert first.origin == "https://157.180.79.112"
    assert first.token == "ephemeral-verdog-session"
    monkeypatch.setattr(
        "sys.stdin", object()
    )  # Every later lookup must use the same read.
    assert account() is first
    assert credential("https://old-project.test", "old-project-secret") is first
    compiler = credential(
        "https://old-project.test", "old-project-secret", anonymous=True
    )
    assert compiler.origin == first.origin
    assert compiler.token is None
    assert saved.read_bytes() == before
    assert first.token not in repr(first)
    monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", "https://different.test")
    with pytest.raises(SessionError, match="backend changed"):
        account()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_configured_backend_never_falls_back_to_saved_or_project_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store(Login("https://configured.test", "saved-secret", "ada"))
    monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", "https://configured.test")
    monkeypatch.setattr("sys.stdin", object())
    with pytest.raises(SessionError, match="sign in from VS Code"):
        credential("https://configured.test", "project-secret")


@pytest.mark.parametrize("token", ["", "two tokens", "x" * 513, "non-ascii-é"])
def test_invalid_ephemeral_session_is_not_echoed_or_saved(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    token: str,
) -> None:
    monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", "https://configured.test")
    monkeypatch.setenv("VERDOG_SESSION_TOKEN_STDIN", "1")
    monkeypatch.setattr("sys.stdin", StringIO(token))
    assert cli.main(["whoami"]) == 1
    captured = capsys.readouterr()
    assert "supply one Verdog session token" in captured.err
    if token:
        assert token not in captured.out + captured.err
    with pytest.raises(SessionError, match="not signed in"):
        load()


def test_authenticated_commands_and_logout_preserve_the_terminal_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = store(Login("https://old-session.test", "old-secret", "ada"))
    before = saved.read_bytes()
    monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", "https://configured.test")
    monkeypatch.setenv("VERDOG_SESSION_TOKEN_STDIN", "1")
    monkeypatch.setattr("sys.stdin", StringIO("ephemeral-secret"))
    requests: list[str] = []

    def respond(request: Request, *, timeout: float) -> BytesIO:
        assert timeout > 0
        assert request.get_header("Authorization") == "Bearer ephemeral-secret"
        requests.append(request.full_url)
        return BytesIO(b'{"user":{"login":"ada"},"entries":[]}')

    monkeypatch.setattr("verdog_runtime.cli.api.urlopen", respond)
    assert cli.main(["whoami"]) == 0
    assert cli.main(["catalogue", "--json"]) == 0
    assert cli.main(["logout"]) == 0
    assert requests == [
        "https://configured.test/api/v1/me",
        "https://configured.test/api/v1/catalogue",
        "https://configured.test/api/v1/logout",
    ]
    assert saved.read_bytes() == before
    captured = capsys.readouterr()
    assert "ephemeral-secret" not in captured.out + captured.err


@pytest.mark.parametrize(
    "configured, explicit, expected",
    [
        (None, None, DEFAULT_ORIGIN),
        ("https://configured.test", None, "https://configured.test"),
        (
            "https://configured.test",
            "https://explicit.test",
            "https://explicit.test",
        ),
    ],
)
def test_terminal_login_origin_precedence(
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    explicit: str | None,
    expected: str,
) -> None:
    if configured:
        monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", configured)

    def signed_in(origin: str, *, from_stdin: bool) -> Login:
        assert origin == expected
        assert not from_stdin
        return Login(origin, "terminal-session", "ada")

    monkeypatch.setattr("verdog_runtime.cli.session.sign_in", signed_in)
    assert cli.main(["login", *([explicit] if explicit else [])]) == 0
    assert load().origin == expected


def test_backend_flag_overrides_the_inherited_origin_before_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "project.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VERDOG_BACKEND_ORIGIN", "https://inherited.test")
    store(Login("https://old-session.test", "old-secret", "ada"))

    def respond(request: Request, *, timeout: float) -> BytesIO:
        assert timeout > 0
        assert request.full_url == "http://127.0.0.1:18765/api/v1/analyze"
        assert request.get_header("Authorization") is None
        return BytesIO(b'{"definitions":{}}')

    monkeypatch.setattr("verdog_runtime.cli.api.urlopen", respond)
    assert (
        cli.main(
            [
                "--backend-origin",
                "http://127.0.0.1:18765",
                "analyze",
                "--json",
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "prefix",
    [
        ["--backend-origin", "https://configured.test"],
        ["--backend-origin=https://configured.test"],
    ],
)
@pytest.mark.parametrize("command", ["run", "restart"])
def test_backend_flag_preserves_project_argument_forwarding(
    prefix: list[str], command: str
) -> None:
    parsed = cli.parse_arguments(
        [
            *prefix,
            command,
            "main" if command == "run" else "run-id",
            *(["--sessions", "fresh"] if command == "restart" else []),
            "--",
            "--backend-origin",
            "a-project-argument",
        ]
    )
    assert parsed.backend_origin == "https://configured.test"
    assert parsed.project_arguments == [
        "--backend-origin",
        "a-project-argument",
    ]
    if command == "restart":
        assert parsed.arguments_overridden


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("authenticated", [False, True])
def test_service_refuses_redirects_of_tokens_and_bodies(
    monkeypatch: pytest.MonkeyPatch, status: int, authenticated: bool
) -> None:
    received: list[str] = []

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - HTTP handler name
            received.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(status if self.path == "/start" else 200)
            self.send_header(
                "Location", f"http://localhost:{server.server_port}/target"
            )
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        do_POST = do_GET

        @override
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    monkeypatch.setattr("verdog_runtime.cli.api.urlopen", service_urlopen)
    with HTTPServer(("127.0.0.1", 0), Redirect) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = Service(
                f"http://127.0.0.1:{server.server_port}",
                "verdog-secret" if authenticated else None,
            )
            with pytest.raises(ServiceError) as raised:
                client.json(
                    "GET" if authenticated else "POST",
                    "/start",
                    None
                    if authenticated
                    else {"access_token": "github-secret"},
                )
            assert raised.value.code == "service.http_error"
            assert received == ["/start"]
        finally:
            server.shutdown()
            worker.join(timeout=2)


@pytest.mark.parametrize("command", ["init", "clone"])
@pytest.mark.parametrize("explicit", [None, "https://explicit.test"])
def test_new_projects_persist_the_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    explicit: str | None,
) -> None:
    destination = tmp_path / "project"
    selected = "http://127.0.0.1:18765"
    expected = explicit or selected
    monkeypatch.chdir(tmp_path)

    def fake_git(root: Path, *arguments: str) -> str:
        if arguments[0] == "clone":
            (Path(arguments[-1]) / ".git").mkdir(parents=True)
        elif arguments[0] == "init":
            (root / ".git").mkdir()
        return ""

    def generated(clone: Clone) -> int:
        assert clone.origin == expected
        return 0

    monkeypatch.setattr(manage.local, "git", fake_git)
    monkeypatch.setattr(cli, "generate_clone", generated)
    arguments = ["--backend-origin", selected, command]
    arguments += (
        ["Project", str(destination), "--package", "ada.project"]
        if command == "init"
        else ["ada/project", str(destination), "--ssh"]
    )
    if explicit:
        arguments += ["--origin", explicit]
    assert cli.main(arguments) == 0
    assert json.loads(
        (destination / ".git/verdog.json").read_text("utf-8")
    ) == {
        "origin": expected,
    }
