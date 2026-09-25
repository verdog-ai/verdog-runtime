"""CLI tests use temporary credentials and explicit service transports."""

from pathlib import Path
from typing import Never

import pytest

from verdog_runtime.cli import session


@pytest.fixture(autouse=True)
def isolate_service_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("VERDOG_BACKEND_ORIGIN", raising=False)
    monkeypatch.delenv("VERDOG_SESSION_TOKEN_STDIN", raising=False)
    session._stdin_account.cache_clear()  # pyright: ignore[reportPrivateUsage]

    def unexpected_request(*_args: object, **_kwargs: object) -> Never:
        pytest.fail("unexpected service request; mock the HTTP transport")

    monkeypatch.setattr("verdog_runtime.cli.api.urlopen", unexpected_request)
