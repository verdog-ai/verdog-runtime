"""Catalogue CLI metadata and machine-facing command contracts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from verdog_runtime.cli.api import Service, ServiceError
from verdog_runtime.cli.local import Clone, WorkspaceError
from verdog_runtime.cli.manage import blank_graph


def _clone(root: Path, package: str = "test.project") -> Clone:
    root.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    (root / "project.json").write_text(
        json.dumps(blank_graph(package, "Research workflow")), encoding="utf-8"
    )
    requirements = (
        root / "src" / Path(*package.split(".")) / "workflows/main/requirements.txt"
    )
    requirements.parent.mkdir(parents=True)
    requirements.write_text("zeta>=2\nalpha==1\n", encoding="utf-8")
    return Clone(root, "http://unused", None)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_describe_json_is_canonical_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from verdog_runtime.cli import main as cli

    clone = _clone(tmp_path / "project")
    requirements = (
        clone.root / clone.workflow_definition("main").source / "requirements.txt"
    )
    requirements.write_text("zeta >= 2\nrequests [security] >= 2\n", encoding="utf-8")
    before = _snapshot(clone.root)
    monkeypatch.chdir(clone.root)

    assert cli.main(["describe", "main", "--json"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "schema_version": 32,
        "package": "test.project",
        "workflow_id": "main",
        "display_name": "Research workflow",
        "preview": {
            "name": "Research workflow",
            "ports": {"enter": "enter", "exit": "exit", "failure": "failure"},
            "nodes": [
                {
                    "id": "enter",
                    "name": "Enter",
                    "kind": "enter",
                    "operation": {},
                },
                {
                    "id": "exit",
                    "name": "Exit",
                    "kind": "exit",
                    "operation": {},
                },
                {
                    "id": "failure",
                    "name": "Failure",
                    "kind": "failure",
                    "operation": {},
                },
            ],
            "edges": [
                {
                    "id": "enter__exit",
                    "name": "Pass through",
                    "source": "enter",
                    "target": "exit",
                    "conditions": [],
                    "effects": [],
                }
            ],
            "features": [],
        },
        "closure": [],
        "environment": {
            "python": ">=3.12",
            "schema_version": 32,
            "requirements": ["requests[security]>=2", "zeta>=2"],
        },
    }
    assert _snapshot(clone.root) == before


def test_catalogue_environment_uses_packaging_normalization() -> None:
    from verdog_runtime.cli.requirements import (
        canonical_python_specifier,
        canonical_requirements,
    )

    assert canonical_python_specifier(" >=3.12 , <4 ") == "<4,>=3.12"
    assert canonical_requirements(
        ["Zoo >= 1", "requests [security] >= 2", "alpha >= 1"]
    ) == (
        "alpha>=1",
        "requests[security]>=2",
        "Zoo>=1",
    )
    with pytest.raises(ValueError, match="declared more than once"):
        canonical_requirements(["httpx>=1", "HTTPX<2"])


def test_catalogue_client_encodes_filters_and_publication_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        self: Service,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del self
        requests.append((method, path, body))
        return {}

    monkeypatch.setattr(Service, "json", request)
    service = Service("http://unused")
    service.catalogue(
        query="proof search", visibility="restricted", limit=25, cursor="page/+="
    )
    service.publish(
        "ada/tools",
        "a" * 40,
        "main",
        "ada.tools",
        "An abstract.",
        {"name": "Tools"},
        environment={
            "python": ">=3.12",
            "schema_version": 32,
            "requirements": ["httpx>=0.27"],
        },
    )

    assert requests == [
        (
            "GET",
            (
                "/api/v1/catalogue?q=proof+search&visibility=restricted&limit=25"
                "&cursor=page%2F%2B%3D"
            ),
            None,
        ),
        (
            "POST",
            "/api/v1/catalogue",
            {
                "repository": "ada/tools",
                "commit": "a" * 40,
                "workflow_id": "main",
                "package": "ada.tools",
                "description": "An abstract.",
                "preview": {"name": "Tools"},
                "environment": {
                    "python": ">=3.12",
                    "schema_version": 32,
                    "requirements": ["httpx>=0.27"],
                },
            },
        ),
    ]


def test_publish_sends_the_same_canonical_description(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verdog_runtime.cli import main as cli
    from verdog_runtime.cli import manage
    from verdog_runtime.cli.catalogue import describe_workflow

    clone = _clone(tmp_path / "publisher", "ada.tools")
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/ada/tools.git"],
        cwd=clone.root,
        check=True,
    )
    expected = describe_workflow(clone, "main")
    published: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Catalogue:
        def publish(self, *args: object, **kwargs: object) -> dict[str, Any]:
            published.append((args, kwargs))
            return {
                "repository": "ada/tools",
                "visibility": "public",
            }

    def catalogue(ignored: Clone | None = None) -> Catalogue:
        del ignored
        return Catalogue()

    def head(ignored: Clone, repository: str | None = None) -> str:
        del ignored
        assert repository == "ada/tools"
        return "d" * 40

    def clean(root: Path, *arguments: str) -> str:
        del root, arguments
        return ""

    def checked(ignored: Clone) -> tuple[int, dict[str, Any]]:
        del ignored
        return 0, {}

    monkeypatch.setattr(manage, "_here", lambda: clone)
    monkeypatch.setattr(manage, "_client", catalogue)
    monkeypatch.setattr(manage, "clone_head", head)
    monkeypatch.setattr(manage, "git", clean)
    monkeypatch.setattr(cli, "check_clone_result", checked)

    assert (
        manage.publish_workflow(
            argparse.Namespace(
                description="A reusable research workflow.",
                workflow="main",
                private=False,
            )
        )
        == 0
    )
    assert published == [
        (
            (
                "ada/tools",
                "d" * 40,
                "main",
                "ada.tools",
                "A reusable research workflow.",
                expected.preview,
            ),
            {
                "private": None,
                "closure": list(expected.closure),
                "environment": expected.environment.as_json(),
            },
        )
    ]


def test_publish_requires_head_on_the_advertised_repository_remote(
    tmp_path: Path,
) -> None:
    from verdog_runtime.cli import manage

    clone = _clone(tmp_path / "publisher", "ada.tools")
    subprocess.run(["git", "add", "-A"], cwd=clone.root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "commit",
            "-qm",
            "publisher",
        ],
        cwd=clone.root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/ada/tools.git"],
        cwd=clone.root,
        check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "other", "https://github.com/eve/fork.git"],
        cwd=clone.root,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/other/main", commit],
        cwd=clone.root,
        check=True,
    )

    with pytest.raises(WorkspaceError, match="remote branch of ada/tools"):
        manage.clone_head(clone, "ada/tools")

    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", commit],
        cwd=clone.root,
        check=True,
    )
    assert manage.clone_head(clone, "ada/tools") == commit


def test_bump_finds_the_newest_release_by_repository_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verdog_runtime.cli import manage

    calls: list[tuple[str | None, int | None, str | None]] = []

    class Catalogue:
        def catalogue(
            self,
            *,
            query: str | None = None,
            visibility: str | None = None,
            limit: int | None = None,
            cursor: str | None = None,
        ) -> dict[str, Any]:
            assert visibility is None
            calls.append((query, limit, cursor))
            if cursor is None:
                return {
                    "entries": [
                        {
                            "repository": "ada/tools-extra",
                            "workflow_id": "main",
                            "commit": "a" * 40,
                        }
                    ],
                    "next_cursor": "page-2",
                }
            return {
                "entries": [
                    {
                        "repository": "Ada/Tools",
                        "workflow_id": "main",
                        "commit": "b" * 40,
                    }
                ],
                "next_cursor": None,
            }

    def catalogue(ignored: Clone | None = None) -> Catalogue:
        del ignored
        return Catalogue()

    monkeypatch.setattr(manage, "_client", catalogue)
    assert (
        manage._newest_release(  # pyright: ignore[reportPrivateUsage]
            Clone(Path("/unused"), "http://unused", None), "ada/tools", "main"
        )
        == "b" * 40
    )
    assert calls == [
        ("ada/tools", 100, None),
        ("ada/tools", 100, "page-2"),
    ]


def test_owned_unpublished_import_reads_package_before_project_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from verdog_runtime.cli import manage

    clone = _clone(tmp_path / "consumer")
    (clone.root / ".gitmodules").mkdir()
    fetched: list[tuple[str, str]] = []

    class Catalogue:
        def preflight(
            self, repository: str, commit: str, workflow: str
        ) -> dict[str, Any]:
            del repository, commit, workflow
            return {
                "package": None,
                "repository_id": 7,
                "closure": [],
                "unpublished": True,
            }

    def remote_package(repository: str, commit: str) -> str:
        fetched.append((repository, commit))
        return "ada.tools"

    def catalogue(ignored: Clone | None = None) -> Catalogue:
        del ignored
        return Catalogue()

    monkeypatch.setattr(manage, "_here", lambda: clone)
    monkeypatch.setattr(manage, "_client", catalogue)
    monkeypatch.setattr(manage, "_remote_package", remote_package)
    commit = "c" * 40
    with pytest.raises(WorkspaceError, match=".gitmodules is not a regular file"):
        manage._import(  # pyright: ignore[reportPrivateUsage]
            argparse.Namespace(
                reference=f"ada/tools@{commit}",
                workflow="main",
                into=None,
                alias=None,
                as_json=False,
            )
        )
    assert fetched == [("ada/tools", commit)]
    assert "externals" in clone.project and clone.project["externals"] == []


def test_import_rolls_back_when_the_exact_checkout_disagrees_with_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from verdog_runtime.cli import manage
    from verdog_runtime.cli.local import git as local_git

    source = _clone(tmp_path / "source", "actual.package")
    subprocess.run(["git", "add", "-A"], cwd=source.root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "commit",
            "-qm",
            "source",
        ],
        cwd=source.root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    consumer = _clone(tmp_path / "consumer")
    subprocess.run(["git", "add", "-A"], cwd=consumer.root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "commit",
            "-qm",
            "consumer",
        ],
        cwd=consumer.root,
        check=True,
    )
    project_before = (consumer.root / "project.json").read_bytes()

    class Catalogue:
        def preflight(
            self, repository: str, selected: str, workflow: str
        ) -> dict[str, Any]:
            del repository, selected, workflow
            return {
                "package": "expected.package",
                "repository_id": 9,
                "closure": [],
            }

    def local_transport(root: Path, *arguments: str) -> str:
        rewritten = [
            str(source.root)
            if item.startswith("https://github.com/ada/tools")
            else item
            for item in arguments
        ]
        if arguments[:2] == ("submodule", "add"):
            return local_git(root, "-c", "protocol.file.allow=always", *rewritten)
        return local_git(root, *rewritten)

    def catalogue(ignored: Clone | None = None) -> Catalogue:
        del ignored
        return Catalogue()

    monkeypatch.setattr(manage, "_here", lambda: consumer)
    monkeypatch.setattr(manage, "_client", catalogue)
    monkeypatch.setattr(manage, "git", local_transport)
    with pytest.raises(WorkspaceError, match="contains package 'actual.package'"):
        manage._import(  # pyright: ignore[reportPrivateUsage]
            argparse.Namespace(
                reference=f"ada/tools@{commit}",
                workflow="main",
                into=None,
                alias="expected.package",
                as_json=False,
            )
        )
    assert (consumer.root / "project.json").read_bytes() == project_before
    assert not (consumer.root / ".gitmodules").exists()
    assert not (consumer.root / "external/expected/package").exists()
    indexed = subprocess.run(
        ["git", "ls-files", "--", "external/expected/package"],
        cwd=consumer.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert indexed == ""


def test_sync_and_errors_use_one_json_object_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from verdog_runtime.cli import main as cli

    clone = _clone(tmp_path / "project")
    monkeypatch.chdir(clone.root)

    def synced(selected: Clone, *, workflow_id: str | None, only_binary: bool) -> int:
        assert selected.root == clone.root
        assert workflow_id == "main"
        assert only_binary
        environment = selected.environment(selected.workflow_definition(workflow_id))
        interpreter = environment / "bin/python"
        interpreter.parent.mkdir(parents=True)
        base_interpreter = tmp_path / "base-python"
        base_interpreter.write_text("", encoding="utf-8")
        try:
            interpreter.symlink_to(base_interpreter)
        except OSError:
            # A platform without unprivileged symlinks still exercises the receipt schema.
            interpreter.write_text("", encoding="utf-8")
        print("prepared")
        return 0

    monkeypatch.setattr(cli, "sync_environment", synced)
    assert cli.main(["sync", "main", "--only-binary", "--json"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result == {
        "status": "ready",
        "workflow_id": "main",
        "environment": str(
            clone.environment(clone.workflow_definition("main")).resolve()
        ),
        "interpreter": str(
            clone.environment(clone.workflow_definition("main")).resolve() / "bin/python"
        ),
        "only_binary": True,
        "requirements": ["alpha==1", "zeta>=2"],
    }
    assert captured.err == "prepared\n"

    (clone.root / "project.json").unlink()
    assert cli.main(["describe", "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "error",
        "error": {
            "code": "workspace.error",
            "message": f"{clone.root} is a git repository with no project.json",
        },
    }
    assert captured.err == ""


def test_import_json_reports_the_created_binding_without_human_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from verdog_runtime.cli import manage

    result: dict[str, Any] = {
        "status": "imported",
        "repository": "ada/tools",
        "commit": "a" * 40,
        "workflow_id": "main",
        "package": "ada.tools",
        "alias": "local.tools",
        "target": "external/local/tools",
        "binding": "main__local_tools_main",
        "generated_status": 0,
    }

    def imported(arguments: argparse.Namespace) -> tuple[int, dict[str, Any]]:
        del arguments
        print("generated progress")
        return 0, result

    monkeypatch.setattr(manage, "_import_result", imported)
    assert (
        manage._import(  # pyright: ignore[reportPrivateUsage]
            argparse.Namespace(as_json=True)
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert captured.err == "generated progress\n"


def test_catalogue_json_preserves_structured_service_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from verdog_runtime.cli import main as cli
    from verdog_runtime.cli import manage

    def unavailable() -> Service:
        raise ServiceError(
            "human rendering [catalogue.cursor_invalid]",
            code="catalogue.cursor_invalid",
            details={"cursor": "opaque"},
            machine_message="the catalogue cursor is invalid",
        )

    monkeypatch.setattr(manage, "account", unavailable)
    assert cli.main(["catalogue", "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "error",
        "error": {
            "code": "catalogue.cursor_invalid",
            "message": "the catalogue cursor is invalid",
            "details": {"cursor": "opaque"},
        },
    }
    assert captured.err == ""


def test_retract_reports_the_entry_id_after_an_empty_204_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from verdog_runtime.cli import manage

    retracted: list[str] = []

    class Catalogue:
        def retract(self, entry_id: str) -> dict[str, Any]:
            retracted.append(entry_id)
            return {}

    def catalogue() -> Catalogue:
        return Catalogue()

    monkeypatch.setattr(manage, "account", catalogue)
    entry_id = "90f34130-ce57-4549-b851-9b20edcec188"
    assert (
        manage._retract(  # pyright: ignore[reportPrivateUsage]
            argparse.Namespace(entry=entry_id)
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out.startswith(f"Retracted catalogue entry {entry_id}.\n")
    assert "@" not in captured.out
    assert retracted == [entry_id]
