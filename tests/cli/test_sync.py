"""`verdog sync`: an editor runtime and one exact environment per workflow."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from verdog_runtime.cli.local import SCHEMA_VERSION, Clone, LocalDefinition, WorkspaceError
from verdog_runtime.cli.requirements import parse_requirements
from verdog_runtime.cli.sync import (
    ENVIRONMENT_MARKER,
    _environment_spec,  # pyright: ignore[reportPrivateUsage]
    _install,  # pyright: ignore[reportPrivateUsage]
    _rebuild_directory,  # pyright: ignore[reportPrivateUsage]
    _site_packages,  # pyright: ignore[reportPrivateUsage]
    _sync_editor_environment,  # pyright: ignore[reportPrivateUsage]
    _sync_one,  # pyright: ignore[reportPrivateUsage]
    _subroutine_sources,  # pyright: ignore[reportPrivateUsage]
    require_current_environment,
    sync,
)


def _pin(alias: str, package: str, commit: str) -> dict[str, Any]:
    return {
        "alias": alias,
        "package": package,
        "owner": "ada",
        "name": package,
        "commit": commit,
    }


def _workflow(subroutine: str) -> dict[str, Any]:
    return {
        "subroutine": subroutine,
        "profiles": [],
        "sessions": [],
        "profile_arguments": {},
        "session_arguments": {},
    }


def _subroutine(
    identifier: str,
    *,
    nodes: list[dict[str, Any]] | None = None,
    workflows: list[dict[str, Any]] | None = None,
    subroutines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "name": identifier,
        "ports": {"enter": "enter", "exit": "exit", "failure": "failure"},
        "features": [],
        "profiles": [],
        "profile_parameters": [],
        "sessions": [],
        "session_parameters": [],
        "nodes": [] if nodes is None else nodes,
        "edges": [],
        "workflows": [] if workflows is None else workflows,
        "subroutines": [] if subroutines is None else subroutines,
    }


def _manifest(
    root: Path,
    package: str,
    *,
    requirements: list[str] | None = None,
    externals: list[dict[str, Any]] | None = None,
    subroutine: dict[str, Any] | None = None,
) -> Clone:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    root_subroutine = subroutine or _subroutine("main")
    (root / "project.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "package": package,
                "workflow": _workflow(cast(str, root_subroutine["id"])),
                "subroutine": root_subroutine,
                "externals": [] if externals is None else externals,
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    clone = Clone(root, "http://unused", None)
    for index, definition in enumerate(clone.workflow_definitions()):
        path = root / definition.source / "requirements.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        declared = requirements if index == 0 and requirements is not None else []
        path.write_text("".join(f"{item}\n" for item in declared), encoding="utf-8")
    return clone


def _clone(tmp_path: Path, requirements: list[str]) -> Clone:
    return _manifest(tmp_path / "clone", "ada.thing", requirements=requirements)


def _root_definition(clone: Clone) -> LocalDefinition:
    return clone.workflow_definitions()[0]


def _interpreter_of(command: list[str]) -> str:
    return command[command.index("--python") + 1]


def test_requirements_are_full_pep_508_lines() -> None:
    assert parse_requirements(
        "\n  # comment\nHTTPX[http2]>=0.27; python_version >= '3.12'\n"
        "helper @ https://example.com/helper.whl\n",
        "workflow/requirements.txt",
    ) == (
        "HTTPX[http2]>=0.27; python_version >= '3.12'",
        "helper @ https://example.com/helper.whl",
    )

    for source, line, problem in (
        ("\n-r shared.txt\n", 2, "pip directives are not supported"),
        ("not a requirement !!!\n", 1, "invalid PEP 508 requirement"),
        (
            "HTTPX>=0.27\nhttpx[http2]\n",
            2,
            "requirement httpx is already declared at line 1",
        ),
    ):
        with pytest.raises(
            ValueError,
            match=rf"workflow/requirements\.txt:{line}:.*{problem}",
        ):
            parse_requirements(source, "workflow/requirements.txt")


def test_the_runtime_and_its_dependencies_are_provisioned_locally(
    tmp_path: Path,
) -> None:
    clone = _clone(tmp_path, [])
    definition = _root_definition(clone)
    assert sync(clone) == 0

    installed = clone.site_packages(definition)
    assert installed is not None
    assert (installed / "verdog_runtime" / "__init__.py").is_file()

    interpreter = clone.environment(definition) / "bin/python"
    proof = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            str(interpreter),
            "-I",
            "-c",
            (
                "import importlib.util, verdog_runtime, verdog_runtime.cli, packaging; "
                "assert importlib.util.find_spec('verdog_compiler') is None; "
                "assert importlib.util.find_spec('pyverdog') is None; "
                "assert importlib.util.find_spec('jinja2') is None; "
                "assert importlib.util.find_spec('markupsafe') is None; "
                "print(verdog_runtime.__file__)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proof.returncode == 0, proof.stderr
    assert "verdog_runtime" in proof.stdout
    assert str(installed) in proof.stdout

    editor = clone.root / ".venv" / "bin/python"
    editor_proof = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [str(editor), "-I", "-c", "import verdog_runtime"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert editor_proof.returncode == 0, editor_proof.stderr

    marker = json.loads(
        (clone.environment(definition) / ENVIRONMENT_MARKER).read_text("utf-8")
    )
    assert marker["python"]
    assert marker["requirements"] == []
    assert marker["runtime_requirements"] == sorted(marker["runtime_requirements"])
    assert marker["source_roots"] == [str((clone.root / "src").resolve())]


def test_syncing_twice_rebuilds_the_generated_environment(tmp_path: Path) -> None:
    clone = _clone(tmp_path, [])
    definition = _root_definition(clone)
    assert sync(clone) == 0
    stale = clone.environment(definition) / "stale"
    stale.write_text("old", encoding="utf-8")
    editor_owned = clone.root / ".venv" / "keep"
    editor_owned.write_text("mine", encoding="utf-8")
    editor_stale = _site_packages(clone.root / ".venv") / "stale.py"
    editor_stale.write_text("old", encoding="utf-8")

    assert sync(clone) == 0
    assert not stale.exists()
    assert editor_owned.read_text("utf-8") == "mine"
    assert not editor_stale.exists()


def test_invalid_requirements_do_not_clear_the_previous_environment(
    tmp_path: Path,
) -> None:
    clone = _clone(tmp_path, [])
    definition = _root_definition(clone)
    environment = clone.environment(definition)
    environment.mkdir(parents=True)
    sentinel = environment / "keep"
    sentinel.write_text("mine", encoding="utf-8")
    (clone.root / definition.source / "requirements.txt").write_text(
        "--index-url https://example.invalid\n", encoding="utf-8"
    )

    with pytest.raises(WorkspaceError, match=r"requirements.txt:1"):
        _sync_one(clone, definition, only_binary=False)
    assert sentinel.read_text("utf-8") == "mine"


def test_failed_install_does_not_mark_the_environment_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = _clone(tmp_path, ["missing-package==1"])
    definition = _root_definition(clone)

    def fail(command: list[str], cwd: Path) -> int:
        del command, cwd
        return 1

    monkeypatch.setattr("verdog_runtime.cli.sync._install", fail)

    assert _sync_one(clone, definition, only_binary=False) == 1
    assert not (clone.environment(definition) / ENVIRONMENT_MARKER).exists()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["create", "seed", "pip", "marker", "interrupt"])
def test_workflow_rebuild_recovers_after_each_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool, failure: str
) -> None:
    clone = _clone(tmp_path, ["missing-package==1"])
    definition = _root_definition(clone)
    environment = clone.environment(definition)
    if existing:
        environment.mkdir(parents=True)
        (environment / "keep").write_text("previous environment", encoding="utf-8")
        (environment / ENVIRONMENT_MARKER).write_text("old marker", encoding="utf-8")

    def create(_builder: object, directory: str) -> None:
        target = Path(directory)
        (target / "bin").mkdir(parents=True)
        (target / "bin/python").write_text("interpreter", encoding="utf-8")
        (target / f"lib/python{sysconfig.get_python_version()}/site-packages").mkdir(parents=True)
        if failure == "create":
            raise OSError("injected create failure")

    def seed(site_packages: Path, requirements: tuple[str, ...]) -> None:
        del requirements
        (site_packages / "partial.py").write_text("partial", encoding="utf-8")
        if failure == "seed":
            raise OSError("injected seed failure")

    def install(command: list[str], cwd: Path) -> int:
        del command, cwd
        if failure == "interrupt":
            raise KeyboardInterrupt
        return 9 if failure == "pip" else 0

    write_text = Path.write_text

    def write_or_fail(path: Path, content: str, **kwargs: Any) -> int:
        if failure == "marker" and path.name == ENVIRONMENT_MARKER:
            raise OSError("injected marker failure")
        return write_text(path, content, **kwargs)

    monkeypatch.setattr("verdog_runtime.cli.sync.venv.EnvBuilder.create", create)
    monkeypatch.setattr("verdog_runtime.cli.sync._seed_distributions", seed)
    monkeypatch.setattr("verdog_runtime.cli.sync._install", install)
    monkeypatch.setattr(Path, "write_text", write_or_fail)
    if failure == "pip":
        assert _sync_one(clone, definition, only_binary=False) == 9
    else:
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
            _sync_one(clone, definition, only_binary=False)
    if existing:
        assert sorted(path.name for path in environment.iterdir()) == [ENVIRONMENT_MARKER, "keep"]
        assert (environment / "keep").read_text("utf-8") == "previous environment"
        assert (environment / ENVIRONMENT_MARKER).read_text("utf-8") == "old marker"
    else:
        assert not environment.exists()
    assert not list(environment.parent.glob(".verdog-rebuild-*"))


def test_editor_rebuild_restores_packages_without_changing_other_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = _clone(tmp_path, [])
    environment = clone.root / ".venv"
    packages = environment / f"lib/python{sysconfig.get_python_version()}/site-packages"
    packages.mkdir(parents=True)
    (packages / "old.py").write_text("old package", encoding="utf-8")
    (environment / "keep").write_text("editor-owned", encoding="utf-8")

    def fail(destination: Path, requirements: tuple[str, ...]) -> None:
        del requirements
        (destination / "partial.py").write_text("partial package", encoding="utf-8")
        raise OSError("injected editor seed failure")

    monkeypatch.setattr("verdog_runtime.cli.sync._seed_distributions", fail)
    with pytest.raises(OSError, match="editor seed"):
        _sync_editor_environment(clone)
    assert (packages / "old.py").read_text("utf-8") == "old package"
    assert not (packages / "partial.py").exists()
    assert (environment / "keep").read_text("utf-8") == "editor-owned"


def test_environment_retains_backup_when_restoration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "keep").write_text("original", encoding="utf-8")
    replace = Path.replace

    def fail_restore(source: Path, target: Path) -> Path:
        if source.name == "previous":
            raise OSError("injected restoration failure")
        return replace(source, target)

    def fail_build() -> int:
        environment.mkdir()
        return 1

    monkeypatch.setattr(Path, "replace", fail_restore)
    with pytest.raises(WorkspaceError, match="Recover the previous environment") as caught:
        _rebuild_directory(environment, fail_build)
    backups = list(tmp_path.glob(".verdog-rebuild-*"))
    assert len(backups) == 1
    assert str(backups[0]) in str(caught.value)
    assert (backups[0] / "previous/keep").read_text("utf-8") == "original"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process group")
def test_install_stops_and_reaps_pip_before_propagating_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class Process:
        pid = 12345

        def __init__(
            self, command: list[str], cwd: Path, *, start_new_session: bool,
            creationflags: int,
        ) -> None:
            assert command == ["pip"] and cwd == tmp_path
            assert start_new_session and creationflags == 0

        def __enter__(self) -> Process:
            return self

        def __exit__(self, *_: object) -> None:
            events.append("closed")

        def wait(self) -> int:
            events.append("wait")
            if events == ["wait"]:
                raise KeyboardInterrupt
            return -9

        def kill(self) -> None:
            events.append("parent-kill")

    def kill_group(pid: int, sig: int) -> None:
        assert pid == Process.pid and sig == signal.SIGKILL
        events.append("kill")

    monkeypatch.setattr("verdog_runtime.cli.sync.subprocess.Popen", Process)
    monkeypatch.setattr("verdog_runtime.cli.sync.os.killpg", kill_group)
    with pytest.raises(KeyboardInterrupt):
        _install(["pip"], tmp_path)
    assert events == ["wait", "kill", "parent-kill", "wait", "closed"]


def test_environment_cleanup_failure_does_not_report_a_failed_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "old").write_text("old", encoding="utf-8")

    def fail_cleanup(path: Path) -> None:
        assert path.name.startswith(".verdog-rebuild-")
        raise PermissionError("injected cleanup failure")

    def build() -> int:
        environment.mkdir()
        (environment / "new").write_text("new", encoding="utf-8")
        return 0

    monkeypatch.setattr("verdog_runtime.cli.local.shutil.rmtree", fail_cleanup)
    assert _rebuild_directory(environment, build) == 0
    assert (environment / "new").read_text("utf-8") == "new"
    assert "temporary files remain" in capsys.readouterr().err
    backup = next(tmp_path.glob(".verdog-rebuild-*"))
    assert (backup / "previous/old").read_text("utf-8") == "old"


@pytest.mark.parametrize("termination", ["success", "missing", "nonzero"])
def test_windows_install_cancellation_preserves_backup_if_tree_stop_is_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination: str
) -> None:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "old").write_text("previous", encoding="utf-8")
    events: list[str] = []

    class Process:
        pid = 12345

        def __init__(self, command: list[str], cwd: Path, **kwargs: object) -> None:
            assert command == ["pip"] and cwd == tmp_path
            assert kwargs == {"start_new_session": False, "creationflags": 512}

        def __enter__(self) -> Process:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def wait(self) -> int:
            events.append("wait")
            if events == ["wait"]:
                raise KeyboardInterrupt
            return -9

        def kill(self) -> None:
            events.append("kill")

    def taskkill(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert command == ["taskkill", "/PID", "12345", "/T", "/F"]
        if termination == "missing":
            raise FileNotFoundError("taskkill unavailable")
        return subprocess.CompletedProcess(command, 0 if termination == "success" else 1)

    def build() -> int:
        environment.mkdir()
        (environment / "partial").write_text("new", encoding="utf-8")
        return _install(["pip"], tmp_path)

    monkeypatch.setattr("verdog_runtime.cli.sync.sys.platform", "win32")
    monkeypatch.setattr("verdog_runtime.cli.sync.subprocess.CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr("verdog_runtime.cli.sync.subprocess.Popen", Process)
    monkeypatch.setattr("verdog_runtime.cli.sync.subprocess.run", taskkill)
    with pytest.raises(KeyboardInterrupt if termination == "success" else WorkspaceError):
        _rebuild_directory(environment, build)
    assert events == ["wait", "kill", "wait"]
    if termination == "success":
        assert (environment / "old").read_text("utf-8") == "previous"
        assert not (environment / "partial").exists()
    else:
        assert (environment / "partial").read_text("utf-8") == "new"
        backup = next(tmp_path.glob(".verdog-rebuild-*"))
        assert (backup / "previous/old").read_text("utf-8") == "previous"


def test_changed_requirements_make_run_and_type_check_request_targeted_sync(
    tmp_path: Path,
) -> None:
    clone = _clone(tmp_path, [])
    definition = _root_definition(clone)
    assert sync(clone) == 0
    assert require_current_environment(clone, definition) == clone.environment(
        definition
    )

    (clone.root / definition.source / "requirements.txt").write_text(
        "six==1.17.0\n", encoding="utf-8"
    )
    with pytest.raises(WorkspaceError, match=r"run `verdog sync main`"):
        require_current_environment(clone, definition)
    with pytest.raises(WorkspaceError, match=r"run `verdog sync main`"):
        clone.type_check_commands()


def test_runtime_content_and_old_markers_make_an_environment_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "installed/verdog_runtime"
    package.mkdir(parents=True)
    runtime = package / "__init__.py"
    runtime.write_text("BUILD = 1\n", encoding="utf-8")

    def locate_file(_: object) -> Path:
        return runtime

    installed = cast(
        metadata.Distribution,
        SimpleNamespace(
            files=[Path("verdog_runtime/__init__.py")],
            locate_file=locate_file,
            requires=(),
            version="0.1.0",
        ),
    )

    def distribution(name: str) -> metadata.Distribution:
        assert name == "verdog-runtime"
        return installed

    monkeypatch.setattr("verdog_runtime.cli.sync.metadata.distribution", distribution)
    clone = _clone(tmp_path, [])
    definition = _root_definition(clone)
    marker = clone.environment(definition) / ENVIRONMENT_MARKER
    marker.parent.mkdir(parents=True)
    specification, _, _ = _environment_spec(clone, definition)
    marker.write_text(json.dumps(specification), encoding="utf-8")
    assert require_current_environment(clone, definition) == marker.parent

    old_marker = dict(specification)
    del old_marker["runtime"]
    marker.write_text(json.dumps(old_marker), encoding="utf-8")
    with pytest.raises(WorkspaceError, match=r"run `verdog sync main`"):
        require_current_environment(clone, definition)

    marker.write_text(json.dumps(specification), encoding="utf-8")
    runtime.write_text("BUILD = 2\n", encoding="utf-8")
    changed, _, _ = _environment_spec(clone, definition)
    assert changed["runtime"] != specification["runtime"]
    with pytest.raises(WorkspaceError, match=r"run `verdog sync main`"):
        require_current_environment(clone, definition)


def test_environment_ids_cannot_escape_the_generated_root(tmp_path: Path) -> None:
    clone = _clone(tmp_path, [])
    project = clone.project
    project["workflow"]["subroutine"] = "../outside"
    (clone.root / "project.json").write_text(json.dumps(project), encoding="utf-8")
    outside = clone.root / ".verdog" / "outside"
    outside.mkdir(parents=True)
    marker = outside / "mine"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="valid id"):
        sync(clone)
    assert marker.read_text("utf-8") == "keep"


@pytest.mark.parametrize("identifier", ["match", "type", "chíld", "child_"])
def test_local_definition_ids_use_the_language_identifier_grammar(
    tmp_path: Path, identifier: str
) -> None:
    clone = _clone(tmp_path, [])
    project = clone.project
    project["subroutine"]["subroutines"] = [_subroutine(identifier)]
    (clone.root / "project.json").write_text(json.dumps(project), encoding="utf-8")

    with pytest.raises(WorkspaceError, match="id"):
        clone.local_definitions()


def test_sync_refuses_a_symlinked_environment_root(tmp_path: Path) -> None:
    clone = _clone(tmp_path, [])
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "mine"
    marker.write_text("keep", encoding="utf-8")
    try:
        (clone.root / ".verdog").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(WorkspaceError, match="symbolic link"):
        sync(clone)
    assert marker.read_text("utf-8") == "keep"


def test_type_check_uses_each_workflow_environment(tmp_path: Path) -> None:
    child = _workflow("main__child")
    graph = _subroutine(
        "main",
        workflows=[child],
        subroutines=[_subroutine("child")],
    )
    clone = _manifest(tmp_path / "clone", "ada.thing", subroutine=graph)
    root, nested = clone.workflow_definitions()
    assert root.module == "ada.thing.workflows.main"
    assert nested.module == "ada.thing.subroutines.main.workflows.child"
    with pytest.raises(WorkspaceError, match=r"verdog sync main"):
        clone.type_check_commands()

    assert sync(clone) == 0
    commands = clone.type_check_commands()
    assert _interpreter_of(commands[0]) == str(clone.environment(root))
    assert _interpreter_of(commands[1]) == str(clone.environment(nested))
    assert clone.environment(root).name == "main"
    assert clone.environment(nested).name == "main__child"


def test_definition_names_repeat_in_nested_scopes(tmp_path: Path) -> None:
    def call(target: str) -> dict[str, Any]:
        return {
            "id": "call",
            "kind": "subroutine_call",
            "operation": {
                "target": target,
                "profile_arguments": {},
                "session_arguments": {},
            },
        }

    nested = _subroutine("loop")
    outer = _subroutine(
        "loop",
        nodes=[call("main__loop__loop")],
        workflows=[_workflow("main__loop__loop")],
        subroutines=[nested],
    )
    root = _subroutine(
        "main",
        nodes=[call("main__loop")],
        workflows=[_workflow("main__loop")],
        subroutines=[outer],
    )
    clone = _manifest(tmp_path / "clone", "ada.thing", subroutine=root)

    assert [item.local_id for item in clone.workflow_definitions()] == [
        "main",
        "main__loop",
        "main__loop__loop",
    ]
    assert [
        item.local_id
        for item in clone.subroutine_closure(clone.workflow_definitions()[0])
    ] == ["main", "main__loop", "main__loop__loop"]
    assert [clone.environment(item).name for item in clone.workflow_definitions()] == [
        "main",
        "main__loop",
        "main__loop__loop",
    ]


def test_undeclared_dependencies_are_not_installed(tmp_path: Path) -> None:
    clone = _clone(tmp_path, [])
    definition = _root_definition(clone)
    assert sync(clone) == 0
    installed = clone.site_packages(definition)
    assert installed is not None
    assert not list(installed.glob("six*"))


def test_sync_visits_every_workflow_in_each_nested_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[
            _pin("pin.left", "publisher.branch", "a" * 40),
            _pin("pin.right", "publisher.branch", "b" * 40),
        ],
    )
    left = clone.root / "external" / "pin" / "left"
    right = clone.root / "external" / "pin" / "right"
    left_clone = _manifest(
        left,
        "publisher.branch",
        externals=[_pin("pin.shared", "publisher.shared", "c" * 40)],
    )
    _manifest(
        right,
        "publisher.branch",
        externals=[_pin("pin.shared", "publisher.shared", "d" * 40)],
    )
    _manifest(left / "external" / "pin" / "shared", "publisher.shared")
    _manifest(right / "external" / "pin" / "shared", "publisher.shared")

    # Local workflow envelopes are provisioned independently, including below a pin.
    left_project = left_clone.project
    left_project["subroutine"]["workflows"] = [_workflow("main__other")]
    left_project["subroutine"]["subroutines"] = [_subroutine("other")]
    (left / "project.json").write_text(json.dumps(left_project), encoding="utf-8")

    visited: list[tuple[str, str, bool]] = []

    def recorded(
        current: Clone, definition: LocalDefinition, *, only_binary: bool
    ) -> int:
        visited.append(
            (
                current.root.relative_to(clone.root).as_posix(),
                definition.local_id,
                only_binary,
            )
        )
        return 0

    monkeypatch.setattr("verdog_runtime.cli.sync._sync_one", recorded)
    assert sync(clone) == 0
    assert visited == [
        (".", "main", False),
        ("external/pin/left", "main", True),
        ("external/pin/left", "main__other", True),
        ("external/pin/left/external/pin/shared", "main", True),
        ("external/pin/right", "main", True),
        ("external/pin/right/external/pin/shared", "main", True),
    ]


def test_full_sync_initializes_missing_direct_and_nested_submodules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[_pin("dependency.child", "publisher.child", "a" * 40)],
    )
    child = clone.root / "external" / "dependency" / "child"
    leaf = child / "external" / "dependency" / "leaf"
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_git(root: Path, *arguments: str) -> str:
        calls.append((root, arguments))
        if arguments[1] == "status":
            return f"-{'a' * 40} {arguments[-1]}\n"
        if root == clone.root:
            _manifest(
                child,
                "publisher.child",
                externals=[_pin("dependency.leaf", "publisher.leaf", "b" * 40)],
            )
        else:
            _manifest(leaf, "publisher.leaf")
        return ""

    visited: list[Path] = []

    def ignore_editor(_: Clone) -> None:
        pass

    def record(
        current: Clone, _definition: LocalDefinition, *, only_binary: bool
    ) -> int:
        del only_binary
        visited.append(current.root)
        return 0

    monkeypatch.setattr("verdog_runtime.cli.local.git", fake_git)
    monkeypatch.setattr("verdog_runtime.cli.sync._sync_editor_environment", ignore_editor)
    monkeypatch.setattr("verdog_runtime.cli.sync._sync_one", record)

    assert sync(clone) == 0
    assert visited == [clone.root, child, leaf]
    assert calls == [
        (
            clone.root,
            ("submodule", "status", "--", "external/dependency/child"),
        ),
        (
            clone.root,
            (
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--",
                "external/dependency/child",
            ),
        ),
        (
            child,
            ("submodule", "status", "--", "external/dependency/leaf"),
        ),
        (
            child,
            (
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--",
                "external/dependency/leaf",
            ),
        ),
    ]


def test_full_sync_does_not_touch_an_existing_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[
            _pin("dependency.missing", "publisher.missing", "a" * 40),
            _pin("dependency.existing", "publisher.existing", "b" * 40),
        ],
    )
    missing = clone.root / "external" / "dependency" / "missing"
    existing = clone.root / "external" / "dependency" / "existing"
    _manifest(existing, "publisher.existing")
    authored = existing / "dirty.txt"
    authored.write_text("unchanged", encoding="utf-8")
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_git(root: Path, *arguments: str) -> str:
        assert arguments[-1] != "external/dependency/existing"
        calls.append((root, arguments))
        if arguments[1] == "status":
            return f"-{'a' * 40} {arguments[-1]}\n"
        _manifest(missing, "publisher.missing")
        return ""

    def ignore_editor(_: Clone) -> None:
        pass

    def ignore_workflow(
        _clone: Clone, _definition: LocalDefinition, *, only_binary: bool
    ) -> int:
        del only_binary
        return 0

    monkeypatch.setattr("verdog_runtime.cli.local.git", fake_git)
    monkeypatch.setattr("verdog_runtime.cli.sync._sync_editor_environment", ignore_editor)
    monkeypatch.setattr("verdog_runtime.cli.sync._sync_one", ignore_workflow)

    assert sync(clone) == 0
    assert authored.read_text("utf-8") == "unchanged"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "status",
    [
        "",
        f"U{'a' * 40} external/dependency/child\n",
        f"+{'a' * 40} external/dependency/child\n",
    ],
)
def test_full_sync_does_not_repair_unrecognized_or_conflicted_checkouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[_pin("dependency.child", "publisher.child", "a" * 40)],
    )
    calls: list[tuple[str, ...]] = []

    def fake_git(_root: Path, *arguments: str) -> str:
        calls.append(arguments)
        return status

    def unexpected_editor(_: Clone) -> None:
        pytest.fail("environment creation preceded dependency validation")

    monkeypatch.setattr("verdog_runtime.cli.local.git", fake_git)
    monkeypatch.setattr("verdog_runtime.cli.sync._sync_editor_environment", unexpected_editor)

    with pytest.raises(WorkspaceError, match="is not an uninitialized submodule"):
        sync(clone)
    assert calls == [("submodule", "status", "--", "external/dependency/child")]


def test_full_sync_requires_a_manifest_after_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[_pin("dependency.child", "publisher.child", "a" * 40)],
    )
    calls: list[tuple[str, ...]] = []

    def fake_git(_root: Path, *arguments: str) -> str:
        calls.append(arguments)
        return f"-{'a' * 40} {arguments[-1]}\n" if arguments[1] == "status" else ""

    def unexpected_editor(_: Clone) -> None:
        pytest.fail("environment creation preceded dependency validation")

    monkeypatch.setattr("verdog_runtime.cli.local.git", fake_git)
    monkeypatch.setattr("verdog_runtime.cli.sync._sync_editor_environment", unexpected_editor)

    with pytest.raises(WorkspaceError, match="after submodule initialization"):
        sync(clone)
    assert [arguments[1] for arguments in calls] == ["status", "update"]


def test_targeted_sync_visits_only_one_workflow_in_the_current_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _subroutine(
        "main",
        workflows=[_workflow("main__child")],
        subroutines=[_subroutine("child")],
    )
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[_pin("missing.child", "publisher.child", "a" * 40)],
        subroutine=graph,
    )
    visited: list[str] = []

    def ignore_editor(_: Clone) -> None:
        pass

    def record(_: Clone, definition: LocalDefinition, *, only_binary: bool) -> int:
        visited.append(f"{definition.local_id}:{only_binary}")
        return 0

    monkeypatch.setattr("verdog_runtime.cli.sync._sync_editor_environment", ignore_editor)
    monkeypatch.setattr("verdog_runtime.cli.sync._sync_one", record)

    assert sync(clone, workflow_id="main__child", only_binary=True) == 0
    assert visited == ["main__child:True"]

    with pytest.raises(WorkspaceError, match="unknown workflow"):
        sync(clone, workflow_id="missing")


def test_external_subroutine_calls_link_source_without_merging_dependencies(
    tmp_path: Path,
) -> None:
    child_call: dict[str, Any] = {
        "id": "nested_call",
        "kind": "subroutine_call",
        "operation": {
            "target": "dependency.leaf/main__tool",
            "profile_arguments": {},
            "session_arguments": {},
        },
    }
    root_call: dict[str, Any] = {
        "id": "call",
        "kind": "subroutine_call",
        "operation": {
            "target": "dependency.child/main__shared",
            "profile_arguments": {},
            "session_arguments": {},
        },
    }
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[_pin("dependency.child", "publisher.child", "a" * 40)],
        subroutine=_subroutine("main", nodes=[root_call]),
    )
    child_clone = _manifest(
        clone.root / "external" / "dependency" / "child",
        "publisher.child",
        requirements=["must-not-be-installed"],
        externals=[_pin("dependency.leaf", "publisher.leaf", "b" * 40)],
        subroutine=_subroutine(
            "main", subroutines=[_subroutine("shared", nodes=[child_call])]
        ),
    )
    leaf_clone = _manifest(
        child_clone.root / "external" / "dependency" / "leaf",
        "publisher.leaf",
        requirements=["also-not-installed"],
        subroutine=_subroutine("main", subroutines=[_subroutine("tool")]),
    )

    root = _root_definition(clone)
    assert clone.workflow_requirements(root) == ()
    assert _subroutine_sources(clone, root) == (
        clone.root.resolve() / "src",
        child_clone.root.resolve() / "src",
        leaf_clone.root.resolve() / "src",
    )


def test_in_process_calls_cannot_load_two_checkouts_of_one_package(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = [
        {
            "id": f"call_{name}",
            "kind": "subroutine_call",
            "operation": {
                "target": f"{alias}/main__tool",
                "profile_arguments": {},
                "session_arguments": {},
            },
        }
        for name, alias in (("stable", "channel.stable"), ("canary", "channel.canary"))
    ]
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[
            _pin("channel.stable", "publisher.shared", "a" * 40),
            _pin("channel.canary", "publisher.shared", "b" * 40),
        ],
        subroutine=_subroutine("main", nodes=calls),
    )
    _manifest(
        clone.root / "external" / "channel" / "stable",
        "publisher.shared",
        subroutine=_subroutine("main", subroutines=[_subroutine("tool")]),
    )
    _manifest(
        clone.root / "external" / "channel" / "canary",
        "publisher.shared",
        subroutine=_subroutine("main", subroutines=[_subroutine("tool")]),
    )

    with pytest.raises(WorkspaceError, match="multiple in-process checkouts"):
        _subroutine_sources(clone, _root_definition(clone))


def test_sync_rejects_a_release_cycle_before_following_it_forever(
    tmp_path: Path,
) -> None:
    clone = _manifest(
        tmp_path / "root",
        "test.root",
        externals=[_pin("dependency.branch", "publisher.branch", "b" * 40)],
    )
    branch = clone.root / "external" / "dependency" / "branch"
    back_to_root = branch / "external" / "dependency" / "root"
    _manifest(
        branch,
        "publisher.branch",
        externals=[_pin("dependency.root", "test.root", "a" * 40)],
    )
    _manifest(
        back_to_root,
        "test.root",
        externals=[_pin("dependency.branch", "publisher.branch", "b" * 40)],
    )

    with pytest.raises(WorkspaceError, match="dependency cycle"):
        sync(clone)
