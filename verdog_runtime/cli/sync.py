"""Prepare isolated workflow environments and recover interrupted installs."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import venv
from collections.abc import Callable
from typing import cast

import packaging.requirements
import packaging.utils

from verdog_runtime.cli import local

SOURCE_LINKS = "_verdog_sources.pth"
ENVIRONMENT_MARKER = ".verdog-environment.json"


class _EnvironmentInUse(local.WorkspaceError):
    """An installer may still be writing to the environment."""


def _runtime_distribution() -> importlib.metadata.Distribution:
    try:
        return importlib.metadata.distribution("verdog-runtime")
    except importlib.metadata.PackageNotFoundError as error:
        raise local.WorkspaceError(
            "verdog_runtime is not installed; reinstall verdog-runtime"
        ) from error


def _runtime_identity(
    distribution: importlib.metadata.Distribution,
) -> dict[str, str]:
    """The installed runtime build, including same-version source changes."""
    files = distribution.files
    if files is None:
        raise local.WorkspaceError(
            "verdog_runtime has no installed file inventory; "
            "reinstall verdog-runtime"
        )

    digest = hashlib.sha256()
    for file in sorted(files, key=lambda item: item.as_posix()):
        if file.suffix == ".pyc" or any(
            part == "__pycache__" or part.endswith(".dist-info")
            for part in file.parts
        ):
            continue
        source = pathlib.Path(str(distribution.locate_file(file)))
        digest.update(file.as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(
                source.read_bytes() if source.is_file() else b"missing"
            )
        except OSError as error:
            raise local.WorkspaceError(
                "verdog_runtime cannot be inspected; reinstall verdog-runtime"
            ) from error
        digest.update(b"\0")
    return {"version": distribution.version, "sha256": digest.hexdigest()}


def _seed_distributions(
    site_packages: pathlib.Path, requirements: tuple[str, ...], /
) -> None:
    """Copy the named installed distributions and their dependency closure."""
    pending = list(requirements)
    seen: set[str] = set()
    root = site_packages.resolve()
    while pending:
        raw = pending.pop()
        try:
            requirement = packaging.requirements.Requirement(raw)
        except packaging.requirements.InvalidRequirement as error:
            raise local.WorkspaceError(
                f"invalid verdog-runtime requirement {raw!r}"
            ) from error
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = requirement.name
        key = packaging.utils.canonicalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise local.WorkspaceError(
                f"verdog-runtime dependency "
                f"{name!r} is not installed; reinstall verdog-runtime"
            ) from error
        pending.extend(distribution.requires or ())
        for file in distribution.files or ():
            source = pathlib.Path(str(distribution.locate_file(file)))
            target = (root / file).resolve()
            if source.is_file() and (target == root or root in target.parents):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def _site_packages(environment: pathlib.Path) -> pathlib.Path:
    candidates = sorted(
        (
            *environment.glob("lib/python3.*/site-packages"),
            *environment.glob("Lib/site-packages"),
        )
    )
    if not candidates:
        raise local.WorkspaceError(f"{environment} has no site-packages")
    site_packages = candidates[0].resolve()
    if not site_packages.is_relative_to(environment.resolve()):
        raise local.WorkspaceError(
            f"refusing site-packages outside {environment}"
        )
    return site_packages


def _rebuild_directory(path: pathlib.Path, build: Callable[[], int]) -> int:
    """Build at the final path, retaining the previous tree on failure."""
    if path.is_symlink():
        raise local.WorkspaceError(
            f"refusing to rebuild through symbolic link {path}"
        )
    target = path.resolve()
    existed = target.exists()
    if existed and not target.is_dir():
        raise local.WorkspaceError(f"{target} is not a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(dir=target.parent, prefix=".verdog-rebuild-")
    )
    backup = staging / "previous"
    status = 1
    keep_backup = False
    try:
        if existed:
            target.replace(backup)
        status = build()
        return status
    except _EnvironmentInUse as error:
        keep_backup = True
        raise local.WorkspaceError(
            f"{error}. Environment left at "
            f"{target}; recover the previous tree from "
            f"{staging} "
            "after stopping the installer."
        ) from error
    finally:
        try:
            if (
                not keep_backup
                and status != 0
                and (backup.exists() or not existed)
            ):
                if target.is_symlink():
                    raise local.WorkspaceError(
                        f"rebuild destination became a symbolic link: {target}"
                    )
                if target.exists():
                    shutil.rmtree(target)
                if backup.exists():
                    backup.replace(target)
        except BaseException as error:
            keep_backup = True
            raise local.WorkspaceError(
                f"environment recovery failed: {error}. "
                f"Recover the previous environment from {staging}"
            ) from error
        finally:
            if not keep_backup:
                local.cleanup_staging(staging)


def _sync_editor_environment(clone: local.Clone) -> None:
    """Refresh editor packages without replacing unrelated files."""
    environment = clone.root / ".venv"
    with local.workspace_lock(clone.root, "sync-editor"):
        if environment.is_symlink():
            raise local.WorkspaceError(
                f"refusing to use environment through symbolic "
                f"link "
                f"{environment}"
            )
        created = not environment.exists()
        destination = environment if created else _site_packages(environment)

        def build() -> int:
            if created:
                venv.EnvBuilder(with_pip=False, symlinks=True).create(
                    str(environment)
                )
            else:
                destination.mkdir()
            _seed_distributions(
                _site_packages(environment), ("verdog-runtime",)
            )
            return 0

        _rebuild_directory(destination, build)
    print(
        f"{'Created' if created else 'Refreshed'} .venv: "
        "verdog_runtime provisioned for the editor."
    )


def _environment_spec(
    clone: local.Clone, definition: local.LocalDefinition
) -> tuple[dict[str, object], tuple[str, ...], tuple[pathlib.Path, ...]]:
    """Everything whose change requires rebuilding one workflow environment."""
    declared = clone.workflow_requirements(definition)
    runtime = _runtime_distribution()
    sources = _subroutine_sources(clone, definition)
    return (
        {
            "python": sysconfig.get_python_version(),
            "requirements": sorted(declared),
            "runtime": _runtime_identity(runtime),
            "runtime_requirements": sorted(runtime.requires or ()),
            "source_roots": sorted(str(path.resolve()) for path in sources),
        },
        declared,
        sources,
    )


def require_current_environment(
    clone: local.Clone, definition: local.LocalDefinition
) -> pathlib.Path:
    """Return the environment or explain which sync command is needed."""
    environment = clone.environment(definition)
    marker = environment / ENVIRONMENT_MARKER
    try:
        actual = json.loads(marker.read_text("utf-8"))
    except (OSError, ValueError):
        actual = None
    expected, _, _ = _environment_spec(clone, definition)
    if actual != expected:
        raise local.WorkspaceError(
            f"environment for workflow "
            f"{definition.local_id!r} is missing or stale; "
            f"run `verdog sync {definition.local_id}`"
        )
    return environment


def sync(
    clone: local.Clone,
    *,
    workflow_id: str | None = None,
    only_binary: bool = False,
) -> int:
    """Refresh the editor runtime and every workflow's isolated environment."""
    if workflow_id is not None:
        definition = clone.workflow_definition(workflow_id)
        _sync_editor_environment(clone)
        return _sync_one(clone, definition, only_binary=only_binary)

    dependencies = local.dependency_clones(clone, initialize=True)
    _sync_editor_environment(clone)
    for index, (_, dependency) in enumerate(dependencies):
        for definition in dependency.workflow_definitions():
            status = _sync_one(
                dependency,
                definition,
                only_binary=only_binary if index == 0 else True,
            )
            if status:
                return status
    return 0


def _sync_one(
    clone: local.Clone,
    definition: local.LocalDefinition,
    *,
    only_binary: bool,
) -> int:
    """Create or refresh one workflow definition's environment.

    `only_binary` refuses source distributions, which is what makes this
    safe to run over code somebody else published. Installing an sdist
    executes its build backend; installing a wheel unpacks it. A reader
    browsing a workflow has consented to *reading* it, and a type checker
    never imports a package -- it reads source and stubs -- so wheels are
    enough to resolve every import without running a line of anyone's code.
    """
    # Resolve and validate the complete specification before clearing a usable
    # environment.
    specification, declared, source_roots = _environment_spec(clone, definition)
    environment = clone.environment(definition)

    def build() -> int:
        # A clean rebuild keeps requirements authoritative, including removed
        # packages.
        venv.EnvBuilder(with_pip=True, symlinks=True).create(str(environment))
        interpreter = local.interpreter_in(environment)
        if interpreter is None:
            raise local.WorkspaceError(f"no interpreter in {environment}")
        site_packages = _site_packages(environment)
        _seed_distributions(site_packages, ("verdog-runtime",))
        (site_packages / SOURCE_LINKS).write_text(
            "".join(f"{root}\n" for root in source_roots), encoding="utf-8"
        )
        runtime_requirements = cast(
            list[str], specification["runtime_requirements"]
        )
        requirements = (*runtime_requirements, *declared)
        if requirements:
            if declared:
                print(
                    f"Installing "
                    f"{len(declared)} declared dependency/dependencies..."
                )
                sys.stdout.flush()
            status = _install(
                [
                    str(interpreter),
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--disable-pip-version-check",
                    *(["--only-binary", ":all:"] if only_binary else []),
                    *requirements,
                ],
                clone.root,
            )
            if status:
                print(
                    "verdog: the declared dependencies could not be "
                    "installed. They are "
                    "yours to resolve -- fix the workflow's "
                    "requirements.txt, then sync again.",
                    file=sys.stderr,
                )
                return status
        (environment / ENVIRONMENT_MARKER).write_text(
            json.dumps(specification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    with local.workspace_lock(clone.root, f"sync-{definition.local_id}"):
        created = not environment.exists()
        status = _rebuild_directory(environment, build)
    if status:
        return status

    print(
        f"{'Created' if created else 'Refreshed'} "
        f"{environment.relative_to(clone.root)} "
        f"({sysconfig.get_python_version()}): verdog_runtime provisioned, "
        f"{len(declared)} declared dependency/dependencies"
        f"{' (wheels only)' if only_binary else ''}."
    )
    return 0


def _install(command: list[str], cwd: pathlib.Path) -> int:
    """Wait for all installer processes before allowing recovery."""
    with subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        command,
        cwd=cwd,
        start_new_session=sys.platform != "win32",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.platform == "win32"
        else 0,
    ) as process:
        try:
            return process.wait()
        except BaseException as interrupted:
            # A source build may have children still writing into this
            # environment.
            try:
                try:
                    if sys.platform == "win32":
                        stopped = subprocess.run(  # noqa: S603,S607
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        if stopped.returncode:
                            raise OSError(
                                "taskkill could not stop the installer tree"
                            ) from interrupted
                    else:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                finally:
                    process.kill()
                    process.wait()
            except BaseException as error:
                raise _EnvironmentInUse(
                    "could not confirm installer termination"
                ) from error
            raise


def _subroutine_sources(
    clone: local.Clone, definition: local.LocalDefinition
) -> tuple[pathlib.Path, ...]:
    """Own source plus every external subroutine source reached in-process."""
    found: list[pathlib.Path] = [clone.root.resolve() / "src"]
    packages: dict[str, pathlib.Path] = {}
    visited: set[tuple[pathlib.Path, str]] = set()

    def reserve(owner: local.Clone) -> None:
        package = owner.project.get("package")
        if not isinstance(package, str) or not package:
            raise local.WorkspaceError(
                f"{owner.root / 'project.json'} has no package"
            )
        physical = owner.root.resolve()
        previous = packages.setdefault(package, physical)
        if previous != physical:
            raise local.WorkspaceError(
                f"workflow "
                f"{definition.id} imports package "
                f"{package!r} from multiple "
                "in-process checkouts; use workflow_call for isolation"
            )

    def visit(owner: local.Clone, current: local.LocalDefinition) -> None:
        reserve(owner)
        key = (owner.root.resolve(), current.local_id)
        if key in visited:
            return
        visited.add(key)
        for alias, subroutine_id in owner.external_subroutine_targets(current):
            child_root = owner.root / local.external_root(alias)
            if not (child_root / "project.json").is_file():
                raise local.WorkspaceError(
                    f"{child_root} is not checked out; run "
                    "`git submodule update --init --recursive`"
                )
            child = local.Clone(child_root, owner.origin, owner.token)
            source = child.root.resolve() / "src"
            if source not in found:
                found.append(source)
            target = next(
                (
                    item
                    for item in child.local_definitions()
                    if item.kind == "subroutine"
                    and item.local_id == subroutine_id
                ),
                None,
            )
            if target is None:
                raise local.WorkspaceError(
                    f"external subroutine "
                    f"{alias}/"
                    f"{subroutine_id} does not exist"
                )
            visit(child, target)

    visit(clone, definition)
    return tuple(found)
