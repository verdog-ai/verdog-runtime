"""Give a clone the editor runtime and each workflow its declared dependencies.

Copy the installed runtime and its dependencies into each workflow environment, then let
pip resolve the workflow's own requirements. Unrelated client-environment packages stay out.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import venv
from importlib import metadata
from pathlib import Path
from typing import cast

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .local import (
    Clone,
    LocalDefinition,
    WorkspaceError,
    cleanup_staging,
    dependency_clones,
    external_root,
    interpreter_in,
    workspace_lock,
)

SOURCE_LINKS = "_verdog_sources.pth"
ENVIRONMENT_MARKER = ".verdog-environment.json"


class _EnvironmentInUse(WorkspaceError):
    """An installer may still be writing; restoring over it would lose recovery data."""


def _runtime_distribution() -> metadata.Distribution:
    try:
        return metadata.distribution("verdog-runtime")
    except metadata.PackageNotFoundError as error:
        raise WorkspaceError(
            "verdog_runtime is not installed; reinstall verdog-runtime"
        ) from error


def _runtime_identity(distribution: metadata.Distribution) -> dict[str, str]:
    """The installed runtime build, including same-version source changes."""

    files = distribution.files
    if files is None:
        raise WorkspaceError(
            "verdog_runtime has no installed file inventory; reinstall verdog-runtime"
        )

    digest = hashlib.sha256()
    for file in sorted(files, key=lambda item: item.as_posix()):
        if file.suffix == ".pyc" or any(
            part == "__pycache__" or part.endswith(".dist-info") for part in file.parts
        ):
            continue
        source = Path(str(distribution.locate_file(file)))
        digest.update(file.as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(source.read_bytes() if source.is_file() else b"missing")
        except OSError as error:
            raise WorkspaceError(
                "verdog_runtime cannot be inspected; reinstall verdog-runtime"
            ) from error
        digest.update(b"\0")
    return {"version": distribution.version, "sha256": digest.hexdigest()}


def _seed_distributions(site_packages: Path, requirements: tuple[str, ...], /) -> None:
    """Copy the named installed distributions and their dependency closure."""

    pending = list(requirements)
    seen: set[str] = set()
    root = site_packages.resolve()
    while pending:
        raw = pending.pop()
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as error:
            raise WorkspaceError(
                f"invalid verdog-runtime requirement {raw!r}"
            ) from error
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = requirement.name
        key = canonicalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as error:
            raise WorkspaceError(
                f"verdog-runtime dependency {name!r} is not installed; reinstall verdog-runtime"
            ) from error
        pending.extend(distribution.requires or ())
        for file in distribution.files or ():
            source = Path(str(distribution.locate_file(file)))
            target = (root / file).resolve()
            if source.is_file() and (target == root or root in target.parents):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def _site_packages(environment: Path) -> Path:
    candidates = sorted(
        (
            *environment.glob("lib/python3.*/site-packages"),
            *environment.glob("Lib/site-packages"),
        )
    )
    if not candidates:
        raise WorkspaceError(f"{environment} has no site-packages")
    site_packages = candidates[0].resolve()
    if not site_packages.is_relative_to(environment.resolve()):
        raise WorkspaceError(f"refusing site-packages outside {environment}")
    return site_packages


def _rebuild_directory(path: Path, build: Callable[[], int]) -> int:
    """Build at the final path (venv scripts embed it), retaining the old tree on failure."""

    if path.is_symlink():
        raise WorkspaceError(f"refusing to rebuild through symbolic link {path}")
    target = path.resolve()
    existed = target.exists()
    if existed and not target.is_dir():
        raise WorkspaceError(f"{target} is not a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".verdog-rebuild-"))
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
        raise WorkspaceError(
            f"{error}. Environment left at {target}; recover the previous tree from {staging} "
            "after stopping the installer."
        ) from error
    finally:
        try:
            if not keep_backup and status != 0 and (backup.exists() or not existed):
                if target.is_symlink():
                    raise WorkspaceError(
                        f"rebuild destination became a symbolic link: {target}"
                    )
                if target.exists():
                    shutil.rmtree(target)
                if backup.exists():
                    backup.replace(target)
        except BaseException as error:
            keep_backup = True
            raise WorkspaceError(
                f"environment recovery failed: {error}. "
                f"Recover the previous environment from {staging}"
            ) from error
        finally:
            if not keep_backup:
                cleanup_staging(staging)


def _sync_editor_environment(clone: Clone) -> None:
    """Refresh only editor packages; leave the editor environment's other files alone."""

    environment = clone.root / ".venv"
    with workspace_lock(clone.root, "sync-editor"):
        if environment.is_symlink():
            raise WorkspaceError(
                f"refusing to use environment through symbolic link {environment}"
            )
        created = not environment.exists()
        destination = environment if created else _site_packages(environment)

        def build() -> int:
            if created:
                venv.EnvBuilder(with_pip=False, symlinks=True).create(str(environment))
            else:
                destination.mkdir()
            _seed_distributions(_site_packages(environment), ("verdog-runtime",))
            return 0

        _rebuild_directory(destination, build)
    print(
        f"{'Created' if created else 'Refreshed'} .venv: "
        "verdog_runtime provisioned for the editor."
    )


def _environment_spec(
    clone: Clone, definition: LocalDefinition
) -> tuple[dict[str, object], tuple[str, ...], tuple[Path, ...]]:
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


def require_current_environment(clone: Clone, definition: LocalDefinition) -> Path:
    """Return a workflow's environment, or name the exact sync command it needs."""

    environment = clone.environment(definition)
    marker = environment / ENVIRONMENT_MARKER
    try:
        actual = json.loads(marker.read_text("utf-8"))
    except (OSError, ValueError):
        actual = None
    expected, _, _ = _environment_spec(clone, definition)
    if actual != expected:
        raise WorkspaceError(
            f"environment for workflow {definition.local_id!r} is missing or stale; "
            f"run `verdog sync {definition.local_id}`"
        )
    return environment


def sync(
    clone: Clone,
    *,
    workflow_id: str | None = None,
    only_binary: bool = False,
) -> int:
    """Refresh the editor runtime and every workflow's isolated environment."""

    if workflow_id is not None:
        definition = clone.workflow_definition(workflow_id)
        _sync_editor_environment(clone)
        return _sync_one(clone, definition, only_binary=only_binary)

    dependencies = dependency_clones(clone, initialize=True)
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
    clone: Clone,
    definition: LocalDefinition,
    *,
    only_binary: bool,
) -> int:
    """Create or refresh one workflow definition's environment.

    `only_binary` refuses source distributions, which is what makes this safe to run over code
    somebody else published. Installing an sdist executes its build backend; installing a wheel
    unpacks it. A reader browsing a workflow has consented to *reading* it, and a type checker
    never imports a package -- it reads source and stubs -- so wheels are enough to resolve
    every import without running a line of anyone's code.
    """

    # Resolve and validate the complete specification before clearing a usable environment.
    specification, declared, source_roots = _environment_spec(clone, definition)
    environment = clone.environment(definition)

    def build() -> int:
        # A clean rebuild keeps requirements authoritative, including removed packages.
        venv.EnvBuilder(with_pip=True, symlinks=True).create(str(environment))
        interpreter = interpreter_in(environment)
        if interpreter is None:
            raise WorkspaceError(f"no interpreter in {environment}")
        site_packages = _site_packages(environment)
        _seed_distributions(site_packages, ("verdog-runtime",))
        (site_packages / SOURCE_LINKS).write_text(
            "".join(f"{root}\n" for root in source_roots), encoding="utf-8"
        )
        runtime_requirements = cast(list[str], specification["runtime_requirements"])
        requirements = (*runtime_requirements, *declared)
        if requirements:
            if declared:
                print(f"Installing {len(declared)} declared dependency/dependencies...")
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
                    "verdog: the declared dependencies could not be installed. They are "
                    "yours to resolve -- fix the workflow's requirements.txt, then sync again.",
                    file=sys.stderr,
                )
                return status
        (environment / ENVIRONMENT_MARKER).write_text(
            json.dumps(specification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    with workspace_lock(clone.root, f"sync-{definition.local_id}"):
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


def _install(command: list[str], cwd: Path) -> int:
    """Wait for pip to stop before the surrounding environment can be restored."""

    with subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        command,
        cwd=cwd,
        start_new_session=sys.platform != "win32",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    ) as process:
        try:
            return process.wait()
        except BaseException:
            # A source build may have children still writing into this environment.
            try:
                try:
                    if sys.platform == "win32":
                        stopped = subprocess.run(  # noqa: S603,S607 - native process-tree termination
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        if stopped.returncode:
                            raise OSError("taskkill could not stop the installer tree")
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                finally:
                    process.kill()
                    process.wait()
            except BaseException as error:
                raise _EnvironmentInUse("could not confirm installer termination") from error
            raise


def _subroutine_sources(clone: Clone, definition: LocalDefinition) -> tuple[Path, ...]:
    """Own source plus every external subroutine source reached in-process."""

    found: list[Path] = [clone.root.resolve() / "src"]
    packages: dict[str, Path] = {}
    visited: set[tuple[Path, str]] = set()

    def reserve(owner: Clone) -> None:
        package = owner.project.get("package")
        if not isinstance(package, str) or not package:
            raise WorkspaceError(f"{owner.root / 'project.json'} has no package")
        physical = owner.root.resolve()
        previous = packages.setdefault(package, physical)
        if previous != physical:
            raise WorkspaceError(
                f"workflow {definition.id} imports package {package!r} from multiple "
                "in-process checkouts; use workflow_call for isolation"
            )

    def visit(owner: Clone, current: LocalDefinition) -> None:
        reserve(owner)
        key = (owner.root.resolve(), current.local_id)
        if key in visited:
            return
        visited.add(key)
        for alias, subroutine_id in owner.external_subroutine_targets(current):
            child_root = owner.root / external_root(alias)
            if not (child_root / "project.json").is_file():
                raise WorkspaceError(
                    f"{child_root} is not checked out; run "
                    "`git submodule update --init --recursive`"
                )
            child = Clone(child_root, owner.origin, owner.token)
            source = child.root.resolve() / "src"
            if source not in found:
                found.append(source)
            target = next(
                (
                    item
                    for item in child.local_definitions()
                    if item.kind == "subroutine" and item.local_id == subroutine_id
                ),
                None,
            )
            if target is None:
                raise WorkspaceError(
                    f"external subroutine {alias}/{subroutine_id} does not exist"
                )
            visit(child, target)

    visit(clone, definition)
    return tuple(found)
