"""The working copy: git, and which of its files make up the project.

A clone is an ordinary GitHub clone. Verdog does not make its commits, does not track its
history, and holds no credential for it -- git already has one. What Verdog keeps beside it
is one small file naming the service, and optionally a token an agent can use.

The project manifest declares its sources and pinned dependencies under `external/`.
`files()` sends those declared files, including the dependency closure, because the service
stores nothing between calls. Unlisted repository files are not project inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import json
import keyword
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Literal, cast

from .._file_lock import locked_file
from ..declarations.ids import is_valid_definition_id

from .requirements import parse_requirements

CONFIG_PATH = "verdog.json"
"""Where a clone records which service it talks to, inside `.git` so it is never content."""

_REMOTE = re.compile(
    r"(?:git@[^:]+:|(?:ssh|https?)://[^/]+/)(?P<owner>[^/]+)/(?P<name>[^/]+?)(?:\.git)?/?\Z"
)
_REQUIREMENTS_FILE = "requirements.txt"

EXTERNAL_ROOT = "external"
"""Where a pinned dependency is checked out, one directory per pin.

Shared by dependency paths, the local type-checking exclusion, and `drop`'s cleanup."""

SCHEMA_VERSION = 32

_DEFINITION_SEPARATOR = "__"


def definition_child(owner: str | None, leaf: str, /) -> str:
    return leaf if owner is None else f"{owner}{_DEFINITION_SEPARATOR}{leaf}"


def definition_leaf(identifier: str, /) -> str:
    return identifier.rsplit(_DEFINITION_SEPARATOR, 1)[-1]


def package_problem(package: str) -> str | None:
    """Why a canonical `<space>.<name>` package is invalid, if it is."""

    parts = package.split(".")
    if len(parts) != 2:
        return "a package is `<space>.<name>`, with one separator"
    for part in parts:
        if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", part) is None:
            return (
                f"{part!r} is not a lowercase snake_case identifier "
                "(letters, digits and single underscores, starting with a letter)"
            )
        if keyword.iskeyword(part) or keyword.issoftkeyword(part):
            return f"{part!r} is a Python keyword"
    return None


class WorkspaceError(Exception):
    """The working copy is missing, misconfigured, or git refused."""


@contextmanager
def workspace_lock(root: Path, name: str) -> Generator[None]:
    """Serialize cooperating local writers; closing the file releases the OS lock."""

    path = _direct_project_path(root, f".verdog/locks/{name}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        with locked_file(lock, blocking=True):
            yield


@dataclass(frozen=True, slots=True)
class _FileChange:
    path: str
    target: Path
    before: bytes | None
    after: bytes | None
    backup: Path
    staged: Path


def _file_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def cleanup_staging(path: Path) -> None:
    """Cleanup cannot turn an already committed or recovered transaction into failure."""

    try:
        shutil.rmtree(path)
    except OSError as error:
        print(f"verdog: temporary files remain at {path}: {error}", file=sys.stderr)


def _prune_empty_parents(path: Path, root: Path) -> None:
    parent = path.parent
    while parent != root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _object(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _objects(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        cast(dict[str, Any], item)
        for item in cast(list[object], value)
        if isinstance(item, dict)
    )


def _direct_project_path(root: Path, relative: str) -> Path:
    """Resolve one service path without permitting syntax or symlinks to leave the clone."""

    pure = PurePosixPath(relative)
    if (
        not relative
        or "\0" in relative
        or "\\" in relative
        or not pure.parts
        or pure.is_absolute()
        or PureWindowsPath(relative).anchor
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise WorkspaceError(f"unsafe project path: {relative or '<empty>'}")

    owner = root.resolve()
    target = contained_path(root, Path(*pure.parts))
    current = owner
    for index, part in enumerate(pure.parts):
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        except OSError as error:
            raise WorkspaceError(
                f"project path cannot be inspected: {relative}"
            ) from error
        final = index == len(pure.parts) - 1
        if (final and not stat.S_ISREG(mode)) or (not final and not stat.S_ISDIR(mode)):
            raise WorkspaceError(f"project path is not a regular file path: {relative}")
    return target


@dataclass(frozen=True, slots=True)
class LocalDefinition:
    """One generated definition owned by a clone."""

    id: str
    kind: Literal["workflow", "subroutine"]
    body: dict[str, Any]
    module: str

    @property
    def local_id(self) -> str:
        return self.id.rsplit(".", 1)[-1]

    @property
    def source(self) -> Path:
        return Path("src") / Path(*self.module.split("."))


@dataclass(frozen=True, slots=True)
class Clone:
    root: Path
    origin: str
    token: str | None

    @property
    def project(self) -> dict[str, Any]:
        try:
            decoded = json.loads((self.root / "project.json").read_text("utf-8"))
        except (OSError, ValueError) as error:
            raise WorkspaceError("project.json is missing or unreadable") from error
        if not isinstance(decoded, dict):
            raise WorkspaceError("project.json is not an object")
        project = cast(dict[str, Any], decoded)
        version = project.get("schema_version")
        if version != SCHEMA_VERSION:
            raise WorkspaceError(
                f"project schema v{version} is unsupported; this CLI requires schema v{SCHEMA_VERSION}"
            )
        return project

    @property
    def generated_from(self) -> str:
        """The graph this tree was generated from, as `project.json` records it.

        Reported rather than recomputed; callers that need staleness ask the compiler.
        """

        value = self.project.get("generated_from")
        if not isinstance(value, str):
            raise WorkspaceError("project.json has no generated_from")
        return value

    def repository(self) -> tuple[str, str] | None:
        """`owner/name` from git's own remote, or `None` for a clone with no remote.

        Read from git rather than recorded by us, because git is where it can change: a
        rename or a transfer on GitHub is `git remote set-url`, and a second copy of the
        address would then be wrong with nothing to notice it.

        `None` is a legitimate state -- a `git init`-ed directory with no remote is exactly
        how a new project starts; the service accepts files without a GitHub repository.
        """

        for remote in ("origin", "upstream"):
            try:
                url = self._git("remote", "get-url", remote).strip()
            except WorkspaceError:
                continue
            match = _REMOTE.search(url)
            if match is not None:
                return match.group("owner"), match.group("name")
        return None

    def require_repository(self) -> tuple[str, str]:
        found = self.repository()
        if found is None:
            raise WorkspaceError(
                "this clone has no GitHub remote; `git remote add origin ...` first"
            )
        return found

    def files(self) -> dict[str, str]:
        """Declared sources and manifests for this project and its pinned dependencies."""

        files: dict[str, str] = {}
        for _, owner in dependency_clones(self):
            prefix = owner.root.relative_to(self.root)
            _direct_project_path(owner.root, "project.json")
            files[(prefix / "project.json").as_posix()] = owner._read("project.json")
            sources = owner.project.get("sources")
            if not isinstance(sources, list):
                raise WorkspaceError(f"{owner.root / 'project.json'} has no sources")
            for source in _objects(cast(list[object], sources)):
                relative = source.get("path")
                if not isinstance(relative, str):
                    continue
                path = Path(relative)
                if (
                    not path.parts
                    or path.parts[0] == EXTERNAL_ROOT
                    or any(
                        part in {".git", ".venv", ".verdog", "__pycache__"}
                        for part in path.parts
                    )
                    or path.suffix == ".pyc"
                ):
                    continue
                target = _direct_project_path(owner.root, relative)
                if not target.is_file() and (
                    owner != self or source.get("ownership") == "user"
                ):
                    continue
                files[(prefix / path).as_posix()] = owner._read(relative)
        return files

    def graph_files(self) -> dict[str, str]:
        """Saved manifests only; the service resolves and validates their closure."""

        try:
            root = contained_path(self.root, Path("project.json"))
            files = {"project.json": root.read_text(encoding="utf-8")}
        except (OSError, UnicodeError) as error:
            raise WorkspaceError("project.json is missing or unreadable") from error
        external = contained_path(self.root, Path(EXTERNAL_ROOT))
        for path in sorted(external.rglob("project.json")):
            relative = path.relative_to(self.root)
            if any(part in {".git", ".venv", ".verdog"} for part in relative.parts):
                continue
            try:
                files[relative.as_posix()] = contained_path(self.root, relative).read_text(
                    encoding="utf-8"
                )
            except (OSError, UnicodeError, WorkspaceError):
                # An unreadable dependency must not hide the root graph's analysis.
                continue
        return files

    def write(
        self, files: Mapping[str, str | None], *, expected: Mapping[str, str]
    ) -> list[str]:
        """Commit a projection only against its input; roll back handled failures."""

        try:
            with workspace_lock(self.root, "projection"):
                return self._write_projection(files, expected)
        except (OSError, UnicodeError) as error:
            raise WorkspaceError(
                f"project files could not be written safely: {error}"
            ) from error

    def _write_projection(
        self, files: Mapping[str, str | None], expected: Mapping[str, str]
    ) -> list[str]:
        root = self.root.resolve()
        targets = {path: _direct_project_path(root, path) for path in files}

        def require_current() -> None:
            if self.files() != expected:
                raise WorkspaceError(
                    "project changed during projection; no stale result was applied. "
                    "Retry the command."
                )

        require_current()
        staging = Path(tempfile.mkdtemp(dir=root / ".verdog", prefix="write-"))
        changes: list[_FileChange] = []
        attempted: list[_FileChange] = []
        keep_backup = False
        try:
            # Sources first, obsolete paths next, the manifest last.
            order = sorted(
                files,
                key=lambda path: (path == "project.json", files[path] is None, path),
            )
            for index, path in enumerate(order):
                target = targets[path]
                before = _file_bytes(target)
                if before is not None and path not in expected:
                    raise WorkspaceError(
                        f"projection destination already exists: {path}; no files were changed"
                    )
                content = files[path]
                if (content is None and before is None) or (
                    content is not None
                    and before is not None
                    and before == content.encode("utf-8")
                ):
                    continue
                change = _FileChange(
                    path,
                    target,
                    before,
                    None if content is None else content.encode("utf-8"),
                    staging / f"{index}.before",
                    staging / f"{index}.after",
                )
                if before is not None:
                    change.backup.write_bytes(before)
                    shutil.copystat(target, change.backup)
                if change.after is not None:
                    change.staged.write_bytes(change.after)
                    if before is not None:
                        shutil.copymode(target, change.staged)
                changes.append(change)
            require_current()
            for change in changes:
                _direct_project_path(root, change.path)
                if _file_bytes(change.target) != change.before:
                    raise WorkspaceError(
                        f"projection destination changed: {change.path}; retry the command"
                    )
                attempted.append(change)
                if change.after is None:
                    change.target.unlink()
                else:
                    change.target.parent.mkdir(parents=True, exist_ok=True)
                    change.staged.replace(change.target)
        except BaseException as error:
            failures: list[str] = []
            for change in reversed(attempted):
                try:
                    _direct_project_path(root, change.path)
                    current = _file_bytes(change.target)
                    if current == change.before:
                        continue
                    if current != change.after:
                        raise WorkspaceError("file changed outside this projection")
                    if change.before is None:
                        change.target.unlink()
                    else:
                        shutil.copy2(change.backup, change.staged)
                        change.staged.replace(change.target)
                except BaseException as recovery_error:
                    failures.append(f"{change.path}: {recovery_error}")
            if failures:
                keep_backup = True
                raise WorkspaceError(
                    f"projection failed ({error}); rollback incomplete. "
                    f"Recover originals from {staging}: "
                    + "; ".join(failures)
                ) from error
            for change in attempted:
                if change.before is None:
                    _prune_empty_parents(change.target, root)
            raise
        finally:
            if not keep_backup:
                cleanup_staging(staging)

        for change in changes:
            if change.after is not None:
                continue
            # Cache/directory cleanup is not part of the source transaction.
            cache = change.target.parent / "__pycache__"
            if (
                change.target.suffix == ".py"
                and cache.is_dir()
                and not cache.is_symlink()
            ):
                try:
                    for compiled in cache.glob(f"{change.target.stem}.*.pyc"):
                        compiled.unlink()
                    cache.rmdir()
                except OSError:
                    pass
            _prune_empty_parents(change.target, root)
        return sorted(change.path for change in changes)

    def local_definitions(self) -> tuple[LocalDefinition, ...]:
        """Every workflow envelope and subroutine graph in the current schema."""

        project = self.project
        package = project.get("package")
        root_workflow = project.get("workflow")
        root_subroutine = project.get("subroutine")
        if (
            not isinstance(package, str)
            or not package
            or package_problem(package) is not None
            or not isinstance(root_workflow, dict)
            or not isinstance(root_subroutine, dict)
        ):
            raise WorkspaceError("project.json has no root workflow and subroutine")

        definitions: list[LocalDefinition] = []
        identifiers: set[tuple[Literal["workflow", "subroutine"], str]] = set()

        def objects(value: object, field: str) -> Iterator[dict[str, Any]]:
            if not isinstance(value, list):
                raise WorkspaceError(f"a definition has no {field} list")
            for item in cast(list[object], value):
                if not isinstance(item, dict):
                    raise WorkspaceError(f"a definition has an invalid {field} entry")
                yield cast(dict[str, Any], item)

        def add(
            raw: dict[str, Any],
            module: str,
            kind: Literal["workflow", "subroutine"],
            local_id: object,
        ) -> None:
            if (
                not isinstance(local_id, str)
                or not is_valid_definition_id(local_id)
            ):
                raise WorkspaceError("a local definition has no valid id")
            definition_id = f"{package}.{local_id}"
            key = (kind, definition_id)
            if key in identifiers:
                raise WorkspaceError(f"duplicate {kind} definition id {definition_id}")
            identifiers.add(key)
            definitions.append(LocalDefinition(definition_id, kind, raw, module))

        def visit_subroutine(
            raw: dict[str, Any], module: str, owner: str | None
        ) -> None:
            leaf = raw.get("id")
            if not isinstance(leaf, str) or not leaf or _DEFINITION_SEPARATOR in leaf:
                raise WorkspaceError("a local subroutine has no id")
            identifier = definition_child(owner, leaf)
            add(raw, module, "subroutine", identifier)
            for child in objects(raw.get("workflows"), "workflows"):
                if "external" in child:
                    continue
                target = child.get("subroutine")
                if not isinstance(target, str) or not target:
                    raise WorkspaceError("a local workflow has no subroutine")
                child_id = definition_child(identifier, definition_leaf(target))
                add(
                    child,
                    f"{module}.workflows.{definition_leaf(target)}",
                    "workflow",
                    child_id,
                )
            for child in objects(raw.get("subroutines"), "subroutines"):
                child_id = child.get("id")
                if not isinstance(child_id, str) or not child_id:
                    raise WorkspaceError("a local subroutine has no id")
                visit_subroutine(
                    child,
                    f"{module}.subroutines.{child_id}",
                    identifier,
                )

        workflow_body = cast(dict[str, Any], root_workflow)
        workflow_id = workflow_body.get("subroutine")
        subroutine_body = cast(dict[str, Any], root_subroutine)
        subroutine_id = subroutine_body.get("id")
        if not isinstance(subroutine_id, str) or not subroutine_id:
            raise WorkspaceError("the root subroutine has no id")
        add(
            workflow_body,
            f"{package}.workflows.{workflow_id}",
            "workflow",
            workflow_id,
        )
        visit_subroutine(
            subroutine_body,
            f"{package}.subroutines.{subroutine_id}",
            None,
        )
        return tuple(definitions)

    def workflow_definitions(self) -> tuple[LocalDefinition, ...]:
        return tuple(
            definition
            for definition in self.local_definitions()
            if definition.kind == "workflow"
        )

    def workflow_definition(self, id: str | None = None, /) -> LocalDefinition:
        workflows = self.workflow_definitions()
        if not workflows:
            raise WorkspaceError("project.json has no workflow")
        if id is None:
            return workflows[0]
        found = next((item for item in workflows if item.local_id == id), None)
        if found is None:
            available = ", ".join(item.local_id for item in workflows)
            raise WorkspaceError(f"unknown workflow {id!r}; choose one of: {available}")
        return found

    def workflow_requirements(self, definition: LocalDefinition) -> tuple[str, ...]:
        """The PEP 508 requirements owned by one workflow envelope."""

        if definition.kind != "workflow":
            raise WorkspaceError(f"definition {definition.id} is not a workflow")
        relative = (definition.source / _REQUIREMENTS_FILE).as_posix()
        try:
            return parse_requirements(self._read(relative), relative)
        except ValueError as error:
            raise WorkspaceError(str(error)) from error

    def subroutine_for(self, workflow: LocalDefinition) -> LocalDefinition:
        target = workflow.body.get("subroutine")
        for definition in self.local_definitions():
            if definition.kind == "subroutine" and definition.local_id == target:
                return definition
        raise WorkspaceError(
            f"workflow {workflow.id} references unknown subroutine {target!r}"
        )

    def subroutine_closure(
        self, definition: LocalDefinition
    ) -> tuple[LocalDefinition, ...]:
        """Subroutines reached from one definition, excluding workflow calls."""

        definitions = {
            item.local_id: item
            for item in self.local_definitions()
            if item.kind == "subroutine"
        }
        found: list[LocalDefinition] = []
        root = (
            self.subroutine_for(definition)
            if definition.kind == "workflow"
            else definition
        )
        seen: set[str] = {root.local_id}

        def visit(current: LocalDefinition) -> None:
            nodes = current.body.get("nodes")
            if not isinstance(nodes, list):
                raise WorkspaceError(f"definition {current.id} has no nodes list")
            for raw_node in cast(list[object], nodes):
                if not isinstance(raw_node, dict):
                    continue
                node = cast(dict[str, Any], raw_node)
                if node.get("kind") != "subroutine_call":
                    continue
                operation = node.get("operation")
                target = (
                    cast(dict[str, Any], operation).get("target")
                    if isinstance(operation, dict)
                    else None
                )
                if isinstance(target, str) and "/" in target:
                    continue
                child = definitions.get(str(target))
                if child is None:
                    raise WorkspaceError(
                        f"definition {current.id} calls unknown subroutine {target!r}"
                    )
                if child.local_id in seen:
                    continue
                seen.add(child.local_id)
                found.append(child)
                visit(child)

        visit(root)
        return (root, *found)

    def external_subroutine_targets(
        self, definition: LocalDefinition
    ) -> tuple[tuple[str, str], ...]:
        """External source trees required by the workflow's in-process graph closure."""

        found: list[tuple[str, str]] = []
        for subroutine in self.subroutine_closure(definition):
            for node in _objects(subroutine.body.get("nodes")):
                if node.get("kind") != "subroutine_call":
                    continue
                target = _object(node.get("operation")).get("target")
                if not isinstance(target, str) or "/" not in target:
                    continue
                alias, _, child_id = target.partition("/")
                pair = (alias, child_id)
                if alias and child_id and pair not in found:
                    found.append(pair)
        return tuple(found)

    def definition_sources(self, definition: LocalDefinition) -> tuple[str, ...]:
        """Files belonging to a workflow and its in-process subroutine closure."""

        owners = (definition, *self.subroutine_closure(definition))
        raw_sources = self.project.get("sources")
        sources = (
            cast(list[object], raw_sources) if isinstance(raw_sources, list) else []
        )
        found: list[str] = []
        for owner in owners:
            prefix = owner.source.as_posix() + "/"
            for raw_source in sources:
                if not isinstance(raw_source, dict):
                    continue
                path = cast(dict[str, Any], raw_source).get("path")
                if not isinstance(path, str) or not path.startswith(prefix):
                    continue
                if Path(path).suffix not in {".py", ".pyi"}:
                    continue
                relative = Path(path.removeprefix(prefix))
                if relative.parts and relative.parts[0] in {"workflows", "subroutines"}:
                    continue
                if path not in found:
                    found.append(path)
        return tuple(found)

    def environment(self, definition: LocalDefinition) -> Path:
        relative = Path(".verdog") / "environments" / definition.local_id
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise WorkspaceError(
                    f"refusing to use environment through symbolic link {current}"
                )
        root = self.root.resolve()
        target = (self.root / relative).resolve()
        if not target.is_relative_to(root):
            raise WorkspaceError(
                f"environment for {definition.id} escapes this project"
            )
        return self.root / relative

    def site_packages(self, definition: LocalDefinition) -> Path | None:
        """Where `sync` put one definition's dependencies, if it has been run."""

        candidates = sorted(
            self.environment(definition).glob("lib/python3.*/site-packages")
        )
        return candidates[0] if candidates else None

    def type_check_commands(
        self, *, output_format: str = "concise"
    ) -> tuple[list[str], ...]:
        """How to type-check each workflow in its own environment.

        `output_format` is `concise` for a person and `gitlab` for `verdog check --json`:
        `ty` has no `json` format, and of the ones it does have, GitLab's code-quality shape
        is the only structured one -- an array of objects with a path, a line, a column and
        a rule name, which is exactly what an editor needs to place a squiggle.

        The local check is the authoritative one for types: the service verifies *structure*
        -- that the graph projects and the contract chains agree -- and never installs a
        user's dependencies, so only this machine can say whether the code type-checks.
        """

        executable = shutil.which("ty") or (
            str(sibling)
            if (sibling := Path(sys.executable).with_name("ty")).is_file()
            else None
        )
        if executable is None:
            raise WorkspaceError("`ty` is unavailable; reinstall the verdog CLI")
        from .sync import require_current_environment

        commands: list[list[str]] = []
        for index, definition in enumerate(self.workflow_definitions()):
            environment = require_current_environment(self, definition)
            command = [
                executable,
                "check",
                "--output-format",
                output_format,
                "--python",
                str(environment),
                "--error",
                "all",
                "--exclude",
                f"{EXTERNAL_ROOT}/",
                *self.definition_sources(definition),
            ]
            if index == 0:
                command.extend(
                    path
                    for source in _objects(self.project.get("sources"))
                    if isinstance(path := source.get("path"), str)
                    and path.startswith("tests/")
                    and Path(path).suffix in {".py", ".pyi"}
                )
            commands.append(command)
        return tuple(commands)

    def _read(self, path: str) -> str:
        try:
            return self.root.joinpath(*path.split("/")).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise WorkspaceError(f"{path} could not be read as UTF-8") from error

    def _git(self, *arguments: str) -> str:
        return git(self.root, *arguments)


def package_directory(package: str) -> str:
    """A package as a directory path: `alice.tools` -> `alice/tools`.

    Mirrors `verdog::compiler::layout::package_directory` without importing the compiler
    into the working copy model. The mirror is one expression on purpose -- there is
    nothing here to drift except the choice of separator.

    The canonical package's dot is the module and directory separator.
    """

    return package.replace(".", "/")


def external_root(alias: str) -> str:
    """Where a direct pin is checked out, relative to the clone that owns its alias."""

    return f"{EXTERNAL_ROOT}/{package_directory(alias)}"


def interpreter_in(environment: Path, /) -> Path | None:
    for relative in ("bin/python", "Scripts/python.exe"):
        candidate = environment / relative
        if candidate.is_file():
            return candidate
    return None


def contained_path(root: Path, relative: Path, /) -> Path:
    """Resolve one relative path without traversing a symlink or leaving its owner."""

    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise WorkspaceError(f"path {relative} escapes {root}")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise WorkspaceError(f"path {relative} crosses a symbolic link")
    owner = root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(owner):
        raise WorkspaceError(f"path {relative} escapes {root}")
    return target


def dependency_clones(
    root: Clone, /, *, initialize: bool = False
) -> tuple[tuple[dict[str, Any] | None, Clone], ...]:
    """The root and its dependency closure, with one traversal owning graph checks."""

    found: list[tuple[dict[str, Any] | None, Clone]] = [(None, root)]
    seen = {root.root.resolve()}
    active: set[tuple[str, str, str]] = set()

    def visit(clone: Clone) -> None:
        aliases: set[str] = set()
        for pin in _objects(clone.project.get("externals")):
            alias = pin.get("alias")
            if not isinstance(alias, str) or not alias:
                raise WorkspaceError(
                    f"{clone.root / 'project.json'} has an external with no alias"
                )
            if alias in aliases:
                raise WorkspaceError(
                    f"{clone.root / 'project.json'} repeats external alias {alias!r}"
                )
            aliases.add(alias)
            release = (
                str(pin.get("owner", "")),
                str(pin.get("name", "")),
                str(pin.get("commit", "")),
            )
            if release in active:
                raise WorkspaceError(
                    f"external dependency cycle reaches {release[0]}/{release[1]}"
                    f"@{release[2][:12]}"
                )
            child_root = clone.root / external_root(alias)
            external = (clone.root / EXTERNAL_ROOT).resolve()
            physical = contained_path(clone.root, Path(external_root(alias)))
            if external not in physical.parents:
                raise WorkspaceError(
                    f"external alias {alias!r} escapes {clone.root / EXTERNAL_ROOT}"
                )
            manifest = child_root / "project.json"
            if not manifest.is_file() and initialize:
                relative = child_root.relative_to(clone.root).as_posix()
                status = git(clone.root, "submodule", "status", "--", relative)
                if status.startswith("-"):
                    git_with_ssh_fallback(
                        clone.root,
                        "submodule",
                        "update",
                        "--init",
                        "--recursive",
                        "--",
                        relative,
                    )
                else:
                    raise WorkspaceError(
                        f"{child_root} has no project.json and is not an "
                        "uninitialized submodule"
                    )
            if not manifest.is_file():
                action = (
                    "has no project.json after submodule initialization"
                    if initialize
                    else "is not checked out; run `git submodule update --init --recursive`"
                )
                raise WorkspaceError(f"{child_root} {action}")
            if physical in seen:
                continue
            seen.add(physical)
            child = Clone(child_root, clone.origin, clone.token)
            found.append((pin, child))
            active.add(release)
            visit(child)
            active.remove(release)

    visit(root)
    return tuple(found)


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        operation = (
            arguments[2]
            if len(arguments) > 2 and arguments[0] == "-c"
            else arguments[0] if arguments else "command"
        )
        raise WorkspaceError(
            f"git {operation} failed: {detail[-1] if detail else result.returncode}"
        )
    return result.stdout


@dataclass(frozen=True, slots=True)
class NetworkGitResult:
    """The successful GitHub transport used by one retryable network operation."""

    stdout: str
    used_ssh_rewrite: bool


_GITHUB_SSH_REWRITE = "url.git@github.com:.insteadOf=https://github.com/"


def git_with_ssh_fallback(
    root: Path,
    *arguments: str,
    command: Callable[..., str] | None = None,
) -> NetworkGitResult:
    """Run a network Git command, retrying GitHub HTTPS through the user's SSH setup.

    The rewrite belongs to this invocation only. In particular, a submodule command still
    records its canonical HTTPS URL in `.gitmodules`, so one author's authentication choice
    never becomes a requirement for everybody who clones their project.
    """

    run = command or git
    try:
        return NetworkGitResult(run(root, *arguments), False)
    except WorkspaceError as default_error:
        try:
            return NetworkGitResult(
                run(root, "-c", _GITHUB_SSH_REWRITE, *arguments), True
            )
        except WorkspaceError as ssh_error:
            raise WorkspaceError(
                f"{default_error}\n"
                f"GitHub SSH retry also failed: {ssh_error}"
            ) from ssh_error


def repository_address(reference: str) -> tuple[str, str]:
    """`owner/name`, however the person wrote it.

    People have the URL, not the address: it is what GitHub's own "Code" button hands them,
    in whichever scheme they use. All four forms name the same repository, so all four are
    accepted -- and every one of them resolves to the same pin, because a pin records
    `provider`, `owner`, `name` and `commit` and no URL at all. The scheme is only ever a
    property of the person fetching.

        owner/name
        https://github.com/owner/name[.git]
        ssh://git@github.com/owner/name[.git]
        git@github.com:owner/name[.git]
    """

    address = reference.strip()
    for prefix in (
        "https://github.com/",
        "http://github.com/",
        "ssh://git@github.com/",
    ):
        if address.lower().startswith(prefix):
            address = address[len(prefix) :]
            break
    else:
        # scp-style, which is not a URL and cannot be parsed as one.
        if address.lower().startswith("git@github.com:"):
            address = address[len("git@github.com:") :]
    address = address.removesuffix(".git").strip("/")
    owner, _, name = address.partition("/")
    if not owner or not name or "/" in name:
        raise WorkspaceError(
            f"give the repository as owner/name or a GitHub url, not {reference!r}"
        )
    return owner, name


def open_clone(start: Path) -> Clone:
    """The clone containing `start`, found the way git finds a repository.

    A missing `.git/verdog.json` is not fatal any more: any git repository with a
    `project.json` is a Verdog project, and it only needs the config file to know which
    service to ask. That is what makes a locally initialized repository a working start.
    """

    for candidate in (start.resolve(), *start.resolve().parents):
        if not (candidate / ".git").exists():
            continue
        if not (candidate / "project.json").is_file():
            raise WorkspaceError(
                f"{candidate} is a git repository with no project.json"
            )
        config = candidate / ".git" / CONFIG_PATH
        origin, token = DEFAULT_ORIGIN, None
        if config.is_file():
            try:
                decoded = cast(dict[str, Any], json.loads(config.read_text("utf-8")))
            except (OSError, ValueError) as error:
                raise WorkspaceError(f"{config} is unreadable") from error
            raw_origin = decoded.get("origin")
            raw_token = decoded.get("token")
            origin = raw_origin if isinstance(raw_origin, str) else DEFAULT_ORIGIN
            token = raw_token if isinstance(raw_token, str) else None
        return Clone(candidate, origin.rstrip("/"), token)
    raise WorkspaceError("not inside a git repository; `git clone` a project first")


DEFAULT_ORIGIN = "https://157.180.79.112"
"""Where to look when a clone has not been told. The development service."""


def write_config(root: Path, origin: str, token: str | None = None) -> None:
    """Record which service this clone asks, inside `.git`.

    Inside `.git` on purpose: it is not project content, it must never be committed, and it
    may hold a token -- so it has no business anywhere a commit or a publish can see. The
    *git* credential is never here: GitHub is the remote, and git's own credential helper
    holds whatever reaches it.
    """

    config = root / ".git" / CONFIG_PATH
    payload: dict[str, Any] = {"origin": origin.rstrip("/")}
    if token is not None:
        payload["token"] = token
    config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config.chmod(0o600)


__all__ = [
    "CONFIG_PATH",
    "Clone",
    "DEFAULT_ORIGIN",
    "LocalDefinition",
    "WorkspaceError",
    "contained_path",
    "dependency_clones",
    "git",
    "interpreter_in",
    "open_clone",
    "repository_address",
    "write_config",
]
