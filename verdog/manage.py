"""Everything that acts on more than the files in front of you.

Signing in, asking what you may do, offering a workflow for reuse, importing somebody
else's, and issuing a token for an agent. Ten verbs became six, and the six that went --
`new`, `fork`, `projects`, `spaces`, `share`, `members`, `invite` -- went because GitHub
already has them: a repository is created, forked, listed and shared there.

`import` is the interesting one. It adds a git submodule at a published commit with *your*
credentials, so this cannot be a way to obtain code you could not already clone. What the
service adds is the pre-flight: whether that workflow is offered for reuse at that commit,
and what else it will pull in.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .api import Service, ServiceError
from .catalogue import describe_workflow, preview_for
from .local import (
    DEFAULT_ORIGIN,
    SCHEMA_VERSION,
    Clone,
    WorkspaceError,
    contained_path,
    definition_child,
    definition_leaf,
    external_root,
    git,
    git_with_ssh_fallback,
    open_clone,
    package_problem,
    repository_address,
    write_config,
)
from .session import SessionError, account, backend_origin, credential


def _object(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _objects(value: Any) -> list[dict[str, Any]]:
    """The object entries of a JSON array, skipping anything that is not one."""

    if not isinstance(value, list):
        return []
    return [
        cast(dict[str, Any], item)
        for item in cast(list[Any], value)
        if isinstance(item, dict)
    ]


def _scoped_subroutine_definitions(
    project: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    root = _object(project.get("subroutine"))
    if not root:
        return []
    found: list[tuple[str, dict[str, Any]]] = []

    def visit(definition: dict[str, Any], owner: str | None) -> None:
        identifier = definition_child(owner, str(definition.get("id", "")))
        found.append((identifier, definition))
        for child in _objects(definition.get("subroutines")):
            visit(child, identifier)

    visit(root, None)
    return found


def _subroutine_definitions(project: dict[str, Any]) -> list[dict[str, Any]]:
    return [definition for _, definition in _scoped_subroutine_definitions(project)]


def _external_bindings(
    project: dict[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    return [
        (identifier, definition, child)
        for identifier, definition in _scoped_subroutine_definitions(project)
        for child in _objects(definition.get("workflows"))
        if "external" in child
    ]


def _rows(values: list[dict[str, Any]], columns: tuple[tuple[str, str], ...]) -> None:
    """Print a table wide enough for its contents, since a person reads this."""

    if not values:
        print("(none)")
        return
    widths = [
        max(len(title), *(len(str(row.get(key, ""))) for row in values))
        for title, key in columns
    ]
    print("  ".join(title.ljust(width) for (title, _), width in zip(columns, widths)))
    for row in values:
        print(
            "  ".join(
                str(row.get(key, "")).ljust(width)
                for (_, key), width in zip(columns, widths)
            )
        )


def _client(clone: Clone | None = None) -> Service:
    """The credential to use: the session if there is one, else the clone's own token.

    That order on purpose. An interactive person has a session and needs no token at all;
    a token exists for an agent working with nobody present, and it is the narrower of the
    two, so it must never shadow the wider one by accident.
    """

    return credential(
        None if clone is None else clone.origin,
        None if clone is None else clone.token,
    )


def _here() -> Clone:
    return open_clone(Path.cwd())


def _login(arguments: argparse.Namespace) -> int:
    """Exchange a GitHub token entered at the terminal or supplied through stdin."""

    from .session import sign_in, store

    login = sign_in(
        str(arguments.origin or backend_origin() or DEFAULT_ORIGIN),
        from_stdin=bool(arguments.github_token_stdin),
    )
    path = store(login)
    print(f"Signed in as {login.login}. Session stored in {path}.")
    return 0


def _logout(arguments: argparse.Namespace) -> int:
    del arguments
    from .session import forget

    try:
        client = account()
    except SessionError:
        print("Not signed in.")
        return 0
    try:
        client.json("POST", "/api/v1/logout")
    except ServiceError as error:
        # The local copy goes either way: a session the service already forgot is not a
        # reason to keep a credential on disk.
        print(
            f"verdog: the service could not be reached ({error}); forgetting locally."
        )
    if backend_origin() is None:
        forget()
    print("Signed out.")
    return 0


def _whoami(arguments: argparse.Namespace) -> int:
    del arguments
    client = account()
    me = client.me()
    user = _object(me.get("user"))
    print(
        f"{user.get('login') or user.get('email') or user.get('id')} at {client.origin}"
    )
    print(f"  seat: {'yes' if me.get('seat') else 'no'}")
    if me.get("system_admin"):
        print("  system administrator")
    return 0


def _clone(arguments: argparse.Namespace) -> int:
    """Clone from GitHub, initialize every pin, then point the clone at a service.

    The whole of what cloning used to do, minus everything Verdog was doing itself: there is
    no bundle to unpack, no pins to materialize one at a time, and no token to store --
    submodule urls are real GitHub urls now, so Git fetches the dependencies with the user's
    own HTTPS credentials or existing SSH setup.
    """

    owner, name = repository_address(str(arguments.repository))
    address = f"{owner}/{name}"
    destination = Path(str(arguments.destination or name)).resolve()
    if destination.exists():
        raise WorkspaceError(f"{destination} already exists")
    https_url = f"https://github.com/{address}.git"
    ssh_url = f"git@github.com:{address}.git"
    if arguments.ssh:
        url = ssh_url
    else:
        access = git_with_ssh_fallback(
            Path.cwd(), "ls-remote", "--quiet", https_url, "HEAD", command=git
        )
        url = ssh_url if access.used_ssh_rewrite else https_url
    git(Path.cwd(), "clone", "--quiet", url, str(destination))
    git_with_ssh_fallback(
        destination, "submodule", "update", "--init", "--recursive", command=git
    )
    write_config(destination, str(arguments.origin or backend_origin() or DEFAULT_ORIGIN))
    print(f"Cloned {address} into {destination}")
    print("Next: `verdog sync`, then `verdog check`.")
    return 0


ENTRY_WORKFLOW = "main"
"""The fixed root workflow/subroutine id.

The service uses the same root id. Private integration tests run this client's output
through the compiler; the client itself depends only on the public runtime.
"""


def package_from(name: str) -> str:
    """A package name derived from something a person typed.

    Lowercase snake_case, because that is what the compiler accepts and what a Python package
    has to be anyway. Not clever: it is offered back to the author, who can override it.
    """

    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    slug = re.sub(r"_{2,}", "_", slug)
    if slug and slug[0].isdigit():
        slug = f"p{slug}"
    return slug or "project"


def _remote_owner(root: Path) -> str | None:
    """The owner of this directory's GitHub remote, if it has one.

    Only a suggestion for the space, and read with `_quiet_git` because "no remote" is the
    ordinary case rather than a failure: `verdog init` works in a directory that is not a
    repository yet, and a project without a remote is a project that has not been published.
    """

    if not (root / ".git").exists():
        return None
    remote = _quiet_git_output(root, "remote", "get-url", "origin")
    if remote is None:
        return None
    try:
        owner, _ = repository_address(remote.strip())
    except WorkspaceError:
        return None
    return owner


def blank_graph(package: str, name: str) -> dict[str, Any]:
    """The smallest graph the compiler accepts: one workflow, its ports, one edge.

    Everything else -- every module, every declaration, `pyproject.toml`, the contract
    scaffolds -- is produced by the ordinary projection on the first `verdog generate`. There is
    no separate "new project" template that could drift from what the generator emits, which
    is the whole reason this is a graph and not a directory of files.

    `generated_from` is a hash no graph can have, which is how a project that has never been
    generated says so. The enter->exit edge is not optional: without it the generator refuses
    the graph, because an enter port with no outgoing edge cannot start anything.
    """

    def port(kind: str, label: str) -> dict[str, Any]:
        return {"id": kind, "name": label, "kind": kind, "operation": {}}

    return {
        "schema_version": SCHEMA_VERSION,
        "package": package,
        "generated_from": "0" * 64,
        "workflow": {
            "subroutine": ENTRY_WORKFLOW,
            "profiles": [],
            "sessions": [],
            "profile_arguments": {},
            "session_arguments": {},
        },
        "subroutine": {
            "id": ENTRY_WORKFLOW,
            "name": name,
            "ports": {"enter": "enter", "exit": "exit", "failure": "failure"},
            "features": [],
            "profiles": [],
            "profile_parameters": [],
            "sessions": [],
            "session_parameters": [],
            "nodes": [
                port("enter", "Enter"),
                port("exit", "Exit"),
                port("failure", "Failure"),
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
            "workflows": [],
            "subroutines": [],
        },
        "externals": [],
        "sources": [],
        "editor": {"layouts": {}},
        "extensions": {},
    }


GITIGNORE = """.verdog/
.venv/
__pycache__/
*.py[cod]
*.egg-info/
"""
"""What git should not carry. `verdog sync` writes `.venv/` and `.verdog/`,
`verdog run` writes into `.verdog/`, and an editable install leaves an `*.egg-info/` beside
the package -- none of it is project content, but the next thing an author does is
`git add -A`, so without this it all lands in their first commit."""


def _init(arguments: argparse.Namespace) -> int:
    """Start a project here.

    Writes the graph, then runs `generate` -- which is what turns one file into a project, so
    leaving it to the author would make `init` a verb that produces something unusable.
    """

    root = Path(str(arguments.directory or ".")).resolve()
    if (root / "project.json").exists():
        raise WorkspaceError(f"{root} already holds a project.json")
    name = str(arguments.name or root.name)
    owner = _remote_owner(root)
    if not arguments.package and owner is None:
        raise WorkspaceError(
            "init requires `--package <space>.<name>` because no GitHub remote provides a space"
        )
    package = str(
        arguments.package or f"{package_from(owner or '')}.{package_from(name)}"
    )
    complaint = package_problem(package)
    if complaint is not None:
        raise WorkspaceError(f"{package!r} cannot be a package name: {complaint}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(
        json.dumps(blank_graph(package, name), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Git, because every other verb finds the project by walking up to a `.git`. A remote is
    # not needed and not invented: the service accepts files; only publishing needs a remote.
    if not (root / ".git").exists():
        git(root, "init", "--quiet")
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(GITIGNORE, encoding="utf-8")
    origin = arguments.origin or backend_origin()
    if origin:
        write_config(root, str(origin))

    print(f"Started {package} in {root}")
    print(f"  workflow {ENTRY_WORKFLOW!r}: enter -> exit, with a failure port")

    from .main import generate_clone

    status = generate_clone(open_clone(root))
    if status == 0:
        print("Next: `verdog sync` to create the runtime environments.")
    return status


def _access(arguments: argparse.Namespace) -> int:
    """Report live catalogue permissions; local editing and compilation are independent."""

    clone = _here()
    owner, name = clone.require_repository()
    answer = _client(clone).access(owner, name)
    if getattr(arguments, "as_json", False):
        import json

        print(json.dumps(answer, indent=2))
        return 0
    if not answer.get("accessible", True):
        print(f"{answer.get('repository')}: your GitHub token cannot access this repository.")
        return 0
    print(f"{answer.get('repository')} ({answer.get('visibility')})")
    held = [
        label
        for label, key in (
            ("read", "may_read"),
            ("write", "may_write"),
            ("publish", "may_publish"),
        )
        if answer.get(key)
    ]
    print(f"  you may: {', '.join(held) or 'nothing'}")
    print(f"  seat: {'yes' if answer.get('seat') else 'no'}")
    return 0


def _catalogue(arguments: argparse.Namespace) -> int:
    entry_id = getattr(arguments, "entry", None)
    if entry_id:
        entry = account().entry(str(entry_id))
        if getattr(arguments, "as_json", False):
            print(json.dumps(entry, indent=2))
        else:
            _print_catalogue_entry(entry)
        return 0
    answer = account().catalogue(
        query=getattr(arguments, "query", None),
        visibility=getattr(arguments, "visibility", None),
        limit=getattr(arguments, "limit", None),
        cursor=getattr(arguments, "cursor", None),
    )
    if getattr(arguments, "as_json", False):
        print(json.dumps(answer, indent=2))
        return 0
    entries = _objects(answer.get("entries"))
    _rows(
        [
            {
                "title": entry.get("display_name") or entry.get("workflow_id"),
                "reference": f"{entry.get('repository')} · {entry.get('workflow_id')}",
                "commit": _release_column(entry),
                "visibility": _entry_visibility(entry),
                "published": str(entry.get("published_at") or "")[:10],
                "needs": entry.get(
                    "repository_dependency_count",
                    len(_objects(entry.get("closure"))),
                ),
                "abstract": str(entry.get("description") or ""),
            }
            for entry in entries
        ],
        (
            ("WORKFLOW", "title"),
            ("REFERENCE", "reference"),
            ("RELEASE", "commit"),
            ("VISIBILITY", "visibility"),
            ("PUBLISHED", "published"),
            ("NEEDS", "needs"),
            ("ABSTRACT", "abstract"),
        ),
    )
    cursor = answer.get("next_cursor")
    if isinstance(cursor, str) and cursor:
        print(f"\nNext cursor: {cursor}")
    return 0


def _entry_visibility(entry: dict[str, Any]) -> str:
    return str(entry.get("visibility") or entry.get("scope") or "")


def _print_catalogue_entry(entry: dict[str, Any]) -> None:
    """Present one immutable release as a compact reproducibility record."""

    preview = _object(entry.get("preview"))
    title = str(
        entry.get("display_name") or preview.get("name") or entry.get("workflow_id")
    )
    print(title)
    print("=" * max(3, len(title)))
    print(str(entry.get("description") or "No abstract supplied."))
    print("\nReproducibility record")
    for label, value in (
        ("Repository", entry.get("repository")),
        ("Workflow", entry.get("workflow_id")),
        ("Commit", entry.get("commit")),
        ("Package", entry.get("package")),
        ("Visibility", _entry_visibility(entry)),
        ("Published", entry.get("published_at")),
        ("Updated", entry.get("updated_at")),
    ):
        if value not in {None, ""}:
            print(f"  {label + ':':12} {value}")
    nodes = _objects(preview.get("nodes"))
    edges = _objects(preview.get("edges"))
    features = _objects(preview.get("features"))
    print("\nWorkflow structure")
    print(f"  {len(nodes)} node(s), {len(edges)} edge(s), {len(features)} feature(s)")
    environment = _object(entry.get("environment"))
    requirements = environment.get("requirements")
    requirement_count = (
        len(cast(list[Any], requirements)) if isinstance(requirements, list) else 0
    )
    print("\nEnvironment")
    python = environment.get("python") or "not recorded"
    print(f"  Python {python}; {requirement_count} requirement(s)")
    print("\nDependencies")
    closure = _objects(entry.get("closure"))
    _rows(
        [
            {
                "repository": item.get("repository")
                or f"{item.get('owner')}/{item.get('name')}",
                "commit": str(item.get("commit") or "")[:12],
            }
            for item in closure
        ],
        (("REPOSITORY", "repository"), ("COMMIT", "commit")),
    )


def _release_column(entry: dict[str, object]) -> str:
    """The newest release, saying so when there are older ones to pin.

    The listing shows one row per workflow rather than one per commit, so a reader who is
    told only the newest hash has no way to know the others exist. `--json` carries them.
    """

    commit = str(entry.get("commit", ""))[:12]
    count = entry.get("release_count")
    releases = (
        int(count) if isinstance(count, int) else len(_objects(entry.get("releases")))
    )
    older = releases - 1
    return f"{commit} (+{older})" if older > 0 else commit


def publish_workflow(arguments: argparse.Namespace) -> int:
    """Offer one workflow at the commit that is checked out here.

    `HEAD`, and it has to be pushed: an offer names a commit somebody else will fetch from
    GitHub, so publishing a commit that exists only on this machine would advertise
    something nobody can get.

    And it has to *work*. A published offer is a claim made to strangers; the least it can
    do is be a claim about a project that compiles.
    """

    clone = _here()
    owner, name = clone.require_repository()
    description = str(arguments.description).strip()
    if not description:
        raise WorkspaceError("--description must not be empty")
    # Deferred, the way `_init` imports it: `main` imports this module, so the cycle is real.
    from .main import check_clone_result

    status, _ = check_clone_result(clone)
    if status != 0:
        raise WorkspaceError(
            "this project does not pass `verdog check`, so there is nothing worth offering. "
            "Fix the problems above and publish the commit that fixes them."
        )
    # `check` writes the generated tree, so a clean check can leave the working copy dirty --
    # and then `HEAD` is not what was just verified. An offer names a commit, not a worktree.
    if git(clone.root, "status", "--porcelain").strip():
        raise WorkspaceError(
            "the working copy has changes that HEAD does not, so an offer at HEAD would "
            "describe code nobody can fetch. Commit and push, then publish."
        )
    repository = f"{owner}/{name}"
    commit = clone_head(clone, repository)
    root_workflow = _object(clone.project.get("workflow"))
    workflow = str(arguments.workflow or root_workflow.get("subroutine", ""))
    descriptor = describe_workflow(clone, workflow)
    if arguments.private:
        print(
            "verdog: --private is deprecated; catalogue visibility follows the GitHub "
            "repository.",
            file=sys.stderr,
        )
    entry = _client(clone).publish(
        repository,
        commit,
        descriptor.workflow_id,
        descriptor.package,
        description,
        descriptor.preview,
        private=True if arguments.private else None,
        closure=list(descriptor.closure),
        environment=descriptor.environment.as_json(),
    )
    print(
        f"Published {entry.get('repository')}@{commit[:12]} {descriptor.workflow_id} "
        f"({_entry_visibility(entry)})."
    )
    print(
        f"  others import it with: verdog import {owner}/{name}@{commit} "
        f"{descriptor.workflow_id}"
    )
    return 0


def _describe(arguments: argparse.Namespace) -> int:
    """Serialize the exact metadata publication and inspection compare."""

    description = describe_workflow(_here(), getattr(arguments, "workflow", None))
    payload = description.as_json()
    if getattr(arguments, "as_json", False):
        print(json.dumps(payload, indent=2))
        return 0
    print(description.display_name)
    print("=" * max(3, len(description.display_name)))
    print(f"Workflow       {description.workflow_id}")
    print(f"Package        {description.package}")
    print(f"Python         {description.environment.python}")
    print(f"Requirements   {len(description.environment.requirements)}")
    print(f"Dependencies   {len(description.closure)}")
    print(
        "Structure      "
        f"{len(_objects(description.preview.get('nodes')))} node(s), "
        f"{len(_objects(description.preview.get('edges')))} edge(s), "
        f"{len(_objects(description.preview.get('features')))} feature(s)"
    )
    return 0


def _retract(arguments: argparse.Namespace) -> int:
    """Withdraw an offer, by the id `catalogue` prints.

    By id and not by address, because an offer names a commit: a repository can have several
    live releases of one workflow and withdrawing "the workflow" is not a thing you can mean.
    `catalogue --json` carries the id of every release, which is where one comes from.

    What this does not do is withdraw code. A commit somebody already fetched stays fetched
    and their pin keeps working -- only the host can make a commit unreachable. Retracting
    says "I no longer offer this for reuse", and nothing stronger.
    """

    entry_id = str(arguments.entry)
    account().retract(entry_id)
    print(f"Retracted catalogue entry {entry_id}.")
    print(
        "  Anyone who already imported it keeps working: a fetched commit stays fetched."
    )
    return 0


def clone_head(clone: Clone, repository: str | None = None) -> str:
    """Require HEAD on a tracking branch of the repository being advertised."""

    commit = git(clone.root, "rev-parse", "HEAD").strip()
    advertised = repository or "/".join(clone.require_repository())
    matching = _repository_remotes(clone, advertised)
    for remote in matching:
        refs = git(
            clone.root,
            "for-each-ref",
            "--format=%(refname)",
            "--contains",
            commit,
            f"refs/remotes/{remote}/",
        ).splitlines()
        if any(ref and not ref.endswith("/HEAD") for ref in refs):
            return commit
    names = ", ".join(matching) or "no matching origin/upstream remote"
    raise WorkspaceError(
        f"{commit[:12]} is not on a remote branch of {advertised} ({names}); "
        "push it to that repository before publishing"
    )


def _repository_remotes(clone: Clone, repository: str) -> tuple[str, ...]:
    """The conventional remotes whose GitHub address is the selected repository."""

    found: list[str] = []
    for remote in ("origin", "upstream"):
        try:
            url = git(clone.root, "remote", "get-url", remote).strip()
            address = "/".join(repository_address(url))
        except WorkspaceError:
            continue
        if address.casefold() == repository.casefold():
            found.append(remote)
    return tuple(found)


def offer_for(clone: Clone, workflow_id: str) -> dict[str, Any]:
    """The graph preview published for a workflow.

    This is the publisher's claim, and nothing the service went and looked for. It carries
    enough of the graph to browse before importing instead of after.

    `editor` is deliberately absent: a preview lays itself out, and a layout is not part of
    what a workflow *is*.
    """

    try:
        return {"preview": preview_for(clone, workflow_id)}
    except WorkspaceError as error:
        if "unknown workflow" not in str(error):
            raise
        raise WorkspaceError(f"this project has no workflow {workflow_id}") from error


def _init_submodules(checkout: Path) -> None:
    """Initialize every dependency owned by one freshly selected checkout."""

    git(checkout, "submodule", "sync", "--recursive")
    git_with_ssh_fallback(
        checkout, "submodule", "update", "--init", "--recursive", command=git
    )


def _external_checkout(clone: Clone, alias: str) -> Path:
    """An alias path proven to stay inside this clone's `external/` tree."""

    relative = Path(external_root(alias))
    root = clone.root.resolve()
    external = (clone.root / "external").resolve()
    checkout = contained_path(clone.root, relative)
    if not external.is_relative_to(root) or external not in checkout.parents:
        raise WorkspaceError(
            f"external alias {alias!r} escapes {clone.root / 'external'}"
        )
    for ancestor in checkout.parents:
        if ancestor == external:
            break
        metadata = ancestor / ".git"
        if metadata.exists() or metadata.is_symlink():
            raise WorkspaceError(
                f"external alias {alias!r} is nested inside another Git worktree"
            )
    return clone.root / relative


def _node_directory(clone: Clone, source: Path, node_id: str) -> Path:
    """A node path proven safe before recursive removal."""

    if source.is_absolute() or not source.parts or ".." in source.parts:
        raise WorkspaceError(f"definition path {source} escapes this project")
    if not node_id or node_id in {".", ".."} or Path(node_id).parts != (node_id,):
        raise WorkspaceError(f"node id {node_id!r} is not a safe path component")
    relative = source / "nodes" / node_id
    return contained_path(clone.root, relative)


def _visit_implementation(
    clone: Clone,
    definition: Path,
    target_id: str,
    edge_id: str,
) -> Path:
    """A target-owned edge visit proven safe before removal."""

    for label, value in (("target node id", target_id), ("edge id", edge_id)):
        if not value or value in {".", ".."} or Path(value).parts != (value,):
            raise WorkspaceError(f"{label} {value!r} is not a safe path component")
    node = _node_directory(clone, definition, target_id)
    relative = node.relative_to(clone.root.resolve()) / "visit" / edge_id / "impl.py"
    return contained_path(clone.root, relative)


def _module_git_dir(clone: Clone, target: str) -> Path:
    raw = git(clone.root, "rev-parse", "--git-path", f"modules/{target}").strip()
    path = Path(raw)
    return path if path.is_absolute() else clone.root / path


def _reuse_recovery(clone: Clone, target: str, repository: str) -> bool:
    """Reuse a dropped submodule's object store only when it is the same repository."""

    module = _module_git_dir(clone, target)
    if not module.exists():
        return False
    if module.is_symlink() or not module.is_dir():
        raise WorkspaceError(f"Git recovery data for {target} is not a directory")
    remote = _quiet_git_output(
        clone.root, f"--git-dir={module}", "remote", "get-url", "origin"
    )
    if remote is None:
        raise WorkspaceError(f"Git recovery data for {target} has no readable origin")
    try:
        recovered = repository_address(remote.strip())
    except WorkspaceError as error:
        raise WorkspaceError(
            f"Git recovery data for {target} has an unrecognised origin; use another alias"
        ) from error
    if recovered != repository_address(repository):
        raise WorkspaceError(
            f"Git recovery data for {target} belongs to {recovered[0]}/{recovered[1]}, "
            f"not {repository}; use another alias"
        )
    return True


def _rollback_import(
    clone: Clone,
    target: str,
    modules_before: bytes | None,
    modules_index_before: tuple[str, str] | None,
    project_before: bytes,
) -> list[str]:
    """Remove only state created by one failed import; keep the recovery object store."""

    errors: list[str] = []
    _quiet_git(clone.root, "submodule", "deinit", "--force", "--", target)
    _quiet_git(clone.root, "update-index", "--force-remove", "--", target)
    _quiet_git(clone.root, "config", "--remove-section", f"submodule.{target}")
    checkout = clone.root / target
    if checkout.is_symlink():
        errors.append(f"refused to remove symbolic link {target}")
    elif checkout.exists():
        try:
            shutil.rmtree(checkout)
        except OSError as error:
            errors.append(f"could not remove {target}: {error}")

    modules = clone.root / ".gitmodules"
    try:
        if modules_index_before is None:
            _quiet_git(
                clone.root, "update-index", "--force-remove", "--", ".gitmodules"
            )
        else:
            mode, object_id = modules_index_before
            if not _quiet_git(
                clone.root,
                "update-index",
                "--add",
                "--cacheinfo",
                f"{mode},{object_id},.gitmodules",
            ):
                errors.append("could not restore .gitmodules in the index")
        if modules.is_symlink() or (modules.exists() and not modules.is_file()):
            raise OSError(".gitmodules is not a regular file")
        if modules_before is None:
            if modules.exists():
                modules.unlink()
        else:
            modules.write_bytes(modules_before)
    except OSError as error:
        errors.append(f"could not restore .gitmodules: {error}")
    try:
        (clone.root / "project.json").write_bytes(project_before)
    except OSError as error:
        errors.append(f"could not restore project.json: {error}")
    if _quiet_git_output(clone.root, "ls-files", "--stage", "--", target):
        errors.append(f"could not remove {target} from the index")
    return errors


def _rollback_bump(
    clone: Clone,
    target: Path,
    target_name: str,
    head: str,
    head_reference: str | None,
    index_entry: tuple[str, str],
    project_before: bytes,
) -> list[str]:
    """Best-effort restoration after a selected checkout has moved."""

    errors: list[str] = []
    if not _quiet_git(target, "submodule", "deinit", "--force", "--all"):
        errors.append(f"could not deinitialize {target_name}'s nested dependencies")
    checkout_restored = False
    try:
        git(target, "checkout", "--force", "--quiet", "--detach", head)
        checkout_restored = True
    except WorkspaceError as error:
        errors.append(str(error))
    if (
        checkout_restored
        and head_reference is not None
        and not _quiet_git(target, "symbolic-ref", "HEAD", head_reference)
    ):
        errors.append(f"could not restore {target_name}'s branch")
    if checkout_restored:
        try:
            _init_submodules(target)
        except WorkspaceError as error:
            errors.append(str(error))
    try:
        (clone.root / "project.json").write_bytes(project_before)
    except OSError as error:
        errors.append(f"could not restore project.json: {error}")
    mode, object_id = index_entry
    if not _quiet_git(
        clone.root,
        "update-index",
        "--add",
        "--cacheinfo",
        f"{mode},{object_id},{target_name}",
    ):
        errors.append(f"could not restore the {target_name} gitlink")
    return errors


def _report_rollback(errors: list[str]) -> None:
    if errors:
        print("verdog: rollback incomplete: " + "; ".join(errors), file=sys.stderr)


def _import(arguments: argparse.Namespace) -> int:
    if not getattr(arguments, "as_json", False):
        status, _ = _import_result(arguments)
        return status
    # Git and generation retain useful progress on stderr; stdout is one protocol object.
    with redirect_stdout(sys.stderr):
        status, result = _import_result(arguments)
    print(json.dumps(result, indent=2))
    return status


def _import_result(arguments: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Add somebody else's published workflow as a dependency of this project.

    Four steps, and only the first is ours: pre-flight the offer, `git submodule add` at the
    commit, write the `externals` entry and a workflow binding, then `check`. The fetch uses
    your credentials, so a repository you cannot clone is one you cannot import -- there is
    no path through this service for the bytes.
    """

    plan = _prepare_import(arguments)
    snapshot = _snapshot_import(plan)
    binding = _materialize_import_with_rollback(plan, snapshot)
    from .main import generate_clone

    status = generate_clone(plan.clone)
    print(f"Added {plan.target} and workflow binding {binding}.")
    print(f"  Add a workflow_call targeting {binding}, then run `verdog sync`.")
    return status, {
        "status": "imported",
        "repository": plan.repository,
        "commit": plan.commit,
        "workflow_id": plan.workflow,
        "package": plan.package,
        "alias": plan.alias,
        "target": plan.target,
        "binding": binding,
        "generated_status": status,
    }


@dataclass(frozen=True, slots=True)
class _ImportPlan:
    clone: Clone
    project: dict[str, Any]
    parent: str
    owner: str
    name: str
    repository: str
    commit: str
    workflow: str
    entry: dict[str, Any]
    closure: list[dict[str, Any]]
    package: str
    alias: str
    target: str
    checkout: Path


@dataclass(frozen=True, slots=True)
class _ImportSnapshot:
    modules: Path
    modules_before: bytes | None
    modules_index_before: tuple[str, str] | None
    project_before: bytes
    reuse: bool


def _prepare_import(arguments: argparse.Namespace) -> _ImportPlan:
    clone = _here()
    project = clone.project
    parent = _import_parent(project, getattr(arguments, "into", None))
    owner, name, repository, commit = _import_reference(str(arguments.reference))
    workflow = str(arguments.workflow)
    entry = _client(clone).preflight(repository, commit, workflow)
    package = _import_package(entry, repository, commit)
    alias, target = _import_alias(project, getattr(arguments, "alias", None), package)
    closure = _objects(entry.get("closure"))
    _print_import_offer(repository, commit, workflow, package, closure)
    checkout = _external_checkout(clone, alias)
    if checkout.exists():
        raise WorkspaceError(f"{target} already exists")
    return _ImportPlan(
        clone,
        project,
        parent,
        owner,
        name,
        repository,
        commit,
        workflow,
        entry,
        closure,
        package,
        alias,
        target,
        checkout,
    )


def _import_parent(project: dict[str, Any], requested: object) -> str:
    root = _object(project.get("subroutine"))
    parent = str(requested or root.get("id", ""))
    known = {identifier for identifier, _ in _scoped_subroutine_definitions(project)}
    if parent not in known:
        raise WorkspaceError(f"this project has no subroutine {parent}")
    return parent


def _import_reference(address: str) -> tuple[str, str, str, str]:
    reference, separator, commit = address.rpartition("@")
    if not separator or re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise WorkspaceError(
            "give the release as owner/name@<40-character commit>, or a GitHub url with "
            "the commit appended the same way"
        )
    owner, name = repository_address(reference)
    return owner, name, f"{owner}/{name}", commit.lower()


def _import_package(entry: dict[str, Any], repository: str, commit: str) -> str:
    raw = entry.get("package")
    package = raw if isinstance(raw, str) else ""
    if not package and entry.get("unpublished") is True:
        package = _remote_package(repository, commit)
    complaint = package_problem(package)
    if complaint:
        raise WorkspaceError(
            f"{repository}@{commit[:12]} publishes an invalid package {package!r}: "
            f"{complaint}"
        )
    return package


def _import_alias(
    project: dict[str, Any], requested: object, package: str
) -> tuple[str, str]:
    alias = str(requested or package)
    complaint = package_problem(alias)
    if complaint:
        raise WorkspaceError(f"{alias!r} cannot be an external alias: {complaint}")
    pins = _objects(project.get("externals"))
    if any(pin.get("alias") == alias for pin in pins):
        raise WorkspaceError(f"this project already pins the alias {alias!r}")
    target = external_root(alias)
    for pin in pins:
        other = external_root(str(pin.get("alias", "")))
        if target.startswith(other + "/") or other.startswith(target + "/"):
            raise WorkspaceError(
                f"external alias {alias!r} overlaps {pin.get('alias')!r} on disk"
            )
    return alias, target


def _print_import_offer(
    repository: str,
    commit: str,
    workflow: str,
    package: str,
    closure: list[dict[str, Any]],
) -> None:
    print(f"{repository}@{commit[:12]} offers {workflow} as {package}")
    if closure:
        needs = ", ".join(
            f"{item.get('repository')}@{str(item.get('commit', ''))[:12]}"
            for item in closure
        )
        print(f"  and needs {len(closure)}: {needs}")


def _snapshot_import(plan: _ImportPlan) -> _ImportSnapshot:
    modules, before, indexed = _gitmodules_snapshot(plan)
    _require_unused_import_target(plan)
    return _ImportSnapshot(
        modules,
        before,
        indexed,
        (plan.clone.root / "project.json").read_bytes(),
        _reuse_recovery(plan.clone, plan.target, plan.repository),
    )


def _gitmodules_snapshot(
    plan: _ImportPlan,
) -> tuple[Path, bytes | None, tuple[str, str] | None]:
    modules = plan.clone.root / ".gitmodules"
    if modules.is_symlink() or (modules.exists() and not modules.is_file()):
        raise WorkspaceError(".gitmodules is not a regular file")
    listed = _quiet_git_output(
        plan.clone.root, "ls-files", "--stage", "--", ".gitmodules"
    )
    if listed is None:
        raise WorkspaceError("Git could not inspect .gitmodules")
    entries = [line.partition("\t")[0].split() for line in listed.splitlines()]
    if any(len(fields) != 3 or fields[2] != "0" for fields in entries):
        raise WorkspaceError(".gitmodules has unresolved index entries")
    if len(entries) > 1:
        raise WorkspaceError(".gitmodules has more than one index entry")
    indexed = (entries[0][0], entries[0][1]) if entries else None
    if not modules.exists() and (
        indexed is not None or _objects(plan.project.get("externals"))
    ):
        raise WorkspaceError(
            ".gitmodules is missing from the working tree; restore it deliberately before "
            "importing"
        )
    before = modules.read_bytes() if modules.exists() else None
    return modules, before, indexed


def _require_unused_import_target(plan: _ImportPlan) -> None:
    indexed = _quiet_git_output(
        plan.clone.root, "ls-files", "--stage", "--", plan.target
    )
    if indexed is None:
        raise WorkspaceError(f"Git could not inspect {plan.target} in the index")
    if indexed.strip():
        raise WorkspaceError(f"{plan.target} is already present in the Git index")
    configured = _quiet_git_output(
        plan.clone.root,
        "config",
        "--local",
        "--name-only",
        "--get-regexp",
        rf"^submodule\.{re.escape(plan.target)}\.",
    )
    if configured:
        raise WorkspaceError(
            f"Git still configures {plan.target}; repair or drop it first"
        )


def _materialize_import_with_rollback(
    plan: _ImportPlan, snapshot: _ImportSnapshot
) -> str:
    try:
        return _materialize_import(plan, snapshot.reuse)
    except BaseException:
        _report_rollback(
            _rollback_import(
                plan.clone,
                plan.target,
                snapshot.modules_before,
                snapshot.modules_index_before,
                snapshot.project_before,
            )
        )
        raise


def _materialize_import(plan: _ImportPlan, reuse: bool) -> str:
    _add_import_submodule(plan, reuse)
    git_with_ssh_fallback(
        plan.checkout,
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "origin",
        plan.commit,
        command=git,
    )
    git(plan.checkout, "checkout", "--quiet", "--detach", plan.commit)
    _verify_import_checkout(plan)
    _init_submodules(plan.checkout)
    git(plan.clone.root, "add", "--", plan.target)
    return _record_external(
        plan.clone,
        {
            "alias": plan.alias,
            "package": plan.package,
            "provider": "github",
            "repository_id": plan.entry.get("repository_id"),
            "owner": plan.owner,
            "name": plan.name,
            "commit": plan.commit,
        },
        plan.workflow,
        plan.parent,
    )


def _add_import_submodule(plan: _ImportPlan, reuse: bool) -> None:
    arguments = ["submodule", "add", "--quiet"]
    if reuse:
        arguments.append("--force")
    git_with_ssh_fallback(
        plan.clone.root,
        *arguments,
        f"https://github.com/{plan.repository}.git",
        plan.target,
        command=git,
    )


def _verify_import_checkout(plan: _ImportPlan) -> None:
    imported = Clone(plan.checkout, plan.clone.origin, None)
    package = imported.project.get("package")
    if package != plan.package:
        raise WorkspaceError(
            f"{plan.repository}@{plan.commit[:12]} contains package {package!r}, "
            f"not the expected {plan.package!r}"
        )
    try:
        imported.workflow_definition(plan.workflow)
    except WorkspaceError as error:
        raise WorkspaceError(
            f"{plan.repository}@{plan.commit[:12]} has no workflow {plan.workflow!r}"
        ) from error


def _remote_package(repository: str, commit: str) -> str:
    """Read an unpublished owned release's package without changing the consumer clone.

    The service deliberately cannot read repository content, so its authorized
    unpublished preflight has no package. A short-lived bare repository obtains exactly
    `project.json`; the final submodule fetch still performs the ordinary import with the
    user's Git setup.
    """

    with tempfile.TemporaryDirectory(prefix="verdog-import-metadata-") as temporary:
        metadata_root = Path(temporary)
        git(metadata_root, "init", "--quiet", "--bare")
        git_with_ssh_fallback(
            metadata_root,
            "fetch",
            "--quiet",
            "--depth",
            "1",
            f"https://github.com/{repository}.git",
            commit,
            command=git,
        )
        try:
            decoded = json.loads(git(metadata_root, "show", f"{commit}:project.json"))
        except ValueError as error:
            raise WorkspaceError(
                f"{repository}@{commit[:12]} has no readable project.json"
            ) from error
    if not isinstance(decoded, dict):
        raise WorkspaceError(
            f"{repository}@{commit[:12]} project.json is not an object"
        )
    project = cast(dict[str, Any], decoded)
    package = project.get("package")
    return package if isinstance(package, str) else ""


def bump_dependency(arguments: argparse.Namespace) -> int:
    """Move a pin to a newer commit, leaving the graph alone.

    The step the loop was missing. Editing a dependency in its submodule, committing and pushing
    it are all ordinary git; what had no command was pointing this project at the result. The
    only way was `drop` then `import`, and `drop` removes "the pin, its calling nodes, and the
    checkout" -- so advancing one commit meant deleting the node that calls the dependency and
    wiring it back. Nothing about a new commit implies a different shape for the graph, so
    nothing here touches nodes, edges or layouts.

    Reads the new commit's manifest with `git show` *before* moving the checkout, so a missing or
    malformed project is refused without changing the selected checkout.
    """

    clone = _here()
    project = clone.project
    alias = str(arguments.alias)
    complaint = package_problem(alias)
    if complaint:
        raise WorkspaceError(f"{alias!r} cannot be an external alias: {complaint}")
    pins = {str(pin.get("alias")): pin for pin in _objects(project.get("externals"))}
    pin = pins.get(alias)
    if pin is None:
        known = ", ".join(sorted(pins)) or "none"
        raise WorkspaceError(
            f"this project does not pin {alias!r}. Pinned aliases: {known}"
        )
    package = str(pin.get("package", ""))

    target_name = external_root(alias)
    target = _external_checkout(clone, alias)
    git_metadata = target / ".git"
    if git_metadata.is_symlink() or not git_metadata.exists():
        raise WorkspaceError(
            f"{target_name} is not an exact initialized Git worktree; "
            "run `git submodule update --init --recursive`"
        )
    if not (target / "project.json").is_file():
        raise WorkspaceError(f"{target_name} has no readable project.json")
    top_level = Path(git(target, "rev-parse", "--show-toplevel").strip()).resolve()
    if top_level != target.resolve():
        raise WorkspaceError(f"{target_name} is not its own Git worktree")

    listed = git(clone.root, "ls-files", "--stage", "--", target_name)
    entries: list[tuple[list[str], str]] = []
    for line in listed.splitlines():
        metadata, separator, path = line.partition("\t")
        if separator:
            entries.append((metadata.split(), path))
    if (
        len(entries) != 1
        or entries[0][1] != target_name
        or len(entries[0][0]) != 3
        or entries[0][0][0] != "160000"
        or entries[0][0][2] != "0"
    ):
        raise WorkspaceError(
            f"the owner does not record {target_name} as one exact, unconflicted gitlink"
        )
    mode, indexed_commit, _ = entries[0][0]
    head = git(target, "rev-parse", "HEAD").strip()
    recorded_commit = str(pin.get("commit", ""))
    if recorded_commit != indexed_commit:
        raise WorkspaceError(
            f"project.json records {recorded_commit[:12] or 'no commit'} for {alias}, but "
            f"its owner records {indexed_commit[:12]}"
        )
    # Refuse before fetching, not after: bumping over uncommitted work in a dependency would
    # discard the one thing here that exists nowhere else.
    if git(
        target,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ).strip():
        raise WorkspaceError(
            f"{target_name} has uncommitted changes. Commit and push them, record the commit, "
            "then restore the owner-recorded checkout before bumping with `--commit`."
        )

    owner, name = str(pin.get("owner")), str(pin.get("name"))
    repository = f"{owner}/{name}"
    workflow = _pinned_workflow(clone, alias)
    commit = str(arguments.commit or "")
    if not commit:
        commit = _newest_release(clone, repository, workflow)
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise WorkspaceError("give the commit as a full 40-character id")
    commit = commit.lower()
    if head not in {indexed_commit, commit}:
        raise WorkspaceError(
            f"{target_name} is at {head[:12]}, but its owner records "
            f"{indexed_commit[:12]} and this bump requests {commit[:12]}. Restore one of those "
            "commits before bumping."
        )
    if commit == recorded_commit:
        print(f"{alias} is already at {commit[:12]}.")
        return 0

    # The same pre-flight an import does, and for the same reason: a commit nobody published is
    # one nobody else can fetch, so a pin naming it is a project only this machine can build.
    entry = _client(clone).preflight(repository, commit, workflow)
    if str(entry.get("package", package)) != package:
        raise WorkspaceError(
            f"{repository}@{commit[:12]} publishes {entry.get('package')!r}, not {package!r}. A "
            "package rename is not a bump: the name is in every generated path."
        )

    git_with_ssh_fallback(
        target,
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "origin",
        commit,
        command=git,
    )
    try:
        decoded = json.loads(git(target, "show", f"{commit}:project.json"))
        if not isinstance(decoded, dict):
            raise ValueError("project.json is not an object")
    except ValueError as error:
        raise WorkspaceError(
            f"{repository}@{commit[:12]} has no readable project.json"
        ) from error

    project_before = (clone.root / "project.json").read_bytes()
    head_reference = _quiet_git_output(target, "symbolic-ref", "-q", "HEAD")
    head_reference = head_reference.strip() if head_reference is not None else None
    try:
        git(target, "checkout", "--quiet", "--detach", commit)
        _init_submodules(target)
        git(clone.root, "add", "--", target_name)
        updated = {
            **pin,
            "commit": commit,
            "repository_id": entry.get("repository_id", pin.get("repository_id")),
        }
        project["externals"] = [
            updated if current.get("alias") == alias else current
            for current in _objects(project.get("externals"))
        ]
        (clone.root / "project.json").write_text(
            json.dumps(project, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except BaseException:
        _report_rollback(
            _rollback_bump(
                clone,
                target,
                target_name,
                head,
                head_reference,
                (mode, indexed_commit),
                project_before,
            )
        )
        raise
    print(f"{alias}: {str(pin.get('commit'))[:12]} -> {commit[:12]}")
    # A dependency may change qualified calls in its parent, so regenerate it through the service.
    from .main import generate_clone

    return generate_clone(clone)


def _pinned_workflow(clone: Clone, alias: str) -> str:
    """Which dependency workflow is bound to an alias."""

    for _, _, binding in _external_bindings(clone.project):
        external = _object(binding.get("external"))
        if str(external.get("alias")) == alias:
            return str(external.get("workflow"))
    raise WorkspaceError(
        f"nothing in this graph binds {alias}, so there is no release to look up. "
        "Import its workflow or pass --commit."
    )


def _newest_release(clone: Clone, repository: str, workflow: str) -> str:
    """The newest commit the catalogue offers for this workflow.

    Search narrows the catalogue, but exact identity still decides: descriptions can contain
    repository names, and the requested workflow may sit beyond the first result page.
    """

    client = _client(clone)
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        listing = client.catalogue(query=repository, limit=100, cursor=cursor)
        for entry in _objects(listing.get("entries")):
            address = entry.get("repository")
            if not isinstance(address, str):
                address = f"{entry.get('owner')}/{entry.get('name')}"
            if (
                address.casefold() == repository.casefold()
                and str(entry.get("workflow_id")) == workflow
            ):
                return str(entry.get("commit"))
        next_cursor = listing.get("next_cursor")
        if next_cursor is None:
            break
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen:
            raise WorkspaceError("the catalogue returned an invalid pagination cursor")
        seen.add(next_cursor)
        cursor = next_cursor
    raise WorkspaceError(
        f"the catalogue offers no release of {workflow} from {repository}. Publish one, or pass "
        "--commit to pin a specific one."
    )


def _quiet_git_output(root: Path, *arguments: str) -> str | None:
    """What git printed, or `None` if it declined. For questions, not orders."""

    import subprocess

    finished = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
    return finished.stdout if finished.returncode == 0 else None


def _quiet_git(root: Path, *arguments: str) -> bool:
    """Run git where failure is an expected answer, not an error.

    `config --remove-section` exits non-zero when the section is not there, which is the
    ordinary case when a previous drop already removed it. `git()` raises, and rightly -- so
    the few places that are asking a question rather than giving an order use this instead.
    """

    import subprocess

    return (
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
        ).returncode
        == 0
    )


def drop_dependency(arguments: argparse.Namespace) -> int:
    """Stop depending on a pinned workflow: the pin, its calling nodes, and the checkout.

    Public because it is tested, the way `blank_graph` and `offer_for` are: a private helper
    cannot be exercised without `pyright` objecting, and the behaviour worth asserting here is
    exactly what a half-undone project looks like afterwards.

    The mirror of `import`, and it has to undo all three or it undoes nothing useful. Deleting
    just the calling node in the editor leaves a pin nothing calls, which the compiler refuses
    (`pinned dependencies are unused`) -- so a project can be edited into a state no gesture
    gets it out of. This is that gesture.

    Tolerant of every partial state, because that is exactly when it is reached: a pin with no
    node, a node with no pin, a `.gitmodules` entry whose directory was deleted by hand. Each
    piece is removed if present and skipped if not.
    """

    clone = _here()
    alias = str(arguments.alias)
    complaint = package_problem(alias)
    if complaint:
        raise WorkspaceError(f"{alias!r} cannot be an external alias: {complaint}")
    project = clone.project
    target = external_root(alias)
    checkout = _external_checkout(clone, alias)
    modules = clone.root / ".gitmodules"
    if modules.is_symlink() or (modules.exists() and not modules.is_file()):
        raise WorkspaceError(".gitmodules is not a regular file")

    pins = _objects(project.get("externals"))
    declared = any(pin.get("alias") == alias for pin in pins)
    matching_bindings = [
        (owner, binding)
        for owner, _, binding in _external_bindings(project)
        if _object(binding.get("external")).get("alias") == alias
    ]
    binding_ids = {
        definition_child(owner, str(binding.get("id", "")))
        for owner, binding in matching_bindings
    }

    def calls_alias(node: dict[str, Any]) -> bool:
        target = _object(node.get("operation")).get("target")
        return (node.get("kind") == "workflow_call" and target in binding_ids) or (
            node.get("kind") == "subroutine_call"
            and isinstance(target, str)
            and target.partition("/")[0] == alias
            and "/" in target
        )

    called = any(
        calls_alias(node)
        for definition in _subroutine_definitions(project)
        for node in _objects(definition.get("nodes"))
    )

    indexed = _quiet_git_output(clone.root, "ls-files", "--stage", "--", target)
    indexed_commit = ""
    nested_gitlinks: list[str] = []
    for line in indexed.splitlines() if indexed else []:
        metadata, separator, path = line.partition("\t")
        fields = metadata.split()
        if not separator or len(fields) < 2 or fields[0] != "160000":
            continue
        if path == target:
            indexed_commit = fields[1]
        elif path.startswith(target + "/"):
            nested_gitlinks.append(path)

    module_sections: list[str] = []
    configured = _quiet_git_output(
        clone.root,
        "config",
        "--file",
        ".gitmodules",
        "--get-regexp",
        r"^submodule\..*\.path$",
    )
    for line in configured.splitlines() if configured else []:
        key, separator, path = line.partition(" ")
        if separator and path == target and key.endswith(".path"):
            module_sections.append(key.removesuffix(".path"))

    nested_aliases = [
        str(pin.get("alias"))
        for pin in pins
        if pin.get("alias") != alias
        and external_root(str(pin.get("alias"))).startswith(target + "/")
    ]
    if nested_gitlinks or nested_aliases:
        nested = sorted(
            {*nested_gitlinks, *(external_root(name) for name in nested_aliases)}
        )
        raise WorkspaceError(
            f"{target} contains other dependencies ({', '.join(nested)}); "
            "drop their exact aliases instead"
        )
    # A child commit not yet recorded by its owner's gitlink, or work not committed anywhere,
    # is recovery data. Refuse before changing the manifest or deleting a scaffold.
    initialized = (checkout / ".git").exists()
    owned = (
        declared
        or bool(matching_bindings)
        or called
        or bool(indexed_commit)
        or bool(module_sections)
    )
    if not owned:
        print(f"Nothing to drop: {alias} is not a dependency of this project.")
        return 0
    if (
        checkout.exists()
        and not initialized
        and (not checkout.is_dir() or any(checkout.iterdir()))
    ):
        raise WorkspaceError(
            f"{target} is non-empty but is not an initialized Git worktree; "
            "move its contents somewhere safe before dropping it"
        )
    dirty = (
        _quiet_git_output(
            checkout,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        if initialized
        else None
    )
    if initialized and dirty is None:
        raise WorkspaceError(f"{target} is initialized but Git could not inspect it")
    if dirty is not None and dirty.strip():
        raise WorkspaceError(
            f"{target} has uncommitted changes. Commit or stash them before dropping it."
        )
    head = _quiet_git_output(checkout, "rev-parse", "HEAD") if initialized else None
    if initialized and head is None:
        raise WorkspaceError(f"{target} is initialized but Git could not read its HEAD")
    if head is not None and head.strip() != indexed_commit:
        recorded = indexed_commit[:12] if indexed_commit else "no gitlink"
        raise WorkspaceError(
            f"{target} is at {head.strip()[:12]}, but its owner records "
            f"{recorded}. Commit the updated gitlink before dropping it."
        )

    kept = [pin for pin in pins if pin.get("alias") != alias]
    dropped = len(pins) - len(kept)
    project["externals"] = kept

    bindings = 0
    callers = 0
    removed_directories: set[Path] = set()
    removed_prefixes: set[str] = set()
    removed_visit_files: set[Path] = set()
    removed_visit_paths: set[str] = set()
    source_by_id = {
        definition.local_id: definition.source
        for definition in clone.local_definitions()
        if definition.kind == "subroutine"
    }
    for definition_id, definition in _scoped_subroutine_definitions(project):
        children = _objects(definition.get("workflows"))
        kept_children = [
            child
            for child in children
            if _object(child.get("external")).get("alias") != alias
        ]
        bindings += len(children) - len(kept_children)
        definition["workflows"] = kept_children

        nodes = _objects(definition.get("nodes"))
        removed = [node for node in nodes if calls_alias(node)]
        remaining = [node for node in nodes if not calls_alias(node)]
        callers += len(removed)
        if len(remaining) != len(nodes):
            source = source_by_id.get(definition_id)
            removed_ids = {str(node.get("id", "")) for node in removed}
            if source is not None:
                for node_id in removed_ids:
                    removed_prefixes.add((source / "nodes" / node_id).as_posix() + "/")
                    removed_directories.add(_node_directory(clone, source, node_id))
                by_id = {str(node.get("id", "")): node for node in nodes}
                for edge in _objects(definition.get("edges")):
                    edge_source = str(edge.get("source", ""))
                    target_id = str(edge.get("target", ""))
                    if edge_source not in removed_ids or target_id in removed_ids:
                        continue
                    target_node = by_id.get(target_id)
                    if target_node is None or target_node.get("kind") in {
                        "enter",
                        "exit",
                        "failure",
                    }:
                        continue
                    edge_id = str(edge.get("id", ""))
                    visit = _visit_implementation(
                        clone,
                        source,
                        target_id,
                        edge_id,
                    )
                    removed_visit_files.add(visit)
                    removed_visit_paths.add(visit.relative_to(clone.root).as_posix())
            definition["nodes"] = remaining
            # An edge to a node that is gone is not a graph, so the edges go with it.
            names = {str(_object(node).get("id")) for node in remaining}
            definition["edges"] = [
                edge
                for edge in _objects(definition.get("edges"))
                if str(edge.get("source")) in names and str(edge.get("target")) in names
            ]

    tracked_paths = {
        str(source.get("path", "")) for source in _objects(project.get("sources"))
    }
    affected = sorted(
        {
            *(
                directory.relative_to(clone.root).as_posix()
                for directory in removed_directories
                if directory.exists()
            ),
            *(
                path
                for path in removed_prefixes
                if any(tracked.startswith(path) for tracked in tracked_paths)
            ),
            *(
                file.relative_to(clone.root).as_posix()
                for file in removed_visit_files
                if file.exists()
            ),
            *(path for path in removed_visit_paths if path in tracked_paths),
        }
    )
    if affected and not bool(getattr(arguments, "discard_node_code", False)):
        listed = "\n".join(f"  {path}" for path in affected)
        raise WorkspaceError(
            "dropping this dependency removes authored node code and visits; rerun with "
            f"--discard-node-code to confirm:\n{listed}"
        )

    for directory in removed_directories:
        if directory.exists():
            shutil.rmtree(directory)
    for file in removed_visit_files:
        file.unlink(missing_ok=True)
    if removed_prefixes or removed_visit_paths:
        # The manifest names every file this project has, and `check` reads them all before it
        # regenerates anything -- so a path left in `sources` after its file is gone fails the
        # read rather than being noticed and corrected. It is the same document, so it is the
        # same edit.
        project["sources"] = [
            source
            for source in _objects(project.get("sources"))
            if not any(
                str(source.get("path", "")).startswith(prefix)
                for prefix in removed_prefixes
            )
            and str(source.get("path", "")) not in removed_visit_paths
        ]

    if dropped or bindings or callers or removed_prefixes or removed_visit_paths:
        (clone.root / "project.json").write_text(
            json.dumps(project, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # `deinit` then `rm` releases the worktree and removes the gitlink. Git deliberately keeps
    # `.git/modules/<path>`; that object store is the recovery copy for commits that may have no
    # other ref yet, so `drop` must not erase it.
    removed_files = checkout.exists()
    _quiet_git(clone.root, "submodule", "deinit", "--force", target)
    _quiet_git(clone.root, "rm", "--force", "--quiet", target)
    # Whatever git declined to take, take anyway.
    shutil.rmtree(checkout, ignore_errors=True)

    # The `.gitmodules` stanza, by hand. `git rm` claims to remove it, and does -- except under
    # conditions that are easy to walk into: dropping two dependencies in a row leaves the
    # first drop's edit staged, and the second `git rm` then leaves its own stanza behind. A
    # `.gitmodules` naming a submodule that is not there breaks `git clone --recursive` for
    # everyone downstream, so this is finished rather than trusted.
    if modules.is_file():
        for section in module_sections:
            _quiet_git(
                clone.root,
                "config",
                "--file",
                ".gitmodules",
                "--remove-section",
                section,
            )
        # An emptied `.gitmodules` is left in place rather than removed, and that is not
        # tidiness lost -- it is the difference between working and not. Removing it stages a
        # deletion, and `git submodule add` then refuses outright: "please make sure that the
        # .gitmodules file is in the working tree". So dropping a dependency and importing
        # another in the same working tree failed on the second step, for a file nobody asked
        # about. An empty one is inert; git reads it and finds no submodules.

    print(
        f"Dropped {alias}: "
        f"{'pin, ' if dropped else ''}"
        f"{f'{bindings} binding(s), ' if bindings else ''}"
        f"{f'{callers} calling node(s), ' if callers else ''}"
        f"{'checkout' if removed_files else 'no checkout'}."
    )
    from .main import generate_clone

    return generate_clone(clone)


def _record_external(
    clone: Clone,
    pin: dict[str, Any],
    workflow: str,
    parent: str,
) -> str:
    """Write a pin and an inert external WorkflowDefinition binding."""

    import json

    project = clone.project
    project["externals"] = [*_objects(project.get("externals")), pin]
    for identifier, candidate in _scoped_subroutine_definitions(project):
        if identifier != parent:
            continue
        workflows = cast(list[Any], candidate.setdefault("workflows", []))
        taken = {
            str(child.get("id", definition_leaf(str(child.get("subroutine", "")))))
            for child in _objects(workflows)
        }
        binding_id = _free_id(definition_leaf(workflow), taken)
        workflows.append(
            {
                "id": binding_id,
                "name": workflow,
                "external": {"alias": pin["alias"], "workflow": workflow},
            }
        )
        break
    else:
        raise WorkspaceError(f"this project has no subroutine {parent}")
    (clone.root / "project.json").write_text(
        json.dumps(project, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return definition_child(parent, binding_id)


def _free_id(requested: str, taken: set[str]) -> str:
    if requested not in taken:
        return requested
    for suffix in range(2, 1000):
        candidate = f"{requested}_{suffix}"
        if candidate not in taken:
            return candidate
    raise WorkspaceError(f"no free identifier near {requested}")


def _rename(arguments: argparse.Namespace) -> int:
    """Rename a graph identity and every module that named it, in one step."""

    clone = _here()
    original = clone.files()
    result = credential(clone.origin, clone.token, anonymous=True).rename(
        original,
        str(arguments.kind),
        (
            None
            if arguments.kind in {"workflow", "subroutine"}
            else str(arguments.subroutine)
        ),
        str(arguments.old),
        str(arguments.new),
    )
    files = _object(result.get("files"))
    touched = clone.write(cast("dict[str, str | None]", files), expected=original)
    print(f"Renamed {arguments.kind} {arguments.old} to {arguments.new}.")
    print(f"  {len(touched)} file(s) written; commit them when you are happy.")
    return 0


def _token(arguments: argparse.Namespace) -> int:
    """Tokens for an agent or CI, scoped to this repository.

    Only a session may manage these: a credential that can mint credentials is one that
    cannot be revoked by revoking it.
    """

    clone = _here()
    owner, name = clone.require_repository()
    address = f"{owner}/{name}"
    client = account()
    action = str(arguments.action)
    if action == "create":
        issued = client.create_token(address, str(arguments.name or "cli"))
        print(str(issued.get("token", "")))
        print(f"# id {issued.get('token_id')} -- shown once")
        print('# store it in .git/verdog.json as {"origin": ..., "token": ...}')
    elif action == "list":
        _rows(
            _objects(client.list_tokens(address).get("tokens")),
            (("NAME", "name"), ("ID", "id"), ("EXPIRES", "expires_at")),
        )
    else:
        client.revoke_token(address, str(arguments.name))
        print("Revoked.")
    return 0


def register(commands: Any) -> None:
    """Add every verb that is not about the files in front of you."""

    login = commands.add_parser("login", help="sign in with your GitHub account")
    login.add_argument("origin", nargs="?", help="service origin (or VERDOG_BACKEND_ORIGIN)")
    login.add_argument(
        "--github-token-stdin",
        action="store_true",
        help="read an existing GitHub token from stdin instead of prompting (used by VS Code)",
    )
    login.set_defaults(handler=_login)

    out = commands.add_parser("logout", help="end this session")
    out.set_defaults(handler=_logout)

    who = commands.add_parser("whoami", help="who you are signed in as")
    who.set_defaults(handler=_whoami)

    started = commands.add_parser("init", help="start a new project here")
    started.add_argument("name", nargs="?", help="what the workflow is called")
    started.add_argument("directory", nargs="?", help="where to put it (default: here)")
    started.add_argument(
        "--package",
        help="its canonical `<space>.<name>` Python package (required without a GitHub remote)",
    )
    started.add_argument("--origin", help="the Verdog service, if not the default")
    started.set_defaults(handler=_init)

    clone = commands.add_parser("clone", help="clone a project from GitHub")
    clone.add_argument("repository", help="owner/name")
    clone.add_argument("destination", nargs="?")
    clone.add_argument("--origin", help="the Verdog service (or VERDOG_BACKEND_ORIGIN)")
    clone.add_argument("--ssh", action="store_true", help="clone over SSH")
    clone.set_defaults(handler=_clone)

    access = commands.add_parser("access", help="what you may do in this repository")
    access.add_argument("--json", action="store_true", dest="as_json")
    access.set_defaults(handler=_access)

    catalogue = commands.add_parser(
        "catalogue", help="workflows published for reuse that you can see"
    )
    catalogue.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit one machine-readable response object",
    )
    catalogue.add_argument("--entry", help="one entry by id, with its preview")
    catalogue.add_argument(
        "--query", help="search titles, repositories, packages, and abstracts"
    )
    catalogue.add_argument(
        "--visibility",
        choices=("all", "public", "restricted", "mine"),
        default="all",
        help="which accessible records to list",
    )
    catalogue.add_argument(
        "--cursor", help="opaque continuation cursor from a previous page"
    )
    catalogue.add_argument(
        "--limit",
        type=int,
        choices=range(1, 101),
        metavar="1..100",
        help="records per page",
    )
    catalogue.set_defaults(handler=_catalogue, machine_json=True)

    describe = commands.add_parser(
        "describe", help="read the canonical catalogue metadata for one workflow"
    )
    describe.add_argument(
        "workflow",
        nargs="?",
        help="project-local workflow path; defaults to the entry workflow",
    )
    describe.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the metadata as one JSON object",
    )
    describe.set_defaults(handler=_describe, machine_json=True)

    publish = commands.add_parser("publish", help="offer a workflow at HEAD for reuse")
    publish.add_argument(
        "--description", required=True, help="one sentence: what this workflow does"
    )
    publish.add_argument(
        "workflow",
        nargs="?",
        help="project-local workflow path; defaults to the entry workflow",
    )
    publish.add_argument(
        "--private",
        action="store_true",
        help="deprecated compatibility flag; visibility follows the GitHub repository",
    )
    publish.set_defaults(handler=publish_workflow)

    retract = commands.add_parser("retract", help="withdraw an offer you published")
    retract.add_argument("entry", help="the entry id, from `verdog catalogue --json`")
    retract.set_defaults(handler=_retract)

    dropped = commands.add_parser("drop", help="stop depending on a pinned workflow")
    dropped.add_argument("alias", help="the direct dependency alias")
    dropped.add_argument(
        "--discard-node-code",
        action="store_true",
        help="permanently discard authored code and visits belonging to removed call nodes",
    )
    dropped.set_defaults(handler=drop_dependency)

    imported = commands.add_parser("import", help="reuse a published workflow")
    imported.add_argument("reference", help="owner/name@commit")
    imported.add_argument("workflow", help="the published workflow path to call")
    imported.add_argument(
        "--into", help="the project-local subroutine path that declares it"
    )
    imported.add_argument(
        "--alias",
        "--as",
        dest="alias",
        help="local name and checkout path (default: the published package)",
    )
    imported.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit one machine-readable import result",
    )
    imported.set_defaults(handler=_import, machine_json=True)

    bump = commands.add_parser("bump", help="move a pin to a newer commit")
    bump.add_argument("alias", help="the direct dependency alias to move")
    bump.add_argument(
        "--commit", help="a specific 40-character commit; default is the newest offered"
    )
    bump.set_defaults(handler=bump_dependency)

    rename = commands.add_parser("rename", help="rename a graph entity and its modules")
    rename.add_argument(
        "kind",
        choices=(
            "workflow",
            "subroutine",
            "node",
            "edge",
            "feature",
            "profile",
            "profile_parameter",
            "session",
            "session_parameter",
        ),
    )
    rename.add_argument("old")
    rename.add_argument("new")
    rename.add_argument(
        "--subroutine",
        help="project-local scope path; required unless renaming a workflow or subroutine",
    )
    rename.set_defaults(handler=_rename)

    token = commands.add_parser("token", help="tokens for an agent working unattended")
    token.add_argument("action", choices=("create", "list", "revoke"))
    token.add_argument("name", nargs="?", help="a label to create, or an id to revoke")
    token.set_defaults(handler=_token)


__all__ = ["register"]
