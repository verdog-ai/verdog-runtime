"""`verdog` -- generate, verify, and run local projects.

The service generates and analyzes project graphs. The CLI sends the required files,
writes returned changes, type-checks Python locally, and runs generated workflows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, cast

from verdog_runtime._run_model import RUN_HISTORY_SCHEMA_VERSION

from . import manage
from .api import Service, ServiceError
from .local import Clone, WorkspaceError, interpreter_in, open_clone
from .runner import run as run_workflow
from .runs import (
    RunCommandError,
    fork_run,
    list_checkpoints,
    list_runs,
    restart_run,
    resume_run,
)
from .session import SessionError, credential, validate_origin
from .sync import sync as sync_environment


def _service(clone: Clone) -> Service:
    """Use the configured compiler service even when this user has not signed in."""

    return credential(clone.origin, clone.token, anonymous=True)


def _report(diagnostics: list[Any]) -> int:
    """Print diagnostics the way a compiler would, and exit non-zero on an error.

    An agent reads the exit code before it reads the text, so a clean check has to be a
    zero and anything else has to not be.
    """

    if not diagnostics:
        print("No problems found.")
        return 0
    failed = False
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        entry = cast(dict[str, Any], item)
        severity = str(entry.get("severity", "error"))
        failed = failed or severity == "error"
        location = entry.get("path")
        position = [location] if isinstance(location, str) and location else []
        for coordinate in (entry.get("line"), entry.get("column")):
            if not position or not isinstance(coordinate, int):
                break
            position.append(str(coordinate))
        where = f"{':'.join(position)}: " if position else ""
        code = entry.get("code")
        suffix = f" [{code}]" if isinstance(code, str) and code else ""
        print(f"{severity}: {where}{entry.get('message', '')}{suffix}")
    return 1 if failed else 0


def _generate(arguments: argparse.Namespace) -> int:
    del arguments
    return generate_clone(open_clone(Path.cwd()))


def _analyze(arguments: argparse.Namespace) -> int:
    """Ask the service to analyze saved manifests without changing project files."""

    clone = open_clone(Path.cwd())
    result = _service(clone).analyze(clone.graph_files())
    if arguments.as_json:
        print(json.dumps(result, indent=2))
    else:
        definitions = cast(dict[str, dict[str, Any]], result["definitions"])
        for scope, definition in definitions.items():
            print(f"{scope}: {definition['status']}: {definition['reason']}")
    return 0


def generate_clone(clone: Clone) -> int:
    """Generate through the service and apply its response without type-checking."""

    files = clone.files()
    result = _service(clone).check(files)
    written = clone.write(
        cast("dict[str, str | None]", result["files"]), expected=files
    )
    print(f"Generated {len(written)} file(s).")
    return _report(cast(list[Any], result["diagnostics"]))


def _check(arguments: argparse.Namespace) -> int:
    """Send the whole project, write back the tree it should have, then type-check it.

    The whole project rather than a diff, because the service keeps nothing between calls --
    there is no base for a delta to be relative to. `project.json` comes back with the rest
    of the tree: it carries the refreshed manifest and generated hash.
    """

    return check_clone(
        open_clone(Path.cwd()), as_json=getattr(arguments, "as_json", False)
    )


def check_clone(clone: Clone, *, as_json: bool = False) -> int:
    """Verify a clone through the service, then type-check it locally."""

    return check_clone_result(clone, as_json=as_json)[0]


def check_clone_result(
    clone: Clone, *, as_json: bool = False
) -> tuple[int, dict[str, Any]]:
    """Check a clone and return both its status and compiler result."""

    files = clone.files()
    result = _service(clone).check(files)
    written = clone.write(
        cast("dict[str, str | None]", result.get("files") or {}), expected=files
    )
    diagnostics = cast(
        list[Any],
        result.get("diagnostics")
        if isinstance(result.get("diagnostics"), list)
        else [],
    )
    if as_json:
        return _check_as_json(clone, written, result, diagnostics), result
    print(f"Checked {len(files)} file(s); {len(written)} written.")
    structural = _report(diagnostics)
    typed = _type_check(clone)
    # `ty` prints its own verdict, and "All checks passed!" is *its* verdict on the types --
    # which now genuinely can be clean while the graph is not, because an unreachable node no
    # longer stops the tree from being generated. Left alone, its last line reads as though
    # the whole check passed.
    if structural and not typed:
        print("The types are fine; the problems above are not. `verdog check` failed.")
    return typed or structural, result


def _check_as_json(
    clone: Clone,
    written: list[str],
    result: dict[str, Any],
    diagnostics: list[Any],
) -> int:
    """One object on stdout, for a caller that parses rather than reads.

    An editor extension and an agent both want the same two things: what is wrong and where.
    Printing them as text and asking the reader to match
    two regular expressions -- one for the service's diagnostics, one for `ty`'s -- was a
    parser in the wrong place, and `ty` already speaks JSON.
    """

    typed, status = _type_check_json(clone)
    print(
        json.dumps(
            {
                "graph_hash": result.get("graph_hash", ""),
                "generated_from": result.get("generated_from", ""),
                "written": written,
                "diagnostics": diagnostics,
                "type_diagnostics": typed,
            },
            indent=2,
        )
    )
    failed = any(
        cast(dict[str, Any], item).get("severity") == "error"
        for item in diagnostics
        if isinstance(item, dict)
    )
    return status or (1 if failed else 0)


def _type_check_json(clone: Clone) -> tuple[list[Any], int]:
    """`ty`'s diagnostics, normalized to the shape the service already uses.

    `ty` speaks GitLab's code-quality JSON, whose keys are its own; translating here means
    every consumer -- an extension, an agent -- reads one diagnostic shape rather than two.
    Paths come back relative to the clone, because absolute paths in a project file are
    nobody's business but this machine's.
    """

    commands = clone.type_check_commands(output_format="gitlab")
    found: list[Any] = []
    status = 0
    for command in commands:
        finished = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, cwd=clone.root, capture_output=True, text=True, check=False
        )
        status = status or finished.returncode
        try:
            decoded = json.loads(finished.stdout or "[]")
        except ValueError:
            found.append(
                {
                    "severity": "error",
                    "message": (finished.stderr or finished.stdout).strip(),
                }
            )
            status = status or 1
            continue
        for item in cast(list[Any], decoded) if isinstance(decoded, list) else []:
            if not isinstance(item, dict):
                continue
            entry = cast(dict[str, Any], item)
            location = entry.get("location")
            place = cast(dict[str, Any], location) if isinstance(location, dict) else {}
            positions = place.get("positions")
            begin = (
                cast(dict[str, Any], cast(dict[str, Any], positions).get("begin", {}))
                if isinstance(positions, dict)
                else {}
            )
            path = str(place.get("path", ""))
            try:
                path = str(Path(path).resolve().relative_to(clone.root.resolve()))
            except (OSError, ValueError):
                pass
            found.append(
                {
                    "severity": "warning"
                    if entry.get("severity") == "minor"
                    else "error",
                    "code": entry.get("check_name", ""),
                    "message": entry.get("description", ""),
                    "path": path,
                    "line": begin.get("line"),
                    "column": begin.get("column"),
                }
            )
    return found, status


def _type_check(clone: Clone) -> int:
    """Run the local type check, whose diagnostics `ty` prints itself.

    After the structural check and not instead of it: the generated declarations have
    just been written, so this is the first moment the tree on disk is the tree the
    service would have produced.
    """

    commands = clone.type_check_commands()
    # `ty` writes straight to the terminal, so anything still sitting in our own buffer
    # would print after it and describe the wrong step.
    sys.stdout.flush()
    status = 0
    for command in commands:
        finished = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, cwd=clone.root, check=False
        )
        status = status or finished.returncode
    return status


def _sync(arguments: argparse.Namespace) -> int:
    clone = open_clone(Path.cwd())
    workflow = getattr(arguments, "workflow", None)
    if not getattr(arguments, "as_json", False):
        return sync_environment(
            clone,
            workflow_id=workflow,
            only_binary=bool(getattr(arguments, "only_binary", False)),
        )
    with redirect_stdout(sys.stderr):
        status = sync_environment(
            clone,
            workflow_id=workflow,
            only_binary=bool(getattr(arguments, "only_binary", False)),
        )
    if status:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": {
                        "code": "sync.install_failed",
                        "message": "the declared dependencies could not be installed",
                    },
                },
                indent=2,
            )
        )
        return status
    payload: dict[str, Any] = {"status": "ready"}
    if workflow is not None:
        definition = clone.workflow_definition(str(workflow))
        environment = clone.environment(definition).resolve()
        interpreter = interpreter_in(environment)
        if interpreter is None:
            raise WorkspaceError(f"no interpreter in {environment}")
        payload.update(
            {
                "workflow_id": definition.local_id,
                "environment": str(environment),
                "interpreter": str(interpreter),
                "only_binary": bool(getattr(arguments, "only_binary", False)),
                "requirements": sorted(clone.workflow_requirements(definition)),
            }
        )
    print(json.dumps(payload, indent=2))
    return 0


def _run(arguments: argparse.Namespace) -> int:
    return run_workflow(
        open_clone(Path.cwd()),
        arguments.output_dir,
        workflow_id=arguments.workflow,
        arguments=tuple(arguments.project_arguments),
        checkpointing=arguments.checkpointing,
    )


def _runs(arguments: argparse.Namespace) -> int:
    return list_runs(
        open_clone(Path.cwd()),
        workflow=arguments.workflow,
        statuses=tuple(arguments.status),
        as_json=arguments.as_json,
    )


def _checkpoints(arguments: argparse.Namespace) -> int:
    return list_checkpoints(
        open_clone(Path.cwd()),
        reference=arguments.run,
        as_json=arguments.as_json,
    )


def _resume(arguments: argparse.Namespace) -> int:
    return resume_run(
        open_clone(Path.cwd()),
        reference=arguments.run,
        retry_incomplete=arguments.retry_incomplete,
        as_json=arguments.as_json,
    )


def _restart(arguments: argparse.Namespace) -> int:
    return restart_run(
        open_clone(Path.cwd()),
        reference=arguments.run,
        sessions=arguments.sessions,
        arguments=(
            tuple(arguments.project_arguments)
            if arguments.arguments_overridden
            else None
        ),
        as_json=arguments.as_json,
    )


def _fork(arguments: argparse.Namespace) -> int:
    return fork_run(
        open_clone(Path.cwd()),
        reference=arguments.run,
        checkpoint=arguments.checkpoint,
        sessions=arguments.sessions,
        as_json=arguments.as_json,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verdog", description=__doc__)
    parser.add_argument(
        "--backend-origin",
        help="editor backend origin; compiler calls are anonymous, sessions arrive through stdin",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate", help="generate through the service and write the returned files"
    )
    generate.set_defaults(handler=_generate)

    analyze = commands.add_parser(
        "analyze", help="analyze saved graph termination through the service without changing files"
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the analysis as one JSON object",
    )
    analyze.set_defaults(handler=_analyze)

    check = commands.add_parser(
        "check", help="verify this project and write the generated tree"
    )
    check.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the verdict as one JSON object, for an editor or an agent",
    )
    check.set_defaults(handler=_check)

    provision = commands.add_parser(
        "sync", help="install this project and its nested dependencies"
    )
    provision.add_argument(
        "workflow",
        nargs="?",
        metavar="WORKFLOW",
        help="sync only this project-local workflow path (for example main__nested)",
    )
    provision.add_argument(
        "--only-binary",
        action="store_true",
        dest="only_binary",
        help="refuse source distributions, so nothing is built and no build code runs",
    )
    provision.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit one machine-readable environment result",
    )
    provision.set_defaults(handler=_sync, machine_json=True)

    execute = commands.add_parser("run", help="execute the workflow on this machine")
    execute.add_argument(
        "workflow",
        nargs="?",
        metavar="WORKFLOW",
        help="project-local workflow path (the root workflow by default)",
    )
    execute.add_argument(
        "--output-dir",
        type=Path,
        help="write this run's outputs to an empty directory",
    )
    execute.add_argument(
        "--checkpointing",
        choices=("off", "auto", "required"),
        default="auto",
        help="checkpoint completed boundaries (default: auto)",
    )
    execute.set_defaults(handler=_run)

    runs = commands.add_parser("runs", help="list this project's local workflow runs")
    runs.add_argument(
        "workflow",
        nargs="?",
        metavar="WORKFLOW",
        help="show only this project-local workflow",
    )
    runs.add_argument(
        "--status",
        action="append",
        default=[],
        choices=("running", "interrupted", "failed", "succeeded"),
        help="show only this status; repeat to include several statuses",
    )
    runs.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit one versioned machine-readable run list",
    )
    runs.set_defaults(handler=_runs, machine_json=True)

    checkpoints = commands.add_parser(
        "checkpoints", help="list the committed checkpoints of a local run"
    )
    checkpoints.add_argument(
        "run",
        nargs="?",
        metavar="RUN",
        help="run id, unique id prefix, managed directory name, or output path",
    )
    checkpoints.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit one versioned machine-readable checkpoint list",
    )
    checkpoints.set_defaults(handler=_checkpoints, machine_json=True)

    resume = commands.add_parser(
        "resume", help="continue a run at its latest exactly committed boundary"
    )
    resume.add_argument(
        "run",
        nargs="?",
        metavar="RUN",
        help="run id, unique id prefix, managed directory name, or output path",
    )
    resume.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="retry an external invocation whose completion cannot be established",
    )
    resume.add_argument(
        "--json", action="store_true", dest="as_json", help="emit one JSON result"
    )
    resume.set_defaults(handler=_resume, machine_json=True)

    restart = commands.add_parser(
        "restart", help="start a new run from a prior run's launch"
    )
    restart.add_argument(
        "run",
        nargs="?",
        metavar="RUN",
        help="run id, unique id prefix, managed directory name, or output path",
    )
    restart.add_argument(
        "--sessions",
        required=True,
        choices=("branch", "fresh"),
        help="branch committed conversations or start fresh ones",
    )
    restart.add_argument(
        "--json", action="store_true", dest="as_json", help="emit one JSON result"
    )
    restart.set_defaults(handler=_restart, machine_json=True)

    fork = commands.add_parser(
        "fork", help="continue a committed checkpoint as a new run"
    )
    fork.add_argument(
        "run",
        nargs="?",
        metavar="RUN",
        help="run id, unique id prefix, managed directory name, or output path",
    )
    fork.add_argument(
        "--checkpoint",
        required=True,
        type=int,
        metavar="N",
        help="checkpoint sequence to continue",
    )
    fork.add_argument(
        "--sessions",
        required=True,
        choices=("branch", "fresh"),
        help="branch committed conversations or start fresh ones",
    )
    fork.add_argument(
        "--json", action="store_true", dest="as_json", help="emit one JSON result"
    )
    fork.set_defaults(handler=_fork, machine_json=True)

    # Everything that is not about the files in front of you: signing in, cloning from
    # GitHub, asking what you may do, publishing and importing, issuing tokens. `save` and
    # `pull` are absent from both halves -- git does those.
    manage.register(commands)
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Split the project CLI at ``--`` before argparse can reinterpret it."""

    raw = list(sys.argv[1:] if argv is None else argv)
    project_arguments: list[str] = []
    arguments_overridden = False
    command_arguments = raw
    while command_arguments and command_arguments[0].split("=", 1)[0] == "--backend-origin":
        command_arguments = command_arguments[1 if "=" in command_arguments[0] else 2 :]
    command = command_arguments[:1]
    if command and command[0] in {"run", "restart"} and "--" in raw:
        boundary = raw.index("--")
        project_arguments = raw[boundary + 1 :]
        raw = raw[:boundary]
        arguments_overridden = command == ["restart"]
    parsed = build_parser().parse_args(raw)
    if parsed.command in {"run", "restart"}:
        parsed.project_arguments = project_arguments
    if parsed.command == "restart":
        parsed.arguments_overridden = arguments_overridden
    return parsed


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        if arguments.backend_origin is not None:
            os.environ["VERDOG_BACKEND_ORIGIN"] = validate_origin(arguments.backend_origin)
        handler = cast(Callable[[argparse.Namespace], int], arguments.handler)
        return handler(arguments)
    except (ServiceError, WorkspaceError, SessionError) as error:
        if getattr(arguments, "machine_json", False) and getattr(
            arguments, "as_json", False
        ):
            code = (
                error.code
                if isinstance(error, ServiceError)
                else error.code
                if isinstance(error, RunCommandError)
                else "session.error"
                if isinstance(error, SessionError)
                else "workspace.error"
            )
            payload: dict[str, Any] = {
                "status": "error",
                "error": {
                    "code": code,
                    "message": (
                        error.machine_message
                        if isinstance(error, ServiceError)
                        else str(error)
                    ),
                },
            }
            if isinstance(error, RunCommandError):
                payload = {
                    "schema_version": RUN_HISTORY_SCHEMA_VERSION,
                    "operation": arguments.command,
                    **payload,
                }
            if isinstance(error, ServiceError) and error.details is not None:
                cast(dict[str, Any], payload["error"])["details"] = error.details
            if isinstance(error, RunCommandError) and error.details is not None:
                cast(dict[str, Any], payload["error"])["details"] = error.details
            print(json.dumps(payload, indent=2))
            return 1
        print(f"verdog: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # `verdog login` waits on a person, so Ctrl+C is an ordinary way to stop.
        print("\nverdog: stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
