from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never, cast

import pytest

from verdog_runtime import entry
from verdog_runtime._lifecycle import LifecycleCommand, encode_lifecycle_command
from verdog_runtime._run_store import RunStatus, RunStore
from verdog_runtime.interpreter import CheckpointPolicy, SessionPolicy


def _source(root: Path) -> RunStore:
    output = root / ".verdog/runs/source"
    output.mkdir(parents=True)
    store = RunStore.create(
        output,
        project_root=root,
        workflow_id="example.main",
        definition_id="example.main.entry",
        module="example.workflows.main",
        workflow_arguments=("--input.seed", "3"),
        run_id="11111111-1111-4111-8111-111111111111",
    )
    store.update(status=RunStatus.INTERRUPTED)
    return store


def _definition() -> tuple[object, object]:
    return SimpleNamespace(id="example.main"), SimpleNamespace(
        input=3,
        params={"seed": 3},
        runtime=SimpleNamespace(mode="test"),
    )


def _request(
    root: Path,
    source: Path,
    *,
    operation: Any,
    sessions: str,
    checkpoint: int | None,
    mode: str,
    arguments: tuple[str, ...],
    retry_incomplete: bool = False,
    as_json: bool = True,
) -> Any:
    return LifecycleCommand(
        operation=operation,
        root=root,
        definition_id="example.main",
        definition_module="example.workflows.main",
        source_output=source,
        sessions=cast(Any, sessions),
        checkpoint=checkpoint,
        arguments_mode=cast(Any, mode),
        retry_incomplete=retry_incomplete,
        as_json=as_json,
        arguments=arguments,
    )


def test_default_output_groups_workflows_and_preserves_explicit_paths(
    tmp_path: Path,
) -> None:
    main = entry._output_directory(  # pyright: ignore[reportPrivateUsage]
        tmp_path, None, "example.main"
    )
    refinement = entry._output_directory(  # pyright: ignore[reportPrivateUsage]
        tmp_path, None, "example.main__refine_choose"
    )
    explicit = tmp_path / "custom-output"

    assert main.parent == tmp_path / ".verdog/runs/main"
    assert refinement.parent == tmp_path / ".verdog/runs/main__refine_choose"
    assert (
        entry._output_directory(tmp_path, explicit, "example.main")  # pyright: ignore[reportPrivateUsage]
        == explicit
    )


@pytest.mark.parametrize("operation", ("resume", "restart", "fork"))
def test_lifecycle_keeps_resume_output_and_groups_new_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    source = _source(tmp_path)

    def configured_definition(*_args: object) -> tuple[object, object]:
        return _definition()

    def dispatch(_prepared: object) -> str:
        return "done"

    monkeypatch.setattr(entry, "_configured_definition", configured_definition)
    monkeypatch.setattr(entry, "_dispatch_lifecycle", dispatch)
    request = _request(
        tmp_path,
        source.output_dir,
        operation=operation,
        sessions="restore" if operation == "resume" else "branch",
        checkpoint=1,
        mode="reused" if operation == "restart" else "checkpoint",
        arguments=("--input.seed", "3"),
    )

    result = entry._attempt_lifecycle(request)  # pyright: ignore[reportPrivateUsage]
    assert result.problem is None
    assert result.target_output is not None
    if operation == "resume":
        assert result.target_output == source.output_dir
    else:
        assert result.target_output.parent == tmp_path / ".verdog/runs/main"


def test_resume_emits_one_json_document_and_redirects_workflow_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = _source(root)
    received: dict[str, object] = {}

    class FakeDispatcher:
        def __init__(self, *, project_root: Path) -> None:
            received["project_root"] = project_root

        def resume(self, definition: object, **kwargs: object) -> str:
            print("authored workflow chatter")
            os.write(1, b"native child chatter\n")
            received.update(kwargs)
            source.update(status=RunStatus.SUCCEEDED)
            return "done"

    def configured_definition(*_args: object) -> tuple[object, object]:
        return _definition()

    monkeypatch.setattr(entry.interpreter, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(entry, "_configured_definition", configured_definition)

    request = _request(
        root,
        source.output_dir,
        operation="resume",
        sessions="restore",
        checkpoint=7,
        mode="checkpoint",
        retry_incomplete=True,
        arguments=("--input.seed", "3"),
    )
    status = entry._lifecycle(request)  # pyright: ignore[reportPrivateUsage]

    assert status == 0
    captured = capfd.readouterr()
    document = cast(dict[str, Any], json.loads(captured.out))
    assert document["operation"] == "resume"
    assert document["status"] == "succeeded"
    assert document["source_run_id"] == "11111111-1111-4111-8111-111111111111"
    assert document["source_checkpoint"] == 7
    assert document["sessions"] == "restore"
    assert document["arguments"] == "checkpoint"
    assert cast(dict[str, object], document["run"])["status"] == "succeeded"
    assert "authored workflow chatter" in captured.err
    assert "native child chatter" in captured.err
    assert received["output_dir"] == source.output_dir
    assert received["retry_incomplete"] is True


def test_restart_allocates_a_child_and_preserves_argument_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = _source(root)
    target = root / ".verdog/runs/restarted"
    received: dict[str, object] = {}

    class FakeDispatcher:
        def __init__(self, *, project_root: Path) -> None:
            received["project_root"] = project_root

        def restart(
            self, definition: object, input: object, **kwargs: object
        ) -> str:
            received["input"] = input
            received.update(kwargs)
            target.mkdir(parents=True)
            store = RunStore.create(
                target,
                project_root=root,
                workflow_id="example.main",
                definition_id="example.main.entry",
                module="example.workflows.main",
                workflow_arguments=cast(
                    tuple[str, ...], kwargs["workflow_arguments"]
                ),
                run_id="22222222-2222-4222-8222-222222222222",
            )
            store.update(status=RunStatus.SUCCEEDED)
            return "done"

    def configured_definition(*_args: object) -> tuple[object, object]:
        return _definition()

    def output_directory(*_args: object) -> Path:
        return target

    monkeypatch.setattr(entry.interpreter, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(entry, "_configured_definition", configured_definition)
    monkeypatch.setattr(entry, "_output_directory", output_directory)

    request = _request(
        root,
        source.output_dir,
        operation="restart",
        sessions="branch",
        checkpoint=4,
        mode="overridden",
        arguments=("--input.seed", "9"),
    )
    status = entry._lifecycle(request)  # pyright: ignore[reportPrivateUsage]

    assert status == 0
    document = cast(dict[str, Any], json.loads(capsys.readouterr().out))
    assert document["operation"] == "restart"
    assert document["source_checkpoint"] == 4
    assert document["sessions"] == "branch"
    assert document["arguments"] == "overridden"
    assert cast(dict[str, object], document["run"])["id"] == (
        "22222222-2222-4222-8222-222222222222"
    )
    assert received["source_output_dir"] == source.output_dir
    assert received["output_dir"] == target
    assert received["sessions"] is SessionPolicy.BRANCH
    assert received["source_checkpoint"] == 4
    assert received["workflow_arguments"] == ("--input.seed", "9")
    assert received["arguments_mode"] == "overridden"


def test_lifecycle_rejects_a_changed_recorded_argument_vector_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = _source(root)

    def should_not_import(*_args: object) -> Never:
        pytest.fail("definition must not be imported")

    monkeypatch.setattr(entry, "_configured_definition", should_not_import)

    request = _request(
        root,
        source.output_dir,
        operation="fork",
        sessions="fresh",
        checkpoint=2,
        mode="checkpoint",
        arguments=("--input.seed", "changed"),
    )
    status = entry._lifecycle(request)  # pyright: ignore[reportPrivateUsage]

    assert status == 1
    document = cast(dict[str, Any], json.loads(capsys.readouterr().out))
    assert document["operation"] == "fork"
    assert document["status"] == "error"
    assert cast(dict[str, object], document["error"])["code"] == (
        "run.operation_failed"
    )


def test_prelaunch_restart_failure_does_not_report_the_source_as_its_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = _source(root)

    def unavailable_definition(*_args: object) -> Never:
        raise ImportError("workflow cannot be imported")

    monkeypatch.setattr(entry, "_configured_definition", unavailable_definition)
    request = _request(
        root,
        source.output_dir,
        operation="restart",
        sessions="fresh",
        checkpoint=None,
        mode="reused",
        arguments=("--input.seed", "3"),
    )

    assert entry._lifecycle(request) == 1  # pyright: ignore[reportPrivateUsage]
    document = cast(dict[str, Any], json.loads(capsys.readouterr().out))
    assert document["operation"] == "restart"
    assert document["status"] == "error"
    assert "run" not in document
    assert cast(dict[str, object], document["error"])["message"] == (
        "workflow cannot be imported"
    )


def test_runtime_failure_returns_a_manifest_backed_failed_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = _source(root)

    class FakeDispatcher:
        def __init__(self, *, project_root: Path) -> None:
            pass

        def resume(self, definition: object, **kwargs: object) -> None:
            source.update(status=RunStatus.FAILED)
            raise ValueError("node failed")

    def configured_definition(*_args: object) -> tuple[object, object]:
        return _definition()

    monkeypatch.setattr(entry.interpreter, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(entry, "_configured_definition", configured_definition)

    request = _request(
        root,
        source.output_dir,
        operation="resume",
        sessions="restore",
        checkpoint=7,
        mode="checkpoint",
        arguments=("--input.seed", "3"),
    )
    status = entry._lifecycle(request)  # pyright: ignore[reportPrivateUsage]

    assert status == 1
    document = cast(dict[str, Any], json.loads(capsys.readouterr().out))
    assert document["status"] == "failed"
    assert cast(dict[str, object], document["run"])["status"] == "failed"
    assert cast(dict[str, object], document["error"]) == {
        "code": "run.operation_failed",
        "message": "node failed",
    }


def test_main_decodes_the_versioned_lifecycle_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[object] = []

    def lifecycle(request: object) -> int:
        received.append(request)
        return 19

    command = LifecycleCommand(
        operation="fork",
        root=tmp_path.resolve(),
        definition_id="example.main",
        definition_module="example.workflows.main",
        source_output=(tmp_path / "source").resolve(),
        sessions="fresh",
        checkpoint=12,
        arguments_mode="checkpoint",
        retry_incomplete=False,
        as_json=True,
        arguments=("--input.seed", "3", "--verdog-lifecycle"),
    )
    monkeypatch.setattr(entry, "_lifecycle", lifecycle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verdog_runtime.entry",
            "--verdog-lifecycle",
            encode_lifecycle_command(command),
        ],
    )

    assert entry.main() == 19
    request = received[0]
    assert isinstance(request, LifecycleCommand)
    assert request == command


def test_run_protocol_preserves_workflow_arguments_with_internal_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[object] = []

    def run(request: object) -> int:
        received.append(request)
        return 0

    monkeypatch.setattr(entry, "_run", run)
    status = entry._run_main(  # pyright: ignore[reportPrivateUsage]
        (
            "verdog_runtime.entry",
            str(tmp_path),
            "example.main",
            "example.workflows.main",
            "",
            "--verdog-checkpointing=auto",
            "--verdog-checkpointing=required",
        )
    )

    assert status == 0
    request = received[0]
    assert isinstance(request, entry._RunRequest)  # pyright: ignore[reportPrivateUsage]
    assert request.checkpointing is CheckpointPolicy.AUTO
    assert request.arguments == ("--verdog-checkpointing=required",)


@pytest.mark.parametrize(
    ("operation", "sessions", "checkpoint", "mode"),
    [
        ("resume", "branch", 1, "checkpoint"),
        ("resume", "restore", None, "checkpoint"),
        ("restart", "fresh", 1, "reused"),
        ("restart", "branch", None, "reused"),
        ("fork", "fresh", None, "checkpoint"),
        ("fork", "restore", 1, "checkpoint"),
        ("fork", "fresh", 0, "checkpoint"),
    ],
)
def test_internal_lifecycle_contract_rejects_ambiguous_combinations(
    tmp_path: Path,
    operation: Any,
    sessions: str,
    checkpoint: int | None,
    mode: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = _source(root).manifest()
    request = _request(
        root,
        root / ".verdog/runs/source",
        operation=operation,
        sessions=sessions,
        checkpoint=checkpoint,
        mode=mode,
        arguments=source.launch.workflow_arguments,
    )
    with pytest.raises(RuntimeError):
        entry._require_lifecycle_contract(  # pyright: ignore[reportPrivateUsage]
            request,
            source,
        )
