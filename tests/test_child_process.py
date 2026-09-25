from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
import venv
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn, TypeVar, cast, override

import pytest
import typing_extensions
import verdog_runtime
from layout_helpers import legacy_graph_create
from report_helpers import call_reports, table_rows
from verdog_runtime._child_checkpoint import (
    ChildCheckpointBundle,
    decode_child_checkpoint,
    encode_child_checkpoint,
    mark_child_checkpoint_fork,
)
from verdog_runtime._protocol import (
    CallFrame,
    CheckpointFrame,
    ErrorDetail,
    ErrorFrame,
    EventFrame,
    OutgoingFrame,
    ReplyFrame,
    SuccessFrame,
    TimingFrame,
    decode_binary_payload,
    decode_call,
    decode_payload,
    decode_reply,
    encode_binary_payload,
    encode_frame,
    encode_payload,
)
from verdog_runtime._run_store import (
    Boundary,
    CheckpointPolicy,
    RunStatus,
    RunStore,
)
from verdog_runtime._statistics import TimingRecord
from verdog_runtime.child import (
    _read_response,  # pyright: ignore[reportPrivateUsage]
    _request,  # pyright: ignore[reportPrivateUsage]
    _serve,  # pyright: ignore[reportPrivateUsage]
)
from verdog_runtime.child import (
    invoke as invoke_child,
)
from verdog_runtime.declarations import (
    CallContext,
    CallVisitDefinition,
    EdgeDefinition,
    GraphDefinition,
    NodeDefinition,
    ParameterAddress,
    ParameterType,
    PortDefinition,
    RemoteWorkflowError,
    SubroutineCall,
    SubroutineDefinition,
    Success,
    VisitDefinition,
    WorkflowCall,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId, RunId
from verdog_runtime.interpreter import (
    CancellationToken,
    Dispatcher,
    EdgeExecution,
    ExecutionCancelled,
    NodeExecution,
    SessionPolicy,
)
from verdog_runtime.interpreter._continuation import (
    CallFrameSnapshot,
    GraphFrameSnapshot,
    decode_continuation,
)
from verdog_runtime.interpreter.execution import (
    _encoded_id,  # pyright: ignore[reportPrivateUsage]
)


def _edge(
    edge_id: str,
    source: NodeId,
    target: NodeId,
    implementation: object | None = None,
    input_type: Any = object,
) -> EdgeDefinition:
    return EdgeDefinition(
        id=EdgeId(edge_id),
        name=edge_id,
        source=source,
        target=target,
        conditions=(),
        effects=(),
        visit=(
            None
            if implementation is None
            else VisitDefinition(
                implementation=cast(Any, implementation),
            )
        ),
    )


def _output(tmp_path: Path, name: str) -> Path:
    return tmp_path / "outputs" / name


def _only_call_output(report_dir: Path, /) -> Path:
    reports = call_reports(report_dir / "config.md")
    assert len(reports) == 1
    return reports[0].parent


def _environment(project: Path) -> None:
    environment = project / ".verdog" / "environments" / "main"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    sites = [
        *environment.glob("lib/python3.*/site-packages"),
        environment / "Lib" / "site-packages",
    ]
    site = next(path for path in sites if path.is_dir())
    runtime_root = Path(verdog_runtime.__file__).resolve().parent.parent
    dependencies = Path(typing_extensions.__file__).resolve().parent
    (site / "_verdog_runtime.pth").write_text(
        f"{runtime_root}\n{dependencies}\n", encoding="utf-8"
    )


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
_OMITTED = object()


@dataclass(frozen=True, slots=True)
class EmptyState:
    pass


class WireMode(StrEnum):
    FAST = "fast"
    CAREFUL = "careful"


@dataclass(frozen=True, slots=True, kw_only=True)
class WireInput:
    root: Path
    mode: WireMode
    limits: tuple[int, ...]
    note: str | None = None


@dataclass(slots=True)
class MutablePayload:
    values: list[int]


def _definition(
    graph: GraphDefinition[InputT, OutputT, None, object],
    envelope_id: str = "parent.workflow",
) -> WorkflowDefinition[InputT, OutputT, None, object]:
    subroutine = SubroutineDefinition(graph=graph)
    module_name = "runtime_test_child_process_entry"
    module = ModuleType(module_name)
    module.__dict__["definition"] = lambda: subroutine
    sys.modules[module_name] = module
    params_types: dict[ParameterAddress, ParameterType] = {
        (".", graph.id): graph.params_type
    }
    for node in graph.nodes:
        if isinstance(node.operation, SubroutineCall):
            params_types.update(node.operation.params_types)
    return WorkflowDefinition(
        id=GraphId(envelope_id),
        input_type=cast(Any, object),
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module_name,
            params_types=params_types,
            profile_arguments={
                parameter.id: parameter.id for parameter in graph.profile_parameters
            },
            session_arguments={
                parameter.id: parameter.id for parameter in graph.session_parameters
            },
        ),
        configuration=WorkflowConfiguration(),
    )


def _child_project(
    root: Path,
    *,
    typed_params: bool = False,
    child_output_module: str | None = None,
    enter_id: str = "enter",
) -> Path:
    project = root / "child"
    package = project / "src" / "child_project"
    workflow = package / "workflows" / "main"
    subroutine = package / "subroutines" / "main"
    workflow.mkdir(parents=True)
    subroutine.mkdir(parents=True)
    (subroutine / "subroutines").mkdir()
    (subroutine / "workflows").mkdir()
    (project / "project.json").write_text(
        json.dumps(
            {
                "schema_version": 32,
                "package": "child_project",
                "externals": [],
                "workflow": {
                    "id": "main",
                    "name": "Main",
                    "subroutine": "main",
                    "profile_arguments": {},
                    "session_arguments": {},
                    "dependencies": [],
                },
                "subroutine": {
                    "id": "main",
                    "name": "Main body",
                    "workflows": [],
                    "subroutines": [],
                    "nodes": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "subroutines/__init__.py").write_text("", encoding="utf-8")
    (package / "workflows/__init__.py").write_text("", encoding="utf-8")
    (subroutine / "subroutines/__init__.py").write_text("", encoding="utf-8")
    (subroutine / "workflows/__init__.py").write_text("", encoding="utf-8")
    params_type = "Params" if typed_params else "type(None)"
    params_import = (
        "from child_project.subroutines.main import Params\n" if typed_params else ""
    )
    (workflow / "__init__.py").write_text(
        "from functools import cache\n"
        "from verdog_runtime.declarations import "
        "SubroutineCall, WorkflowConfiguration, WorkflowDefinition\n"
        "from verdog_runtime.declarations.ids import GraphId\n"
        + params_import
        + "@cache\n"
        "def definition():\n"
        "    return WorkflowDefinition(id=GraphId('child_project.main'), "
        "input_type=int, entry=SubroutineCall("
        "definition_id=GraphId('child_project.main'), "
        "definition_module='child_project.subroutines.main', "
        f"params_types={{('.', GraphId('child_project.main')): {params_type}}}, "
        "profile_arguments={}, session_arguments={}), "
        "configuration=WorkflowConfiguration())\n",
        encoding="utf-8",
    )
    params_declaration = (
        "@dataclass(frozen=True, slots=True, kw_only=True)\n"
        "class Params:\n"
        "    delta: int = 1\n"
        if typed_params
        else ""
    )
    increment = (
        "(0 if context.params is None else context.params.delta)"
        if typed_params
        else "1"
    )
    child_only_output = (
        f"        from {child_output_module} import ChildOnly\n"
        "        return Success(output=ChildOnly(), state=state)"
        if child_output_module is not None
        else "        return Success(output=EmptyState(), state=state)"
    )
    (subroutine / "__init__.py").write_text(
        f"""
import os
import shutil
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from verdog_runtime.declarations import (
    EdgeDefinition, GraphDefinition, NodeDefinition,
    PortDefinition, Python, Success,
    SubroutineDefinition, VisitDefinition,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId


{params_declaration}
@dataclass(frozen=True, slots=True)
class EmptyState:
    pass


def work(input, state, context, /):
    print("ordinary child stdout")
    (context.output_dir / "visited").write_text("child", encoding="utf-8")
    if input == -1:
        time.sleep(5)
    if input in (-4, -9):
        os._exit(17)
    if input == -5:
        Path(os.environ["VERDOG_TEST_PID_FILE"]).write_text(str(os.getpid()))
        time.sleep(30)
    if input == -2:
        raise ValueError("expected child failure")
    if input == -8:
{child_only_output}
    if input == -3:
        expected_environment = os.path.join(
            os.getcwd(), ".verdog", "environments", "main"
        )
        assert os.path.samefile(os.environ["VIRTUAL_ENV"], expected_environment), (
            os.environ["VIRTUAL_ENV"], expected_environment)
        resolved_python = shutil.which("python")
        assert resolved_python is not None
        assert os.path.samefile(resolved_python, sys.executable), (resolved_python, sys.executable)
    if not isinstance(input, int):
        return Success(output=input, state=state)
    return Success(output=(input + {increment}, os.getpid()), state=state)


ENTER = PortDefinition(id=NodeId({enter_id!r}))
EXIT = PortDefinition(id=NodeId("exit"))
FAILURE = PortDefinition(id=NodeId("failure"))
WORK = NodeDefinition(
    id=NodeId("work"),
    name="Work",
    state_type=EmptyState,
    operation=Python(),
)
GRAPH = GraphDefinition(
    id=GraphId("child_project.main"),
    params_type={params_type},
    enter=ENTER,
    exit=EXIT,
    failure=FAILURE,
    nodes=(WORK,),
    edges=(
        EdgeDefinition(id=EdgeId("in"), name="in", source=ENTER.id, target=WORK.id,
                       conditions=(), effects=(),
                       visit=VisitDefinition(implementation=work)),
        EdgeDefinition(id=EdgeId("out"), name="out", source=WORK.id, target=EXIT.id,
                       conditions=(), effects=()),
    ),
)
@cache
def definition():
    return SubroutineDefinition(graph=GRAPH)
""".lstrip(),
        encoding="utf-8",
    )
    _environment(project)
    return project


def _bridge_project(
    root: Path,
    project_path: str,
    target_definition_id: str,
    *,
    in_process: bool = False,
    enter_id: str = "enter",
) -> Path:
    project = root / "branch"
    package = project / "src" / "branch_project"
    workflow = package / "workflows" / "main"
    subroutine = package / "subroutines" / "main"
    workflow.mkdir(parents=True)
    subroutine.mkdir(parents=True)
    (subroutine / "subroutines").mkdir()
    (subroutine / "workflows").mkdir()
    (project / "project.json").write_text(
        json.dumps(
            {
                "schema_version": 32,
                "package": "branch_project",
                "externals": [],
                "workflow": {
                    "id": "main",
                    "name": "Main",
                    "subroutine": "main",
                    "profile_arguments": {},
                    "session_arguments": {},
                    "dependencies": [],
                },
                "subroutine": {
                    "id": "main",
                    "name": "Main body",
                    "workflows": [],
                    "subroutines": [],
                    "nodes": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "subroutines/__init__.py").write_text("", encoding="utf-8")
    (package / "workflows/__init__.py").write_text("", encoding="utf-8")
    (subroutine / "subroutines/__init__.py").write_text("", encoding="utf-8")
    (subroutine / "workflows/__init__.py").write_text("", encoding="utf-8")
    (workflow / "__init__.py").write_text(
        "from functools import cache\n"
        "from verdog_runtime.declarations import "
        "SubroutineCall, WorkflowConfiguration, WorkflowDefinition\n"
        "from verdog_runtime.declarations.ids import GraphId\n"
        "@cache\n"
        "def definition():\n"
        "    return WorkflowDefinition(id=GraphId('branch_project.main'), "
        "input_type=int, entry=SubroutineCall("
        "definition_id=GraphId('branch_project.main'), "
        "definition_module='branch_project.subroutines.main', "
        "params_types={('.', GraphId('branch_project.main')): type(None)}, "
        "profile_arguments={}, session_arguments={}), "
        "configuration=WorkflowConfiguration())\n",
        encoding="utf-8",
    )
    call = "SubroutineCall" if in_process else "WorkflowCall"
    target_package, _, target_name = target_definition_id.rpartition(".")
    target_module = (
        f"{target_package}.subroutines.{target_name}"
        if in_process
        else f"{target_package}.workflows.{target_name}"
    )
    target_params_id = (
        target_definition_id
        if in_process
        else target_definition_id.rsplit(".", maxsplit=1)[0] + ".main"
    )
    nested_params = (
        f"            ({project_path + '/external/child'!r}, "
        "GraphId('child_project.main')): type(None),\n"
        if target_definition_id == "branch_project.main"
        else ""
    )
    call_params = (
        f"        params_types={{\n"
        f"            ({project_path!r}, GraphId({target_params_id!r})): type(None),\n"
        f"{nested_params}        }},\n"
        "        profile_arguments={},\n"
        "        session_arguments={},\n"
        if in_process
        else ""
    )
    (subroutine / "__init__.py").write_text(
        f"""
import os
from dataclasses import dataclass
from functools import cache

from verdog_runtime.declarations import (
    CallContext, CallVisitDefinition,
    EdgeDefinition,
    GraphDefinition, NodeDefinition,
    PortDefinition, SubroutineCall, SubroutineDefinition, Success,
    WorkflowCall,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId


@dataclass(frozen=True, slots=True)
class EmptyState:
    pass


def adapt(input, state, context, /):
    result = context.invoke(input)
    return Success(output=result, state=state)


ENTER = PortDefinition(id=NodeId({enter_id!r}))
EXIT = PortDefinition(id=NodeId("exit"))
FAILURE = PortDefinition(id=NodeId("failure"))
CALL = NodeDefinition(
    id=NodeId("bridge"),
    name="Bridge",
    state_type=EmptyState,
    operation={call}(
        definition_id=GraphId({target_definition_id!r}),
        definition_module={target_module!r},
{call_params}
        project_path={project_path!r},
    ),
)
GRAPH = GraphDefinition(
    id=GraphId("branch_project.main"),
    params_type=type(None),
    enter=ENTER,
    exit=EXIT,
    failure=FAILURE,
    nodes=(CALL,),
    edges=(
        EdgeDefinition(id=EdgeId("in"), source=ENTER.id, target=CALL.id,
                       visit=CallVisitDefinition(implementation=adapt)),
        EdgeDefinition(id=EdgeId("out"), source=CALL.id, target=EXIT.id),
    ),
)
@cache
def definition():
    return SubroutineDefinition(graph=GRAPH)
""".lstrip(),
        encoding="utf-8",
    )
    _environment(project)
    return project


def _parent(
    project_path: str,
    captured: list[tuple[str, str]],
    definition_id: str = "child_project.main",
    params_override: object = _OMITTED,
) -> GraphDefinition[int, tuple[int, int], None, object]:
    def adapt(
        input: int,
        state: EmptyState,
        context: CallContext[None, None, int, tuple[int, int]],
        /,
    ):
        output = (
            context.invoke(input)
            if params_override is _OMITTED
            else context.invoke(input, params=cast(Any, params_override))
        )
        return Success(output=output, state=state)

    enter = PortDefinition(id=NodeId("parent_enter"))
    exit = PortDefinition(id=NodeId("parent_exit"))
    failure = PortDefinition(id=NodeId("parent_failure"))
    call = NodeDefinition[EmptyState, object](
        id=NodeId("call"),
        name="Call",
        state_type=EmptyState,
        operation=WorkflowCall(
            definition_id=GraphId(definition_id),
            definition_module=(
                definition_id.rpartition(".")[0]
                + ".workflows."
                + definition_id.rpartition(".")[2]
            ),
            project_path=project_path,
        ),
    )
    return GraphDefinition(
        id=GraphId("parent.main"),
        params_type=type(None),
        enter=enter,
        exit=exit,
        failure=failure,
        nodes=(call,),
        edges=(
            EdgeDefinition(
                id=EdgeId("parent_in"),
                source=enter.id,
                target=call.id,
                visit=CallVisitDefinition(implementation=adapt),
            ),
            _edge("parent_out", call.id, exit.id),
        ),
    )


def _resumable_child_project(root: Path) -> Path:
    project = _child_project(root)
    source = project / "src/child_project/subroutines/main/__init__.py"
    source.write_text(
        """
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from verdog_runtime.declarations import (
    EdgeDefinition, GraphDefinition, NodeDefinition, PortDefinition, Python,
    SubroutineDefinition, Success, VisitDefinition,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId


@dataclass(frozen=True, slots=True)
class EmptyState:
    pass


def first(input, state, context, /):
    (context.output_dir / "first.txt").write_text("first")
    with Path("first-visits").open("a", encoding="utf-8") as stream:
        stream.write("first\\n")
    return Success(output=input + 1, state=state)


def second(input, state, context, /):
    (context.output_dir / "later.txt").write_text("later")
    with Path("second-visits").open("a", encoding="utf-8") as stream:
        stream.write("second\\n")
    if not Path("allow-second").is_file():
        raise RuntimeError("second node is deliberately blocked")
    return Success(output=input + 1, state=state)


ENTER = PortDefinition(id=NodeId("enter"))
EXIT = PortDefinition(id=NodeId("exit"))
FAILURE = PortDefinition(id=NodeId("failure"))
FIRST = NodeDefinition(id=NodeId("first"), name="First", state_type=EmptyState,
                       operation=Python())
SECOND = NodeDefinition(id=NodeId("second"), name="Second", state_type=EmptyState,
                        operation=Python())
GRAPH = GraphDefinition(
    id=GraphId("child_project.main"), params_type=type(None), enter=ENTER,
    exit=EXIT, failure=FAILURE, nodes=(FIRST, SECOND),
    edges=(
        EdgeDefinition(id=EdgeId("in"), source=ENTER.id, target=FIRST.id,
                       visit=VisitDefinition(implementation=first)),
        EdgeDefinition(id=EdgeId("next"), source=FIRST.id, target=SECOND.id,
                       visit=VisitDefinition(implementation=second)),
        EdgeDefinition(id=EdgeId("out"), source=SECOND.id, target=EXIT.id),
    ),
)


@cache
def definition():
    return SubroutineDefinition(graph=GRAPH)
""".lstrip(),
        encoding="utf-8",
    )
    return project


def _durable_parent(
    project_path: str,
    *,
    in_process: bool = False,
    catch_remote_error: bool = False,
    interrupt_after_return: dict[str, bool] | None = None,
) -> GraphDefinition[int, int, None, object]:
    def call_impl(
        input: int,
        state: EmptyState,
        context: CallContext[None, object, int, tuple[int, int] | int],
        /,
    ) -> Success[int, EmptyState]:
        try:
            child_output = context.invoke(input, params=None)
        except RemoteWorkflowError:
            if not catch_remote_error:
                raise
            return Success(output=-1, state=state)
        if interrupt_after_return is not None and interrupt_after_return["enabled"]:
            raise KeyboardInterrupt("interrupt after isolated child return")
        value = child_output[0] if isinstance(child_output, tuple) else child_output
        return Success(output=value, state=state)

    enter = PortDefinition(id=NodeId("parent_enter"))
    exit_ = PortDefinition(id=NodeId("parent_exit"))
    failure = PortDefinition(id=NodeId("parent_failure"))
    call = NodeDefinition[EmptyState, object](
        id=NodeId("call"),
        name="Call",
        state_type=EmptyState,
        operation=(
            SubroutineCall(
                definition_id=GraphId("child_project.main"),
                definition_module="child_project.subroutines.main",
                project_path=project_path,
                params_types={(project_path, GraphId("child_project.main")): type(None)},
                profile_arguments={},
                session_arguments={},
            )
            if in_process
            else WorkflowCall(
                definition_id=GraphId("child_project.main"),
                definition_module="child_project.workflows.main",
                project_path=project_path,
            )
        ),
    )
    return GraphDefinition(
        id=GraphId("parent.main"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(call,),
        edges=(
            EdgeDefinition(
                id=EdgeId("parent_in"),
                source=enter.id,
                target=call.id,
                visit=CallVisitDefinition(implementation=call_impl),
            ),
            _edge("parent_out", call.id, exit_.id),
        ),
    )


def test_payloads_round_trip_without_a_verdog_type_whitelist() -> None:
    values = (
        None,
        False,
        0,
        10**5000,
        "x\ny",
        float("inf"),
        float("nan"),
        (1, ("x",)),
        [1, "x"],
        {"key": [1, 2]},
        {1, 2},
        frozenset({1, 2}),
        b"bytes",
        bytearray(b"mutable bytes"),
        2 + 3j,
        Path("tasks/blocks"),
        WireInput(
            root=Path("tasks/blocks"),
            mode=WireMode.CAREFUL,
            limits=(5, 3),
        ),
        MutablePayload([1, 2]),
    )
    for value in values:
        decoded = decode_payload(encode_payload(value))
        if type(value) is float:
            assert cast(float, decoded).hex() == value.hex()
        else:
            assert decoded == value

    shared: list[object] = []
    decoded_shared = cast(
        list[list[object]], decode_payload(encode_payload([shared, shared]))
    )
    assert decoded_shared[0] is decoded_shared[1]

    cycle: list[object] = []
    cycle.append(cycle)
    decoded_cycle = cast(list[object], decode_payload(encode_payload(cycle)))
    assert decoded_cycle[0] is decoded_cycle


def test_workflow_process_round_trips_a_shared_composite_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _child_project(tmp_path)
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    module_name = f"_verdog_shared_payload_{abs(hash(tmp_path))}"
    (shared_root / f"{module_name}.py").write_text(
        """
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Mode(StrEnum):
    CAREFUL = "careful"


@dataclass(frozen=True, slots=True)
class Record:
    data: bytes
    root: Path
    mode: Mode


class Token:
    def __init__(self, value):
        self.value = value
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(shared_root), *sys.path])
    shared_module = cast(Any, importlib.import_module(module_name))
    environment = child / ".verdog" / "environments" / "main"
    site = next(
        path
        for path in (
            *environment.glob("lib/python3.*/site-packages"),
            environment / "Lib" / "site-packages",
        )
        if path.is_dir()
    )
    (site / "_verdog_test_shared.pth").write_text(
        str(shared_root.resolve()) + "\n", encoding="utf-8"
    )

    token = shared_module.Token("same object")
    cycle: list[object] = []
    cycle.append(cycle)
    payload: dict[str, object] = {
        "record": shared_module.Record(
            b"binary", Path("tasks/gripper"), shared_module.Mode.CAREFUL
        ),
        "aliases": (token, token),
        "cycle": cycle,
    }
    result = Dispatcher(project_root=tmp_path).run(
        _definition(_parent(child.name, [])),
        cast(Any, payload),
        output_dir=_output(tmp_path, "composite"),
    )
    output = cast(dict[str, object], cast(object, result.output))
    record = cast(Any, output["record"])
    assert type(record) is shared_module.Record
    assert record.data == b"binary"
    assert record.root == Path("tasks/gripper")
    assert record.mode is shared_module.Mode.CAREFUL
    aliases = cast(tuple[object, object], output["aliases"])
    assert type(aliases[0]) is shared_module.Token
    assert aliases[0] is aliases[1]
    decoded_cycle = cast(list[object], output["cycle"])
    assert decoded_cycle[0] is decoded_cycle


@pytest.mark.parametrize("payload", [None, "not base64", "bm90IGEgcGlja2xl"])
def test_payloads_reject_invalid_frames(payload: object) -> None:
    with pytest.raises(ValueError, match="invalid child payload"):
        decode_payload(payload)


def test_binary_payloads_preserve_their_specific_validation_errors() -> None:
    with pytest.raises(ValueError, match="invalid child binary payload"):
        decode_binary_payload("not base64")


def test_child_checkpoint_envelope_is_strict_and_marks_nested_forks(
    tmp_path: Path,
) -> None:
    compatibility = {"format": "1", "source_sha256": "abc"}
    leaf = encode_child_checkpoint(
        ChildCheckpointBundle(
            compatibility=compatibility,
            runtime=b"authored leaf state",
            shards={},
        )
    )
    outer = encode_child_checkpoint(
        ChildCheckpointBundle(
            compatibility=compatibility,
            runtime=b"authored outer state",
            shards={"children/leaf.pkl": leaf, "journal.bin": b"unchanged"},
        )
    )
    marked = decode_child_checkpoint(
        mark_child_checkpoint_fork(
            outer,
            run_id="forked-run",
            source_output=tmp_path / "source",
            target_output=tmp_path / "target",
            sessions="fresh",
        )
    )

    assert marked.runtime == b"authored outer state"
    assert marked.shards["journal.bin"] == b"unchanged"
    assert [item.run_id for item in marked.pending_forks] == ["forked-run"]
    nested = decode_child_checkpoint(marked.shards["children/leaf.pkl"])
    assert nested.runtime == b"authored leaf state"
    assert [item.run_id for item in nested.pending_forks] == ["forked-run"]

    malformed = json.loads(outer)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="unsupported child checkpoint bundle"):
        decode_child_checkpoint(json.dumps(malformed).encode("utf-8"))

    malformed = json.loads(outer)
    malformed["shards"][0]["name"] = "../escape"
    with pytest.raises(ValueError, match="shard name is invalid"):
        decode_child_checkpoint(json.dumps(malformed).encode("utf-8"))

    malformed = json.loads(outer)
    malformed["runtime"] = "not base64"
    with pytest.raises(ValueError, match="checkpoint runtime is not encoded bytes"):
        decode_child_checkpoint(json.dumps(malformed).encode("utf-8"))


def test_request_distinguishes_omitted_params_from_explicit_none(
    tmp_path: Path,
) -> None:
    arguments = (
        GraphId("child.main"),
        "child.workflows.main",
        4,
        RunId("run"),
        5,
        tmp_path,
        Path("call"),
        ".",
    )
    omitted = cast(dict[str, object], json.loads(_request(*arguments)))
    explicit = cast(dict[str, object], json.loads(_request(*arguments, None)))

    assert omitted["version"] == 13
    assert "params_override" not in omitted
    assert decode_payload(explicit["params_override"]) is None
    omitted_frame = decode_call(omitted)
    explicit_frame = decode_call(explicit)
    assert isinstance(omitted_frame, CallFrame)
    assert omitted_frame.params_override is None
    assert explicit_frame.params_override is not None
    assert decode_payload(explicit_frame.params_override) is None
    assert json.loads(encode_frame(omitted_frame)) == omitted
    assert json.loads(encode_frame(explicit_frame)) == explicit

    for field in omitted:
        malformed = dict(omitted)
        del malformed[field]
        with pytest.raises(ValueError):
            decode_call(malformed)
    assert decode_call({**omitted, "unexpected": True}) == omitted_frame
    with pytest.raises(ValueError):
        decode_call({**omitted, "params_override": None})


def _reply(frame: dict[str, object], remaining: int = 5, /) -> ReplyFrame:
    return decode_reply(
        json.dumps(frame).encode("utf-8"),
        run_id=RunId("run"),
        transitions_remaining=remaining,
        call_path=Path("call"),
        project_path="external/child",
    )


def test_result_frames_keep_errors_structured_and_encode_successes() -> None:
    error = _reply(
        {
            "version": 13,
            "type": "result",
            "outcome": "error",
            "error": {
                "exception_type": "builtins.ValueError",
                "message": "bad",
                "traceback": "ValueError: bad\n",
            },
            "transitions_remaining": 4,
        },
        5,
    )
    assert isinstance(error, ErrorFrame)
    assert error.error == ErrorDetail(
        exception_type="builtins.ValueError",
        message="bad",
        traceback="ValueError: bad\n",
    )
    assert error.transitions_remaining == 4

    success = _reply(
        {
            "version": 13,
            "type": "result",
            "outcome": "success",
            "output": encode_payload(7),
            "transitions_remaining": 3,
        },
        4,
    )
    assert isinstance(success, SuccessFrame)
    assert success.output == 7
    assert success.transitions_remaining == 3


def test_checkpoint_frames_keep_child_continuations_opaque_and_scoped() -> None:
    frame = _reply(
        {
            "version": 13,
            "type": "checkpoint",
            "run_id": "run",
            "transitions_remaining": 4,
            "kind": "node",
            "completed": {
                "project_path": "external/child",
                "graph": "child_project.main",
                "node": "first",
                "visit": 1,
                "call_path": "call/graph-child_project.main",
            },
            "next": {
                "project_path": "external/child",
                "graph": "child_project.main",
                "node": "second",
                "visit": 1,
                "call_path": "call/graph-child_project.main",
            },
            "restore_available": True,
            "session_branch_available": True,
            "continuation": encode_binary_payload(b"opaque child state"),
            "artifact_references": {"schema_version": 1, "kind": "references", "directories": [], "files": []},
            "unavailable_code": None,
            "unavailable_reason": None,
        }
    )
    assert isinstance(frame, CheckpointFrame)
    assert frame.completed == Boundary(
        project_path="external/child",
        graph="child_project.main",
        node="first",
        visit=1,
        call_path="call/graph-child_project.main",
    )
    assert frame.continuation is not None
    assert decode_binary_payload(frame.continuation) == b"opaque child state"
    assert frame.artifact_references is not None

    malformed = json.loads(encode_frame(frame))
    malformed["next"]["call_path"] = "outside"
    with pytest.raises(ValueError, match="escapes its call"):
        _reply(malformed)

    malformed = json.loads(encode_frame(frame))
    malformed["artifact_references"] = None
    with pytest.raises(ValueError, match="no artifact references"):
        _reply(malformed)

    malformed = json.loads(encode_frame(frame))
    malformed["artifact_references"]["directories"] = [
        {"path": "call/.verdog/workspace", "mode": 0o700}
    ]
    with pytest.raises(ValueError, match="unsafe checkpoint artifact"):
        _reply(malformed)


def test_previous_timing_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="child protocol version is invalid"):
        _reply({"version": 10, "type": "result"})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"type": "unknown"}, "child frame type is invalid"),
        ({"type": "timing"}, "invalid child timing frame"),
        (
            {"transitions_remaining": 6, "outcome": "unknown"},
            "invalid transition budget",
        ),
        ({"outcome": "unknown", "extra": True}, "child result has no outcome"),
        ({"output": None, "extra": True}, "invalid child result"),
        ({"output": None}, "invalid child payload"),
    ],
)
def test_reply_dispatch_preserves_validation_order(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _reply(
            {
                "version": 13,
                "type": "result",
                "outcome": "success",
                "output": encode_payload(7),
                "transitions_remaining": 4,
                **changes,
            }
        )


def test_protocol_reader_stops_at_the_result_without_waiting_for_pipe_eof() -> None:
    events: list[EventFrame] = []
    checkpoints: list[CheckpointFrame] = []
    timings: list[TimingRecord] = []
    received: list[str] = []

    def record_event(frame: EventFrame) -> None:
        events.append(frame)
        received.append("event")

    def record_checkpoint(frame: CheckpointFrame) -> None:
        checkpoints.append(frame)
        received.append("checkpoint")

    def record_timing(record: TimingRecord) -> None:
        timings.append(record)
        received.append("timing")

    frames: list[dict[str, object]] = [
        {
            "version": 13,
            "type": "event",
            "kind": "node",
            "run_id": "run",
            "graph_id": "child_project.main",
            "entity_id": "work",
            "status": "succeeded",
            "project_path": "external/child",
            "transitions_remaining": 4,
        },
        {
            "version": 13,
            "type": "checkpoint",
            "run_id": "run",
            "transitions_remaining": 4,
            "kind": "node",
            "completed": None,
            "next": None,
            "restore_available": True,
            "session_branch_available": True,
            "continuation": encode_binary_payload(b"snapshot"),
            "artifact_references": {"schema_version": 1, "kind": "references", "directories": [], "files": []},
            "unavailable_code": None,
            "unavailable_reason": None,
        },
        {"version": 13, "type": "timing", "record": asdict(_timing_record())},
        {
            "version": 13,
            "type": "result",
            "outcome": "success",
            "output": encode_payload("x" * 70_000),
            "transitions_remaining": 3,
        },
    ]
    wire = "".join(json.dumps(frame) + "\n" for frame in frames).encode("utf-8")
    read_descriptor, write_descriptor = os.pipe()
    stream = os.fdopen(read_descriptor, "rb", buffering=0)
    writer = os.fdopen(write_descriptor, "wb", buffering=0)
    release_writer = threading.Event()

    def write_response() -> None:
        try:
            writer.write(wire[:17])
            writer.write(wire[17:])
            writer.flush()
            release_writer.wait(timeout=2)
        finally:
            writer.close()

    writing = threading.Thread(target=write_response, daemon=True)
    writing.start()
    try:
        started = time.monotonic()
        terminal, error = _read_response(
            stream,
            run_id=RunId("run"),
            transitions_remaining=5,
            call_path=Path("call"),
            project_path="external/child",
            event_handler=record_event,
            checkpoint_handler=record_checkpoint,
            timing_handler=record_timing,
            deadline=time.monotonic() + 1,
        )
        assert time.monotonic() - started < 1
    finally:
        release_writer.set()
        stream.close()
        writing.join(timeout=2)

    assert terminal == SuccessFrame(output="x" * 70_000, transitions_remaining=3)
    assert error is None
    assert len(events) == 1 and events[0].transitions_remaining == 4
    assert len(checkpoints) == 1
    assert checkpoints[0].continuation is not None
    assert decode_binary_payload(checkpoints[0].continuation) == b"snapshot"
    assert timings == [_timing_record()]
    assert received == ["event", "checkpoint", "timing"]


def _timing_record() -> TimingRecord:
    return TimingRecord(
        path="call/graph-child_project.main/work/000001",
        project_path="external/child",
        graph_id="child_project.main",
        node_id="work",
        node_type="python",
        status="succeeded",
        duration_seconds=1.0,
    )


def test_typed_reply_frames_round_trip_the_v13_wire_shape() -> None:
    event = EventFrame(
        kind="node",
        run_id=RunId("run"),
        graph_id=GraphId("child_project.main"),
        entity_id="work",
        status="succeeded",
        project_path="external/child",
        transitions_remaining=4,
    )
    timing = TimingFrame(record=_timing_record())
    success = SuccessFrame(output=encode_payload(7), transitions_remaining=3)
    error = ErrorFrame(
        error=ErrorDetail(
            exception_type="builtins.ValueError",
            message="bad",
            traceback="ValueError: bad\n",
        ),
        transitions_remaining=2,
    )
    cases: tuple[tuple[OutgoingFrame, dict[str, object], ReplyFrame], ...] = (
        (
            event,
            {
                "version": 13,
                "type": "event",
                "kind": "node",
                "run_id": "run",
                "graph_id": "child_project.main",
                "entity_id": "work",
                "status": "succeeded",
                "project_path": "external/child",
                "transitions_remaining": 4,
            },
            event,
        ),
        (
            timing,
            {"version": 13, "type": "timing", "record": asdict(_timing_record())},
            timing,
        ),
        (
            success,
            {
                "version": 13,
                "type": "result",
                "outcome": "success",
                "output": success.output,
                "transitions_remaining": 3,
            },
            SuccessFrame(output=7, transitions_remaining=3),
        ),
        (
            error,
            {
                "version": 13,
                "type": "result",
                "outcome": "error",
                "error": {
                    "exception_type": "builtins.ValueError",
                    "message": "bad",
                    "traceback": "ValueError: bad\n",
                },
                "transitions_remaining": 2,
            },
            error,
        ),
    )
    for outgoing, wire, expected in cases:
        raw = encode_frame(outgoing)
        assert raw.endswith(b"\n")
        assert json.loads(raw) == wire
        assert (
            decode_reply(
                raw,
                run_id=RunId("run"),
                transitions_remaining=5,
                call_path=Path("call"),
                project_path="external/child",
            )
            == expected
        )
        malformed_frames = [{**wire, "unexpected": True}]
        for field in wire:
            malformed = dict(wire)
            del malformed[field]
            malformed_frames.append(malformed)
        for malformed in malformed_frames:
            with pytest.raises(ValueError):
                decode_reply(
                    json.dumps(malformed).encode("utf-8"),
                    run_id=RunId("run"),
                    transitions_remaining=5,
                    call_path=Path("call"),
                    project_path="external/child",
                )


@pytest.mark.parametrize(
    ("path", "project_path"),
    [
        ("call/graph-child_project.main/work/000001", "external/child"),
        (
            "call/nested/graph-child_project.main/work/000001",
            "external/child/external/grandchild",
        ),
    ],
)
def test_timing_frames_accept_only_the_child_subtree(
    path: str, project_path: str
) -> None:
    record = replace(_timing_record(), path=path, project_path=project_path)
    frame: dict[str, object] = {
        "version": 13,
        "type": "timing",
        "record": asdict(record),
    }
    assert _reply(frame) == TimingFrame(record=record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "caller/graph-child_project.main/work/000001"),
        ("path", "call/../other"),
        ("path", "/call/work"),
        ("project_path", "external/childish"),
        ("project_path", "external/child/../sibling"),
        ("status", "unknown"),
        ("status", "running"),
        ("status", "interrupted"),
        ("status", []),
        ("kind", "edge"),
        ("graph_id", ""),
        ("node_id", ""),
        ("node_type", ""),
        ("node_type", 3),
        ("duration_seconds", True),
        ("duration_seconds", None),
        ("duration_seconds", float("nan")),
        ("duration_seconds", -1),
        ("duration_seconds", float("inf")),
        ("duration_seconds", 10**400),
    ],
)
def test_timing_frames_reject_invalid_or_out_of_scope_records(
    field: str, value: object
) -> None:
    record = asdict(_timing_record())
    record[field] = value
    with pytest.raises(ValueError):
        _reply(
            {"version": 13, "type": "timing", "record": record},
        )


def test_timing_frames_cannot_change_the_transition_budget() -> None:
    with pytest.raises(ValueError, match="invalid child timing frame"):
        _reply(
            {
                "version": 13,
                "type": "timing",
                "record": asdict(_timing_record()),
                "transitions_remaining": 100,
            },
        )


@pytest.mark.parametrize("record", [None, [], {}])
def test_timing_frames_require_a_complete_record(record: object) -> None:
    with pytest.raises(ValueError, match="invalid child timing record"):
        _reply(
            {"version": 13, "type": "timing", "record": record},
        )


def test_timing_identity_strings_are_opaque_like_execution_events() -> None:
    record = replace(_timing_record(), graph_id="../child", node_id="../work")
    assert _reply(
        {"version": 13, "type": "timing", "record": asdict(record)},
    ) == TimingFrame(record=record)


@pytest.mark.parametrize(
    "started_at", [None, True, "1.0", -1, float("nan"), float("inf"), 10**400]
)
def test_child_rejects_invalid_shared_clock(tmp_path: Path, started_at: object) -> None:
    request = cast(
        dict[str, object],
        json.loads(
            _request(
                GraphId("child_project.main"),
                "child_project.workflows.main",
                4,
                RunId("run"),
                4,
                tmp_path,
                Path("call"),
                ".",
            )
        ),
    )
    request["started_at"] = started_at
    response = _serve(request, tmp_path, lambda _frame: None)
    assert isinstance(response, ErrorFrame)
    assert "invalid child time" in response.error.message


def test_nested_processes_stream_timings_on_the_root_clock(tmp_path: Path) -> None:
    branch = _bridge_project(tmp_path, "external/child", "child_project.main")
    child = _child_project(branch / "external")
    output = _output(tmp_path, "timing")
    (output / "call").mkdir(parents=True)
    (output / "trace").touch()
    timings: list[TimingRecord] = []
    budgets: list[int] = []
    started_at = time.monotonic() - 20
    value, remaining = invoke_child(
        project_root=tmp_path,
        project_path=branch.name,
        definition_id=GraphId("branch_project.main"),
        definition_module="branch_project.workflows.main",
        input=4,
        run_id=RunId("timing-run"),
        transitions_remaining=10,
        output_dir=output,
        call_path=Path("call"),
        owner_project_path=".",
        started_at=started_at,
        timing_handler=timings.append,
        event_handler=lambda event: budgets.append(event.transitions_remaining),
    )
    assert cast(tuple[int, int], value)[0] == 5
    assert remaining == 6  # Two transitions per child graph; timing costs none.
    assert budgets == sorted(budgets, reverse=True)
    assert timings
    assert {record.project_path for record in timings} == {
        branch.name,
        f"{branch.name}/external/{child.name}",
    }
    for record in timings:
        assert Path(record.path).is_relative_to("call")
        assert record.status == "succeeded"
        assert record.duration_seconds >= 0
    assert len(timings) == len({record.path for record in timings})
    trace = (output / "trace").read_text("utf-8").splitlines()
    starts = [line.split(" START ", 1)[1] for line in trace if " START " in line]
    ends = [
        line.split(" END ", 1)[1].split(" status=", 1)[0]
        for line in trace
        if " END " in line
    ]
    assert sorted(starts) == sorted(ends) == sorted(record.path for record in timings)
    for line in trace:
        elapsed = float(line.split(" +", 1)[1].split("s]", 1)[0])
        assert elapsed >= 20
    assert {path.name for path in output.iterdir()} == {"call", "trace"}


def test_output_ids_are_portable_path_components() -> None:
    assert _encoded_id("../NUL.") == "..%2FNUL%2E"
    assert _encoded_id("NUL") == "%5FNUL"


@pytest.mark.parametrize("call_path", ["../outside", "missing"])
def test_child_call_path_must_be_an_existing_output_directory(
    tmp_path: Path,
    call_path: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "trace").touch()
    outside = tmp_path / "outside"
    outside.mkdir()
    response = _serve(
        {
            "version": 13,
            "type": "call",
            "definition_id": "child_project.main",
            "definition_module": "child_project.workflows.main",
            "input": encode_payload(None),
            "run_id": "run",
            "transitions_remaining": 4,
            "output_dir": str(output.resolve()),
            "call_path": call_path,
            "project_path": ".",
            "started_at": time.monotonic(),
            "checkpointing": "off",
            "retry_incomplete": False,
        },
        tmp_path,
        lambda _frame: None,
    )
    assert isinstance(response, ErrorFrame)
    assert "call path" in response.error.message
    assert not tuple(outside.iterdir())


def test_child_transport_error_visits_its_failure_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _child_project(tmp_path)
    output = tmp_path / "output"
    call = output / "call"
    call.mkdir(parents=True)
    (output / "trace").touch()
    events: list[EventFrame | TimingFrame | CheckpointFrame] = []

    encoded_input = encode_payload(4)

    def reject_transport(_value: object, /) -> NoReturn:
        raise TypeError("test output is not transportable")

    monkeypatch.setattr("verdog_runtime.child.encode_payload", reject_transport)
    response = _serve(
        {
            "version": 13,
            "type": "call",
            "definition_id": "child_project.main",
            "definition_module": "child_project.workflows.main",
            "input": encoded_input,
            "run_id": "run",
            "transitions_remaining": 4,
            "output_dir": str(output.resolve()),
            "call_path": "call",
            "project_path": child.name,
            "started_at": time.monotonic(),
            "checkpointing": "off",
            "retry_incomplete": False,
        },
        child,
        events.append,
    )

    assert isinstance(response, ErrorFrame)
    assert response.error.exception_type == "builtins.TypeError"
    assert "test output is not transportable" in response.error.message
    assert any(
        isinstance(event, EventFrame)
        and event.entity_id == "failure"
        and event.status == "failed"
        for event in events
    ), events
    stack = (call / "failure/000001/stacktrace.txt").read_text(
        "utf-8"
    )
    assert "TypeError: test output is not transportable" in stack
    assert "Verdog check: entity=exit" in stack
    assert "Verdog failure boundary:" in stack
    timings = [event.record for event in events if isinstance(event, TimingFrame)]
    exit_timings = [record for record in timings if record.node_id == "exit"]
    assert [record.status for record in exit_timings] == ["failed"]
    assert exit_timings[0].duration_seconds >= 0
    exit_ends = [
        line.split(" END ", 1)[1]
        for line in (output / "trace").read_text("utf-8").splitlines()
        if " END call/exit/000001 " in line
    ]
    assert len(exit_ends) == 1
    assert "status=failed" in exit_ends[0]


def test_child_success_encodes_its_output_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _child_project(tmp_path)
    (child / "project.json").unlink()
    output = tmp_path / "output"
    call = output / "call"
    call.mkdir(parents=True)
    (output / "trace").touch()
    encoded_input = encode_payload(4)
    encode = encode_payload
    encoded: list[object] = []

    def count(value: object, /) -> str:
        encoded.append(value)
        return encode(value)

    monkeypatch.setattr("verdog_runtime.child.encode_payload", count)
    response = _serve(
        {
            "version": 13,
            "type": "call",
            "definition_id": "child_project.main",
            "definition_module": "child_project.workflows.main",
            "input": encoded_input,
            "run_id": "run",
            "transitions_remaining": 4,
            "output_dir": str(output.resolve()),
            "call_path": "call",
            "project_path": child.name,
            "started_at": time.monotonic(),
            "checkpointing": "off",
            "retry_incomplete": False,
        },
        child,
        lambda _frame: None,
    )

    assert isinstance(response, SuccessFrame)
    assert len(encoded) == 1
    assert cast(tuple[int, int], decode_payload(response.output))[0] == 5


def test_project_dataclasses_cross_both_process_directions_by_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_output_module = "child_project.payload"
    child = _child_project(tmp_path, child_output_module=child_output_module)
    (child / "src/child_project/payload.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True, slots=True)\n"
        "class ChildOnly:\n"
        "    value: int = 23\n",
        encoding="utf-8",
    )
    parent = _definition(_parent(child.name, []))
    package_name = f"parent_project_{abs(hash(tmp_path))}"
    package = tmp_path / "src" / package_name
    package.mkdir(parents=True)
    (tmp_path / "project.json").write_text(
        json.dumps(
            {
                "schema_version": 32,
                "package": package_name,
                "workflow": {
                    "subroutine": "main",
                    "profile_arguments": {},
                    "session_arguments": {},
                },
                "subroutine": {
                    "id": "main",
                    "subroutines": [],
                    "workflows": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (package / "__init__.py").touch()
    (package / "payload.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True, slots=True)\n"
        "class ParentOnly:\n"
        "    value: int\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path / "src"), *sys.path])
    parent_module = cast(Any, importlib.import_module(f"{package_name}.payload"))
    parent_value = parent_module.ParentOnly(17)

    echoed = (
        Dispatcher(project_root=tmp_path)
        .run(
            parent,
            parent_value,
            output_dir=_output(tmp_path, "parent-value"),
        )
        .output
    )
    assert echoed == parent_value
    assert type(echoed) is type(parent_value)

    child_value = cast(
        Any,
        Dispatcher(project_root=tmp_path)
        .run(
            parent,
            -8,
            output_dir=_output(tmp_path, "child-value"),
        )
        .output,
    )
    assert type(child_value).__module__ == child_output_module
    assert type(child_value).__name__ == "ChildOnly"
    assert child_value.value == 23


class _InterruptAfterFirstChildCheckpoint(Dispatcher):
    @override
    def _accept_remote_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        super()._accept_remote_checkpoint(*args, **kwargs)
        frame = cast(CheckpointFrame, args[1])
        if frame.completed is not None and frame.completed.node == "first":
            raise KeyboardInterrupt("interrupt after first child checkpoint")


def test_remote_checkpoint_references_are_captured_before_child_continues(
    tmp_path: Path,
) -> None:
    child = _resumable_child_project(tmp_path)
    definition = _definition(_durable_parent(child.name))
    output = _output(tmp_path, "remote-boundary")

    class DelayedCheckpoint(Dispatcher):
        @override
        def _accept_remote_checkpoint(self, *args: Any, **kwargs: Any) -> None:
            frame = cast(CheckpointFrame, args[1])
            first = frame.completed is not None and frame.completed.node == "first"
            if first:
                deadline = time.monotonic() + 10
                while not list(output.rglob("later.txt")):
                    if time.monotonic() >= deadline:
                        raise AssertionError("child did not continue before checkpoint commit")
                    time.sleep(0.01)
            super()._accept_remote_checkpoint(*args, **kwargs)
            if first:
                raise KeyboardInterrupt("stop at delayed checkpoint")

    with pytest.raises(KeyboardInterrupt, match="delayed checkpoint"):
        DelayedCheckpoint(project_root=tmp_path).run(
            definition,
            0,
            output_dir=output,
            checkpointing=CheckpointPolicy.REQUIRED,
        )

    assert list(output.rglob("later.txt"))
    store = RunStore.open(output)
    checkpoint = next(
        item for item in store.checkpoints()
        if item.completed is not None and item.completed.node == "first"
    )
    directory = store.checkpoint_directory(checkpoint.sequence)
    manifest = json.loads((directory / "manifest.json").read_text())
    paths = [record["path"] for record in manifest["artifacts"]["files"]]
    assert any(path.endswith("/first.txt") for path in paths)
    assert not any(path.endswith("/later.txt") for path in paths)
    assert not (directory / "artifacts").exists()


@pytest.mark.parametrize("legacy_layout", [False, True])
def test_durable_workflow_checkpoint_resumes_inside_the_isolated_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_layout: bool,
) -> None:
    child = _resumable_child_project(tmp_path)
    definition = _definition(_durable_parent(child.name))
    output = _output(tmp_path, "remote-resume")
    legacy_marker = child / "legacy-output-layout"
    if legacy_layout:
        source = child / "src/child_project/subroutines/main/__init__.py"
        source.write_text(
            "from pathlib import Path as _LegacyPath\n"
            "if _LegacyPath('legacy-output-layout').exists():\n"
            "    import sys as _legacy_sys\n"
            f"    _legacy_sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
            "    from layout_helpers import legacy_graph_create as _legacy_create\n"
            "    from verdog_runtime.interpreter.execution import _GraphOutput\n"
            "    _GraphOutput.create = classmethod(_legacy_create)\n"
            + source.read_text("utf-8"),
            encoding="utf-8",
        )
        legacy_marker.touch()

    with monkeypatch.context() as legacy:
        if legacy_layout:
            original_register = Dispatcher._register_workflow_activation  # pyright: ignore[reportPrivateUsage]

            def legacy_register(dispatcher: Dispatcher, parent: Any, call: Any, /) -> None:
                visit = Path(call.visit_path)
                path = visit.with_name(f"{visit.name}-child-legacy")
                (parent.graph_output.root / path).mkdir()
                call.child_call_path = path.as_posix()
                original_register(dispatcher, parent, call)

            legacy.setattr(
                "verdog_runtime.interpreter.execution._GraphOutput.create",
                classmethod(legacy_graph_create),
            )
            legacy.setattr(Dispatcher, "_register_workflow_activation", legacy_register)
        with pytest.raises(KeyboardInterrupt, match="first child checkpoint"):
            _InterruptAfterFirstChildCheckpoint(project_root=tmp_path).run(
                definition,
                4,
                output_dir=output,
                checkpointing=CheckpointPolicy.AUTO,
            )
    legacy_marker.unlink(missing_ok=True)

    store = RunStore.open(output)
    manifest = store.manifest()
    assert manifest.status is RunStatus.INTERRUPTED
    sequence = manifest.checkpoints.latest_completed
    assert sequence is not None
    assert sequence == manifest.checkpoints.latest_restorable
    checkpoint_manifest = json.loads(
        (store.checkpoint_directory(sequence) / "manifest.json").read_text("utf-8")
    )
    shard_names = {item["name"] for item in checkpoint_manifest["shards"]}
    assert "runtime.pkl" in shard_names
    assert any(name.startswith("children/") for name in shard_names)
    assert (child / "first-visits").read_text("utf-8").splitlines() == ["first"]
    # The isolated process may execute ahead before parent-side cancellation,
    # but the committed child continuation still precedes the second node.
    assert (child / "second-visits").read_text("utf-8").splitlines() == ["second"]

    call_directory = (
        output / "graph-parent.main/call/000001" if legacy_layout else output / "call/000001"
    )
    child_output = (
        call_directory.with_name("000001-child-legacy")
        if legacy_layout
        else call_directory
    )
    assert _only_call_output(output) == child_output
    assert not list(call_directory.glob("attempt-*"))

    (child / "allow-second").touch()
    resumed = Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)

    assert resumed.output == 6
    assert (child / "first-visits").read_text("utf-8").splitlines() == ["first"]
    assert (child / "second-visits").read_text("utf-8").splitlines() == [
        "second",
        "second",
    ]
    assert _only_call_output(output) == child_output
    assert not list(call_directory.glob("attempt-*"))
    child_graph = child_output / "graph-child_project.main" if legacy_layout else child_output
    assert [path.name for path in (child_graph / "enter").iterdir()] == ["000001"]
    assert not (child_graph / "first/000002").exists()
    assert (child_graph / "second/000002").is_dir()
    assert RunStore.open(output).manifest().status is RunStatus.SUCCEEDED


@pytest.mark.parametrize("in_process", [False, True])
def test_repeated_call_visits_each_contain_their_own_child_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_process: bool,
) -> None:
    for name in tuple(sys.modules):
        if name == "child_project" or name.startswith("child_project."):
            monkeypatch.delitem(sys.modules, name)
    child = _child_project(tmp_path)
    monkeypatch.setattr(sys, "path", [str(child / "src"), *sys.path])
    parent = _durable_parent(child.name, in_process=in_process)
    call = parent.nodes[0]
    assert isinstance(call, NodeDefinition)
    repeated = replace(
        parent,
        nodes=(call,),
        edges=(
            parent.edges[0],
            replace(parent.edges[1], target=call.id, visit=parent.edges[0].visit),
        ),
    )
    output = _output(tmp_path, "repeated-calls")
    # Each round consumes one caller transition and two child transitions.
    # This bound stops the loop after exactly two completed child invocations.
    with pytest.raises(RuntimeError, match="workflow transition limit exceeded"):
        Dispatcher(project_root=tmp_path, transition_limit=6).run(
            _definition(repeated), 4, output_dir=output
        )

    call_directory = output / "call"
    visits = sorted(call_directory.iterdir())
    assert [visit.name for visit in visits] == ["000001", "000002"]
    for visit in visits:
        nodes = {entry.name for entry in visit.iterdir() if entry.is_dir()}
        assert nodes == {"enter", "work", "exit"}
        for node in nodes:
            assert [entry.name for entry in (visit / node).iterdir()] == ["000001"]
    for report in ("config.md", "stats.md"):
        assert call_reports(output / report) == [visit / report for visit in visits]
    assert not (output / "activations").exists()


def test_durable_workflow_call_can_catch_remote_child_error(tmp_path: Path) -> None:
    child = _resumable_child_project(tmp_path)
    definition = _definition(_durable_parent(child.name, catch_remote_error=True))
    output = _output(tmp_path, "remote-error-caught")

    result = Dispatcher(project_root=tmp_path).run(
        definition,
        4,
        output_dir=output,
        checkpointing=CheckpointPolicy.AUTO,
    )

    assert result.output == -1
    assert (child / "first-visits").read_text("utf-8").splitlines() == ["first"]
    assert (child / "second-visits").read_text("utf-8").splitlines() == ["second"]
    child_failure = (
        _only_call_output(output) / "failure/000001"
    )
    assert (child_failure / "stacktrace.txt").is_file()


def test_resume_after_isolated_child_return_does_not_restart_child(
    tmp_path: Path,
) -> None:
    child = _resumable_child_project(tmp_path)
    (child / "allow-second").touch()
    control = {"enabled": True}
    definition = _definition(
        _durable_parent(child.name, interrupt_after_return=control)
    )
    output = _output(tmp_path, "remote-return-interrupted")

    with pytest.raises(KeyboardInterrupt, match="isolated child return"):
        Dispatcher(project_root=tmp_path).run(
            definition,
            4,
            output_dir=output,
            checkpointing=CheckpointPolicy.REQUIRED,
        )
    assert RunStore.open(output).manifest().status is RunStatus.INTERRUPTED
    assert (child / "first-visits").read_text("utf-8").splitlines() == ["first"]
    assert (child / "second-visits").read_text("utf-8").splitlines() == ["second"]

    control["enabled"] = False
    result = Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)

    assert result.output == 6
    assert (child / "first-visits").read_text("utf-8").splitlines() == ["first"]
    assert (child / "second-visits").read_text("utf-8").splitlines() == ["second"]


def test_durable_workflow_fork_defers_child_state_transform_and_keeps_base(
    tmp_path: Path,
) -> None:
    child = _resumable_child_project(tmp_path)
    definition = _definition(_durable_parent(child.name))
    source_output = _output(tmp_path, "remote-fork-source")
    target_output = _output(tmp_path, "remote-fork-target")

    with pytest.raises(KeyboardInterrupt, match="first child checkpoint"):
        _InterruptAfterFirstChildCheckpoint(project_root=tmp_path).run(
            definition,
            4,
            output_dir=source_output,
            checkpointing=CheckpointPolicy.AUTO,
        )
    source = RunStore.open(source_output).manifest()
    checkpoint = source.checkpoints.latest_restorable
    assert checkpoint is not None

    # The child snapshot contains child_project.EmptyState, which is not
    # importable in this parent interpreter.  Forking must leave those bytes
    # opaque and ask the child environment to transform them.
    (child / "allow-second").touch()
    forked = Dispatcher(project_root=tmp_path).fork(
        definition,
        source_output_dir=source_output,
        checkpoint=checkpoint,
        output_dir=target_output,
        sessions=SessionPolicy.FRESH,
    )

    target = RunStore.open(target_output).manifest()
    assert forked.output == 6
    assert target.status is RunStatus.SUCCEEDED
    assert RunStore.open(source_output).manifest().status is RunStatus.INTERRUPTED
    assert (child / "first-visits").read_text("utf-8").splitlines() == ["first"]
    assert (child / "second-visits").read_text("utf-8").splitlines() == [
        "second",
        "second",
    ]
    child_graph = _only_call_output(target_output)
    assert not (child_graph / "first/000002").exists()
    assert (child_graph / "second/000001").is_dir()


def test_child_source_drift_is_rejected_before_runtime_decode(tmp_path: Path) -> None:
    child = _resumable_child_project(tmp_path)
    definition = _definition(_durable_parent(child.name))
    output = _output(tmp_path, "remote-drift")

    with pytest.raises(KeyboardInterrupt, match="first child checkpoint"):
        _InterruptAfterFirstChildCheckpoint(project_root=tmp_path).run(
            definition,
            4,
            output_dir=output,
            checkpointing=CheckpointPolicy.AUTO,
        )
    store = RunStore.open(output)
    manifest = store.manifest()
    sequence = manifest.checkpoints.latest_restorable
    assert sequence is not None
    checkpoint_manifest = json.loads(
        (store.checkpoint_directory(sequence) / "manifest.json").read_text("utf-8")
    )
    child_shard = next(
        item["name"]
        for item in checkpoint_manifest["shards"]
        if item["name"].startswith("children/")
    )
    bundle = decode_child_checkpoint(store.checkpoint_shard(sequence, child_shard))
    deliberately_undecodable = encode_child_checkpoint(
        ChildCheckpointBundle(
            compatibility=bundle.compatibility,
            runtime=b"this is not a serialized continuation",
            shards=bundle.shards,
            pending_forks=bundle.pending_forks,
        )
    )

    source = child / "src/child_project/subroutines/main/__init__.py"
    source.write_text(source.read_text("utf-8") + "\n# incompatible source drift\n")
    request = json.loads(
        _request(
            GraphId("child_project.main"),
            "child_project.workflows.main",
            4,
            RunId(manifest.id),
            9_997,
            output,
            Path("call/000001"),
            child.name,
            None,
            checkpointing=CheckpointPolicy.AUTO,
            resume_continuation=deliberately_undecodable,
        )
    )
    response = _serve(request, child, lambda _frame: None)

    assert isinstance(response, ErrorFrame)
    assert "compatibility fingerprints do not match" in response.error.message
    assert "changed source_sha256" in response.error.message
    assert "serialized continuation" not in response.error.message


def test_workflow_runs_in_a_child_process_and_returns_its_budget(
    tmp_path: Path,
) -> None:
    child = _child_project(tmp_path)
    captured: list[tuple[str, str]] = []
    parent = _parent(child.name, captured)
    events: list[NodeExecution | EdgeExecution] = []

    result = Dispatcher(project_root=tmp_path, execution_handler=events.append).run(
        _definition(parent), 4, output_dir=_output(tmp_path, "normal")
    )
    assert isinstance(result, Success)
    assert result.output[0] == 5
    assert result.output[1] != os.getpid()
    child_events = [
        event for event in events if event.graph_id == GraphId("child_project.main")
    ]
    assert child_events
    assert all(event.remote and event.state is None for event in child_events)
    assert {event.project_path for event in child_events} == {child.name}
    assert any(
        isinstance(event, NodeExecution) and event.node_id == NodeId("work")
        for event in child_events
    )
    output = _output(tmp_path, "normal")
    child_output = _only_call_output(output)
    child_graph = child_output
    child_prefix = child_graph.relative_to(output).as_posix()
    starts = [
        line.split(" START ", 1)[1]
        for line in (output / "trace.log").read_text("utf-8").splitlines()
        if " START " in line
    ]
    assert starts == [
        "parent_enter/000001",
        "call/000001",
        f"{child_prefix}/enter/000001",
        f"{child_prefix}/work/000001",
        f"{child_prefix}/exit/000001",
        "parent_exit/000001",
    ]
    assert (child_graph / "work/000001/visited").read_text("utf-8") == "child"

    # One encoded frame above the transport's 64 KiB read chunk.
    large = 10**80_000
    large_result = Dispatcher(project_root=tmp_path).run(
        _definition(parent), large, output_dir=_output(tmp_path, "large")
    )
    assert isinstance(large_result, Success)
    assert large_result.output[0] == large + 1

    with pytest.raises((RuntimeError, RemoteWorkflowError), match="transition limit"):
        Dispatcher(project_root=tmp_path, transition_limit=3).run(
            _definition(parent), 4, output_dir=_output(tmp_path, "exhausted")
        )

    with pytest.raises(RemoteWorkflowError, match="expected child failure") as caught:
        Dispatcher(project_root=tmp_path).run(
            _definition(parent), -2, output_dir=_output(tmp_path, "failed")
        )
    assert caught.value.exception_type == "builtins.ValueError"
    failed_output = _output(tmp_path, "failed")
    failed_child = _only_call_output(failed_output)
    failed_stack = (
        failed_child / "failure/000001/stacktrace.txt"
    ).read_text("utf-8")
    assert "ValueError: expected child failure" in failed_stack
    assert "graph=child_project.main node=work" in failed_stack
    assert "graph=child_project.main failure=failure" in failed_stack
    for name in ("config.md", "stats.md"):
        assert call_reports(failed_output / name) == [(failed_child / name).resolve()]
        assert call_reports(failed_child / name) == []
    assert any(
        row[2:5] == ["failure", "failure", "1"]
        for row in table_rows(failed_child / "stats.md", "Nodes")
    )

    unchecked = Dispatcher(project_root=tmp_path).run(
        _definition(parent), -7, output_dir=_output(tmp_path, "unchecked")
    )
    assert unchecked.output[0] == -6

    isolated = Dispatcher(project_root=tmp_path).run(
        _definition(parent), -3, output_dir=_output(tmp_path, "isolated")
    )
    assert isinstance(isolated, Success)
    assert isolated.output[0] == -2

    before_crash = len(events)
    with pytest.raises(RuntimeError, match="child_process_failed"):
        Dispatcher(
            project_root=tmp_path,
            transition_limit=2,
            execution_handler=events.append,
        ).run(_definition(parent), -4, output_dir=_output(tmp_path, "crashed"))
    assert any(
        event.remote and event.project_path == child.name
        for event in events[before_crash:]
    )

    with pytest.raises(RuntimeError, match="child_process_failed"):
        Dispatcher(project_root=tmp_path).run(
            _definition(parent), -9, output_dir=_output(tmp_path, "mishandled")
        )

    cancellation = CancellationToken.with_timeout(5)
    cancelled_at: list[float] = []

    def cancel_started_child() -> None:
        time.sleep(0.2)
        cancelled_at.append(time.monotonic())
        cancellation.cancel()

    canceller = threading.Thread(target=cancel_started_child, daemon=True)
    canceller.start()
    with pytest.raises(ExecutionCancelled):
        Dispatcher(
            project_root=tmp_path,
            execution_handler=events.append,
            cancellation=cancellation,
        ).run(_definition(parent), -1, output_dir=_output(tmp_path, "cancelled"))
    canceller.join(timeout=1)
    assert cancelled_at and time.monotonic() - cancelled_at[0] < 3


def test_workflow_process_owns_defaults_and_accepts_an_immediate_override(
    tmp_path: Path,
) -> None:
    child = _child_project(tmp_path, typed_params=True)
    default_result = Dispatcher(project_root=tmp_path).run(
        _definition(_parent(child.name, [])),
        4,
        output_dir=_output(tmp_path, "default-params"),
    )
    override_result = Dispatcher(project_root=tmp_path).run(
        _definition(_parent(child.name, [], params_override=SimpleNamespace(delta=6))),
        4,
        output_dir=_output(tmp_path, "override-params"),
    )
    none_result = Dispatcher(project_root=tmp_path).run(
        _definition(_parent(child.name, [], params_override=None)),
        4,
        output_dir=_output(tmp_path, "none-params"),
    )
    assert default_result.output[0] == 5
    assert override_result.output[0] == 10
    assert none_result.output[0] == 4
    for name, expected in (
        ("default-params", ["params.delta", "1"]),
        ("override-params", ["params", "namespace(delta=6)"]),
        ("none-params", ["params", "None"]),
    ):
        report = _only_call_output(_output(tmp_path, name)) / "config.md"
        assert table_rows(report) == [expected]


def _process_exists(pid: int) -> bool:
    if os.name == "nt":
        import subprocess

        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - the test owns its process
        return True
    return True


def _wait_until_gone(pid: int) -> None:
    deadline = time.monotonic() + 3
    while _process_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_exists(pid), f"child process {pid} survived cancellation"


def test_blocking_child_transport_honors_a_monotonic_deadline(
    tmp_path: Path,
) -> None:
    child = _child_project(tmp_path)
    output = _output(tmp_path, "transport-timeout")
    (output / "call").mkdir(parents=True)
    (output / "trace").touch()

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="child_process_timeout"):
        invoke_child(
            project_root=tmp_path,
            project_path=child.name,
            definition_id=GraphId("child_project.main"),
            definition_module="child_project.workflows.main",
            input=-1,
            run_id=RunId("timeout-run"),
            transitions_remaining=10,
            output_dir=output,
            call_path=Path("call"),
            owner_project_path=".",
            deadline=time.monotonic() + 0.2,
        )
    assert time.monotonic() - started < 3


def test_blocking_child_transport_keyboard_interrupt_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _child_project(tmp_path)
    output = _output(tmp_path, "transport-cancel")
    (output / "call").mkdir(parents=True)
    (output / "trace").touch()
    pid_file = tmp_path / "transport-child.pid"
    monkeypatch.setenv("VERDOG_TEST_PID_FILE", str(pid_file))
    cancellation = threading.Event()
    cancelled_at: list[float] = []

    def cancel_after_child_starts() -> None:
        deadline = time.monotonic() + 5
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        cancelled_at.append(time.monotonic())
        cancellation.set()

    def check_cancelled() -> None:
        if cancellation.is_set():
            raise KeyboardInterrupt("simulated Ctrl+C")

    watcher = threading.Thread(target=cancel_after_child_starts, daemon=True)
    watcher.start()
    with pytest.raises(KeyboardInterrupt, match=r"simulated Ctrl\+C"):
        invoke_child(
            project_root=tmp_path,
            project_path=child.name,
            definition_id=GraphId("child_project.main"),
            definition_module="child_project.workflows.main",
            input=-5,
            run_id=RunId("cancel-run"),
            transitions_remaining=10,
            output_dir=output,
            call_path=Path("call"),
            owner_project_path=".",
            check_cancelled=check_cancelled,
            deadline=time.monotonic() + 6,
        )
    watcher.join(timeout=1)

    assert pid_file.is_file(), "blocking child did not start"
    assert cancelled_at and time.monotonic() - cancelled_at[0] < 3
    _wait_until_gone(int(pid_file.read_text("utf-8")))


def test_durable_workflow_cancellation_keeps_pending_start_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _child_project(tmp_path)
    pid_file = tmp_path / "durable-child.pid"
    monkeypatch.setenv("VERDOG_TEST_PID_FILE", str(pid_file))
    definition = _definition(_durable_parent(child.name))
    output = _output(tmp_path, "durable-cancelled")

    cancellation = CancellationToken.with_timeout(5)

    def cancel_active_child() -> None:
        while not pid_file.is_file() and not cancellation.expired:
            time.sleep(0.01)
        cancellation.cancel()

    canceller = threading.Thread(target=cancel_active_child, daemon=True)
    canceller.start()
    with pytest.raises(ExecutionCancelled):
        Dispatcher(project_root=tmp_path, cancellation=cancellation).run(
            definition,
            -5,
            output_dir=output,
            checkpointing=CheckpointPolicy.AUTO,
        )
    canceller.join(timeout=1)
    assert pid_file.is_file(), "isolated child did not start"
    child_pid = int(pid_file.read_text("utf-8"))
    _wait_until_gone(child_pid)

    store = RunStore.open(output)
    manifest = store.manifest()
    assert manifest.status is RunStatus.INTERRUPTED
    checkpoint = manifest.checkpoints.latest_restorable
    assert checkpoint is not None
    snapshot = decode_continuation(store.checkpoint_shard(checkpoint, "runtime.pkl"))
    call_frames = [
        frame for frame in snapshot.frames if isinstance(frame, CallFrameSnapshot)
    ]
    assert len(call_frames) == 1
    call = call_frames[0]
    assert call.phase == "child_pending"
    assert call.child_call_path is None
    assert not any(
        name.startswith("children/") for name in store.checkpoint_shards(checkpoint)
    )


def test_nested_process_paths_and_tree_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _bridge_project(tmp_path, "external/branch", "branch_project.main")
    middle = _bridge_project(
        branch / "external", "external/child", "child_project.main"
    )
    child = _child_project(middle / "external")
    assert middle == branch / "external" / "branch"
    assert child == middle / "external" / "child"
    pid_file = tmp_path / "deep.pid"
    monkeypatch.setenv("VERDOG_TEST_PID_FILE", str(pid_file))
    parent = _parent(
        branch.name,
        [],
        "branch_project.main",
    )
    events: list[NodeExecution | EdgeExecution] = []

    complete = Dispatcher(
        project_root=tmp_path,
        execution_handler=events.append,
    ).run(_definition(parent), 4, output_dir=_output(tmp_path, "nested-paths"))
    assert isinstance(complete, Success)
    remote_paths = {event.project_path for event in events if event.remote}
    assert {
        branch.name,
        f"{branch.name}/external/{middle.name}",
        f"{branch.name}/external/{middle.name}/external/{child.name}",
    } <= remote_paths
    events.clear()

    cancellation = CancellationToken.with_timeout(5)
    cancelled_at: list[float] = []

    def cancel_outer() -> None:
        while not pid_file.is_file() and not cancellation.expired:
            time.sleep(0.01)
        cancelled_at.append(time.monotonic())
        cancellation.cancel()

    canceller = threading.Thread(target=cancel_outer, daemon=True)
    canceller.start()
    with pytest.raises(ExecutionCancelled):
        Dispatcher(
            project_root=tmp_path,
            execution_handler=events.append,
            cancellation=cancellation,
        ).run(_definition(parent), -5, output_dir=_output(tmp_path, "outer"))
    canceller.join(timeout=1)
    assert pid_file.is_file(), "deep child did not start"
    assert cancelled_at and time.monotonic() - cancelled_at[0] < 3
    _wait_until_gone(int(pid_file.read_text("utf-8")))


def _external_subroutine_workflow(
    branch: Path,
) -> WorkflowDefinition[int, tuple[int, int], None, object]:
    def adapt(
        input: int,
        state: EmptyState,
        context: CallContext[None, None, int, tuple[int, int]],
        /,
    ) -> Success[tuple[int, int], EmptyState]:
        return Success(output=context.invoke(input), state=state)

    enter = PortDefinition(id=NodeId("root_enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    call = NodeDefinition[EmptyState, object](
        id=NodeId("call [branch]"),
        name="Call external subroutine",
        state_type=EmptyState,
        operation=SubroutineCall(
            definition_id=GraphId("branch_project.main"),
            definition_module="branch_project.subroutines.main",
            params_types={
                (branch.name, GraphId("branch_project.main")): type(None),
                (
                    f"{branch.name}/external/child",
                    GraphId("child_project.main"),
                ): type(None),
            },
            profile_arguments={},
            session_arguments={},
            project_path=branch.name,
        ),
    )
    parent: GraphDefinition[int, tuple[int, int], None, object] = GraphDefinition(
        id=GraphId("parent.main"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(call,),
        edges=(
            EdgeDefinition(
                id=EdgeId("in"),
                source=enter.id,
                target=call.id,
                visit=CallVisitDefinition(implementation=adapt),
            ),
            _edge("out", call.id, exit_.id),
        ),
    )
    return _definition(parent)


@pytest.mark.parametrize(
    ("in_process", "target"),
    ((False, "child_project.main"), (True, "child_project.main")),
)
def test_external_subroutine_uses_one_dispatcher_for_nested_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_process: bool,
    target: str,
) -> None:
    for name in tuple(sys.modules):
        if name == "branch_project" or name.startswith("branch_project."):
            monkeypatch.delitem(sys.modules, name)
        if name == "child_project" or name.startswith("child_project."):
            monkeypatch.delitem(sys.modules, name)
    branch = _bridge_project(
        tmp_path,
        "external/child",
        target,
        in_process=in_process,
        enter_id="branch_enter",
    )
    child = _child_project(branch / "external", enter_id="child_enter")
    monkeypatch.setattr(
        sys, "path", [str(branch / "src"), str(child / "src"), *sys.path]
    )
    definition = _external_subroutine_workflow(branch)
    events: list[NodeExecution | EdgeExecution] = []
    output = _output(tmp_path, "external-subroutine")
    dispatchers: list[Dispatcher] = []
    original_init = Dispatcher.__init__

    def record_dispatcher(
        dispatcher: Dispatcher, *args: Any, **kwargs: Any
    ) -> None:
        dispatchers.append(dispatcher)
        original_init(dispatcher, *args, **kwargs)

    monkeypatch.setattr(Dispatcher, "__init__", record_dispatcher)

    def observe(event: NodeExecution | EdgeExecution) -> None:
        events.append(event)
        if (
            isinstance(event, NodeExecution)
            and event.graph_id == GraphId("child_project.main")
            and event.node_id == NodeId("work")
        ):
            branch_output = _only_call_output(output)
            child_output = _only_call_output(branch_output)
            assert call_reports(output / "config.md") == [
                (branch_output / "config.md").resolve()
            ]
            assert call_reports(branch_output / "config.md") == [
                (child_output / "config.md").resolve()
            ]
            assert not (output / "stats.md").exists()

    result = Dispatcher(project_root=tmp_path, execution_handler=observe).run(
        definition, 4, output_dir=output
    )

    assert isinstance(result, Success)
    assert result.output[0] == 5
    assert len(dispatchers) == 1
    assert branch.name in {event.project_path for event in events}
    nested = [
        event
        for event in events
        if event.project_path == f"{branch.name}/external/child"
    ]
    assert nested
    assert all(event.remote is not in_process for event in nested)
    branch_output = _only_call_output(output)
    child_output = _only_call_output(branch_output)
    for name in ("config.md", "stats.md"):
        assert call_reports(output / name) == [(branch_output / name).resolve()]
        assert call_reports(branch_output / name) == [(child_output / name).resolve()]
        assert call_reports(child_output / name) == []
        assert set(output.rglob(name)) == {
            output / name,
            branch_output / name,
            child_output / name,
        }
    assert not list(output.rglob("configuration.md"))
    assert table_rows(output / "config.md") == [["input", "4"], ["params", "None"]]
    for report_dir, graph_id, node_type in (
        (output, "parent.main", "subroutine_call"),
        (
            branch_output,
            "branch_project.main",
            "subroutine_call" if in_process else "workflow_call",
        ),
        (child_output, "child_project.main", "python"),
    ):
        rows = table_rows(report_dir / "stats.md", "Nodes")
        assert {row[1] for row in rows} == {graph_id}
        assert {row[3] for row in rows} == {"enter", "exit", node_type}
    assert table_rows(branch_output / "config.md") == [["params", "None"]]
    assert table_rows(child_output / "config.md") == [["params", "None"]]


def test_resume_continues_inside_an_external_in_process_subroutine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(sys.modules):
        if name == "branch_project" or name.startswith("branch_project."):
            monkeypatch.delitem(sys.modules, name)
        if name == "child_project" or name.startswith("child_project."):
            monkeypatch.delitem(sys.modules, name)
    branch = _bridge_project(
        tmp_path,
        "external/child",
        "child_project.main",
        in_process=True,
        enter_id="branch_enter",
    )
    child = _child_project(branch / "external", enter_id="child_enter")
    monkeypatch.setattr(
        sys, "path", [str(branch / "src"), str(child / "src"), *sys.path]
    )
    definition = _external_subroutine_workflow(branch)
    output = _output(tmp_path, "external-subroutine-resume")
    interrupted = False

    def interrupt_in_external_child(event: NodeExecution | EdgeExecution) -> None:
        nonlocal interrupted
        if (
            not interrupted
            and isinstance(event, NodeExecution)
            and event.graph_id == GraphId("child_project.main")
            and event.node_id == NodeId("work")
            and event.status.value == "running"
        ):
            interrupted = True
            raise KeyboardInterrupt("external child interrupted")

    with pytest.raises(KeyboardInterrupt, match="external child interrupted"):
        Dispatcher(
            project_root=tmp_path,
            execution_handler=interrupt_in_external_child,
        ).run(
            definition,
            4,
            output_dir=output,
            checkpointing=CheckpointPolicy.REQUIRED,
        )

    store = RunStore.open(output)
    assert store.manifest().status is RunStatus.INTERRUPTED
    latest = store.manifest().checkpoints.latest_restorable
    assert latest is not None
    snapshot = decode_continuation(store.checkpoint_shard(latest, "runtime.pkl"))
    graph_paths = [
        frame.definition.project_path
        for frame in snapshot.frames
        if isinstance(frame, GraphFrameSnapshot)
    ]
    assert graph_paths == [".", branch.name, f"{branch.name}/external/child"]

    result = Dispatcher(project_root=tmp_path).resume(definition, output_dir=output)

    assert result.output[0] == 5
    assert store.manifest().status is RunStatus.SUCCEEDED
