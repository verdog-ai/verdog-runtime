from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path
from threading import Thread
from types import ModuleType
from typing import cast

import pytest
from report_helpers import table_rows

from verdog_runtime import CancellationToken, ExecutionCancelled
from verdog_runtime.agents import (
    AgentInvocationError,
    AgentReply,
    AgentRequest,
    AgentSessionAction,
    AgentSessionCapabilities,
)
from verdog_runtime.agents._command import CommandResult, json_objects
from verdog_runtime.agents.claude import ClaudeInvoker
from verdog_runtime.agents.codex import CodexInvoker
from verdog_runtime.declarations import (
    Agent,
    AgentAccess,
    AgentNodeContext,
    AgentProfileDefinition,
    AgentProfileParameter,
    AgentSessionDefinition,
    AgentSessionParameter,
    CallContext,
    CallVisitDefinition,
    EdgeDefinition,
    GraphDefinition,
    NodeContext,
    NodeDefinition,
    ParameterAddress,
    ParameterType,
    PortDefinition,
    SubroutineCall,
    SubroutineDefinition,
    Success,
    VisitDefinition,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import (
    AgentProfileId,
    AgentSessionId,
    EdgeId,
    GraphId,
    NodeId,
    ProviderSessionId,
    RunId,
)
from verdog_runtime.interpreter import (
    CheckpointPolicy,
    Dispatcher,
    validate_graph,
)
from verdog_runtime.interpreter._agents import SessionResource
from verdog_runtime.interpreter._calls import subroutine_scope
from verdog_runtime.interpreter._invocations import InvocationJournalError


@dataclass(frozen=True, slots=True)
class EmptyState:
    pass


class CustomAgent(Agent):
    pass


AgentImplementation = Callable[
    [object, EmptyState, AgentNodeContext],
    Success[object, EmptyState],
]
_workflow_module_ids = count()


def _agent_graph(
    implementation: AgentImplementation,
    /,
    *,
    graph_id: str = "agent.graph",
    profile_id: str = "default",
    session_id: str = "conversation",
    operation_type: type[Agent] = Agent,
) -> GraphDefinition[object, object, None, object]:
    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    node = NodeDefinition[EmptyState, object](
        id=NodeId("agent"),
        name="Agent",
        state_type=EmptyState,
        operation=operation_type(
            profile=AgentProfileId(profile_id),
            session=AgentSessionId(session_id),
        ),
    )
    return GraphDefinition(
        id=GraphId(graph_id),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=(node,),
        profile_parameters=(
            AgentProfileParameter(
                id=AgentProfileId(profile_id), name=profile_id
            ),
        ),
        session_parameters=(
            AgentSessionParameter(
                id=AgentSessionId(session_id),
                name=session_id,
            ),
        ),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter-agent"),
                source=enter.id,
                target=node.id,
                visit=VisitDefinition(
                    implementation=implementation,
                ),
            ),
            EdgeDefinition(
                id=EdgeId("agent-exit"), source=node.id, target=exit_.id
            ),
        ),
    )


def _workflow(
    graph: GraphDefinition[object, object, None, object],
    configuration: WorkflowConfiguration | None = None,
    *,
    workflow_id: GraphId | None = None,
) -> WorkflowDefinition[object, object, None, object]:
    subroutine = SubroutineDefinition(graph=graph)
    module_name = f"runtime_test_workflow_entry_{next(_workflow_module_ids)}"
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
        id=workflow_id or GraphId(f"{graph.id}_workflow"),
        input_type=object,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module_name,
            params_types=params_types,
            profile_arguments={
                parameter.id: parameter.id
                for parameter in graph.profile_parameters
            },
            session_arguments={
                parameter.id: parameter.id
                for parameter in graph.session_parameters
            },
        ),
        configuration=(
            WorkflowConfiguration() if configuration is None else configuration
        ),
        sessions=tuple(
            AgentSessionDefinition(
                id=parameter.id,
                name=parameter.name,
                persistent=True,
            )
            for parameter in graph.session_parameters
        ),
    )


def test_profile_and_session_identifiers_have_separate_namespaces() -> None:
    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        return Success(output=input, state=state)

    graph = _agent_graph(implementation, session_id="default")
    validate_graph(graph)

    def profile(input: object, params: None, /) -> _RecordingInvoker:
        return _RecordingInvoker()

    with pytest.raises(ValueError, match="duplicate agent profile id: default"):
        validate_graph(
            replace(
                graph,
                profiles=(
                    AgentProfileDefinition(
                        id=AgentProfileId("default"),
                        name="duplicate",
                        implementation=profile,
                    ),
                ),
            )
        )

    with pytest.raises(ValueError, match="duplicate agent session id: default"):
        validate_graph(
            replace(
                graph,
                sessions=(
                    AgentSessionDefinition(
                        id=AgentSessionId("default"),
                        name="collision",
                        persistent=False,
                    ),
                ),
            )
        )


def _subroutine_project(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    definition: SubroutineDefinition[object, object, None, object],
    /,
) -> Path:
    project = root / "project"
    project.mkdir()
    root_id = GraphId("runtime_test_project.main")
    root_module = "runtime_test_project.subroutines.main"
    module_name = _subroutine_module(definition)

    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    root_definition = SubroutineDefinition[object, object, None, object](
        graph=GraphDefinition[object, object, None, object](
            id=root_id,
            params_type=type(None),
            enter=enter,
            exit=exit_,
            failure=PortDefinition(id=NodeId("failure")),
            nodes=(),
            edges=(
                EdgeDefinition(
                    id=EdgeId("pass"), source=enter.id, target=exit_.id
                ),
            ),
        ),
    )

    def add_module(name: str, **attributes: object) -> None:
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    add_module("runtime_test_project")
    add_module("runtime_test_project.subroutines")
    add_module("runtime_test_project.workflows")
    add_module(root_module, definition=lambda: root_definition)
    add_module(module_name, definition=lambda: definition)
    return project


def _subroutine_module(
    definition: SubroutineDefinition[object, object, None, object], /
) -> str:
    target_name = str(definition.graph.id).rsplit("__", 1)[-1]
    return "runtime_test_project.subroutines.main.subroutines." + target_name


def _register_root_workflow(
    monkeypatch: pytest.MonkeyPatch,
    definition: WorkflowDefinition[object, object, None, object],
    /,
) -> None:
    module = ModuleType("runtime_test_project.workflows.main")
    module.__dict__["definition"] = lambda: definition
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_subroutine_module_requires_matching_definition_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        return Success(output=input, state=state)

    definition = SubroutineDefinition(
        graph=_agent_graph(
            implementation, graph_id="runtime_test_project.main__actual"
        )
    )
    project = _subroutine_project(tmp_path, monkeypatch, definition)
    requested = ModuleType(
        "runtime_test_project.subroutines.main.subroutines.requested"
    )
    requested.__dict__["definition"] = lambda: definition
    monkeypatch.setitem(sys.modules, requested.__name__, requested)

    with pytest.raises(ValueError, match="definition id does not match"):
        subroutine_scope(
            project,
            SubroutineCall(
                definition_id=GraphId("runtime_test_project.main__requested"),
                definition_module=requested.__name__,
                params_types={},
                profile_arguments={},
                session_arguments={},
            ),
        )


@dataclass(slots=True)
class _RecordingInvoker:
    session_provider = "test"

    requests: list[AgentRequest] = field(default_factory=list[AgentRequest])
    poison_first_response: bool = False

    def __call__(self, request: AgentRequest, /) -> AgentReply:
        self.requests.append(request)
        if self.poison_first_response and len(self.requests) == 1:
            (request.artifact_dir / "response.txt").mkdir()
        number = len(self.requests)
        return AgentReply(
            text=f"reply-{number}",
            provider_session_id=ProviderSessionId(f"session-{number}"),
        )


@dataclass(slots=True)
class _OtherRecordingInvoker(_RecordingInvoker):
    session_provider = "other"


@dataclass(slots=True)
class _InterruptingInvoker:
    session_provider = "interrupting-test"
    session_capabilities = AgentSessionCapabilities(fork_latest=True)

    requests: list[AgentRequest] = field(default_factory=list[AgentRequest])
    interrupt_first: bool = True

    def __call__(self, request: AgentRequest, /) -> AgentReply:
        self.requests.append(request)
        if self.interrupt_first and len(self.requests) == 1:
            raise KeyboardInterrupt
        return AgentReply(
            text=f"reply-{len(self.requests)}",
            provider_session_id=ProviderSessionId(
                f"session-{len(self.requests)}"
            ),
        )


@dataclass(slots=True)
class _ForkingRecordingInvoker(_RecordingInvoker):
    session_capabilities = AgentSessionCapabilities(fork_latest=True)


def test_restored_persistent_session_forks_once_then_continues() -> None:
    invoker = _ForkingRecordingInvoker()
    resource = SessionResource(
        persistent=True,
        provider="test",
        provider_session_id=ProviderSessionId("checkpoint-anchor"),
        access=AgentAccess.READ_ONLY,
        copy_on_write=True,
        require_copy_on_write=True,
    )

    source, action = resource.request(
        AgentSessionId("conversation"), invoker, AgentAccess.READ_ONLY
    )
    assert (source, action) == (
        ProviderSessionId("checkpoint-anchor"),
        AgentSessionAction.FORK,
    )
    resource.advance(
        "test",
        AgentAccess.READ_ONLY,
        ProviderSessionId("independent-branch"),
        action,
        source,
    )

    source, action = resource.request(
        AgentSessionId("conversation"), invoker, AgentAccess.READ_ONLY
    )
    assert (source, action) == (
        ProviderSessionId("independent-branch"),
        AgentSessionAction.CONTINUE,
    )
    assert not resource.copy_on_write
    assert resource.branch_supported is True


def test_required_restored_session_rejects_provider_without_branching() -> None:
    resource = SessionResource(
        persistent=True,
        provider="test",
        provider_session_id=ProviderSessionId("checkpoint-anchor"),
        access=AgentAccess.READ_ONLY,
        copy_on_write=True,
        require_copy_on_write=True,
    )

    with pytest.raises(
        AgentInvocationError, match="cannot fork persistent session"
    ):
        resource.request(
            AgentSessionId("conversation"),
            _RecordingInvoker(),
            AgentAccess.READ_ONLY,
        )
    assert resource.branch_supported is False


@pytest.mark.parametrize(
    ("profile_id", "session_id", "message"),
    (("", "conversation", "profile id"), ("default", "", "session id")),
)
def test_agent_identifiers_must_not_be_empty(
    profile_id: str,
    session_id: str,
    message: str,
) -> None:
    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        return Success(output=input, state=state)

    with pytest.raises(ValueError, match=message):
        validate_graph(
            _agent_graph(
                implementation,
                profile_id=profile_id,
                session_id=session_id,
            )
        )


@pytest.mark.parametrize("operation_type", [Agent, CustomAgent])
def test_agent_implementation_may_choose_not_to_invoke(
    tmp_path: Path, operation_type: type[Agent]
) -> None:
    invoker = _RecordingInvoker()

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        return Success(output="skipped", state=state)

    result = Dispatcher().run(
        _workflow(
            _agent_graph(implementation, operation_type=operation_type),
            WorkflowConfiguration(
                profile_arguments={
                    AgentProfileId("default"): invoker,
                    AgentProfileId("unused"): invoker,
                }
            ),
        ),
        None,
        output_dir=tmp_path / "output",
    )
    assert isinstance(result, Success)
    assert result.output == "skipped"
    assert invoker.requests == []
    rows = table_rows(tmp_path / "output/stats.md", "Nodes")
    assert any(
        row[1:5] == ["agent.graph", "agent", "agent", "1"] for row in rows
    )


def test_missing_agent_profiles_fail_before_creating_outputs(
    tmp_path: Path,
) -> None:
    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        return Success(output=input, state=state)

    graph = _agent_graph(implementation, profile_id="zeta")
    first = cast(NodeDefinition[EmptyState, object], graph.nodes[0])
    graph = replace(
        graph,
        nodes=(
            first,
            replace(
                first,
                id=NodeId("another-agent"),
                operation=Agent(
                    profile=AgentProfileId("alpha"),
                    session=AgentSessionId("conversation"),
                ),
            ),
        ),
        profile_parameters=(
            *graph.profile_parameters,
            AgentProfileParameter(id=AgentProfileId("alpha"), name="alpha"),
        ),
    )
    output = tmp_path / "output"

    with pytest.raises(
        ValueError,
        match=(
            "workflow configuration is missing profile arguments: alpha, zeta"
        ),
    ):
        Dispatcher().run(_workflow(graph), None, output_dir=output)

    assert not output.exists()


def test_workflow_entry_maps_caller_profile_and_session_ids(
    tmp_path: Path,
) -> None:
    invoker = _RecordingInvoker()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        return Success(
            output=context.invoke("mapped", workspace=workspace), state=state
        )

    graph = _agent_graph(
        implementation,
        profile_id="target_profile",
        session_id="target_session",
    )
    workflow = _workflow(
        graph,
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("caller_profile"): invoker}
        ),
    )
    workflow = replace(
        workflow,
        entry=replace(
            workflow.entry,
            profile_arguments={
                AgentProfileId("target_profile"): AgentProfileId(
                    "caller_profile"
                )
            },
            session_arguments={
                AgentSessionId("target_session"): AgentSessionId(
                    "caller_session"
                )
            },
        ),
        sessions=(
            AgentSessionDefinition(
                id=AgentSessionId("caller_session"),
                name="caller session",
                persistent=True,
            ),
        ),
    )

    result = Dispatcher().run(workflow, None, output_dir=tmp_path / "output")

    assert result.output == "reply-1"
    assert len(invoker.requests) == 1


def test_agent_sessions_are_run_local_and_invocations_are_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invoker = _RecordingInvoker()

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        assert context.graph_id == GraphId("agent.graph")
        first = context.invoke("first", workspace=workspace)
        second = context.invoke("second", workspace=workspace)
        return Success(output=(first, second), state=state)

    dispatcher = Dispatcher()
    definition = _workflow(
        _agent_graph(implementation),
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    first_root = tmp_path / "first-run"
    first = dispatcher.run(
        definition,
        None,
        output_dir=first_root,
    )
    second_root = tmp_path / "second-run"
    second = dispatcher.run(
        definition,
        None,
        output_dir=second_root,
    )

    assert isinstance(first, Success)
    assert first.output == ("reply-1", "reply-2")
    assert isinstance(second, Success)
    assert second.output == ("reply-3", "reply-4")
    assert [request.provider_session_id for request in invoker.requests] == [
        None,
        ProviderSessionId("session-1"),
        None,
        ProviderSessionId("session-3"),
    ]
    assert [
        request.artifact_dir.relative_to(first_root)
        for request in invoker.requests[:2]
    ] == [
        Path("agent/000001/invocations/000001"),
        Path("agent/000001/invocations/000002"),
    ]
    for request, prompt, response in zip(
        invoker.requests,
        ("first", "second", "first", "second"),
        ("reply-1", "reply-2", "reply-3", "reply-4"),
        strict=True,
    ):
        assert request.prompt == prompt
        assert request.profile_id == AgentProfileId("default")
        assert request.session_id == AgentSessionId("conversation")
        assert request.persistent is True
        assert request.workspace == workspace.resolve()
        assert request.access is AgentAccess.READ_ONLY
        assert (
            request.node_context.output_dir
            == request.artifact_dir.parent.parent
        )
        assert (request.artifact_dir / "prompt.txt").read_text(
            "utf-8"
        ) == prompt
        assert (request.artifact_dir / "response.txt").read_text(
            "utf-8"
        ) == response


def test_agent_session_advances_before_response_artifact_write(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invoker = _RecordingInvoker(poison_first_response=True)

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        try:
            context.invoke("first", workspace=workspace)
        except IsADirectoryError:
            (context.output_dir / "invocations/000001/response.txt").rmdir()
        else:  # pragma: no cover - deliberately poisoned by the fake invoker
            raise AssertionError(
                "response artifact write unexpectedly succeeded"
            )
        response = context.invoke("second", workspace=workspace)
        return Success(output=response, state=state)

    result = Dispatcher().run(
        _workflow(
            _agent_graph(implementation),
            WorkflowConfiguration(
                profile_arguments={AgentProfileId("default"): invoker}
            ),
        ),
        None,
        output_dir=tmp_path / "output",
    )
    assert isinstance(result, Success)
    assert result.output == "reply-2"
    assert [request.provider_session_id for request in invoker.requests] == [
        None,
        ProviderSessionId("session-1"),
    ]


def test_persistent_agent_session_cannot_cross_providers(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        return Success(
            output=context.invoke(str(input), workspace=workspace), state=state
        )

    graph = _agent_graph(implementation)
    first = cast(NodeDefinition[EmptyState, object], graph.nodes[0])
    second = replace(
        first,
        id=NodeId("other-agent"),
        operation=Agent(
            profile=AgentProfileId("other"),
            session=AgentSessionId("conversation"),
        ),
    )
    graph = replace(
        graph,
        nodes=(first, second),
        profile_parameters=(
            *graph.profile_parameters,
            AgentProfileParameter(id=AgentProfileId("other"), name="other"),
        ),
        edges=(
            graph.edges[0],
            EdgeDefinition(
                id=EdgeId("agent-other"),
                source=first.id,
                target=second.id,
                visit=VisitDefinition(implementation=implementation),
            ),
            EdgeDefinition(
                id=EdgeId("other-exit"), source=second.id, target=graph.exit.id
            ),
        ),
    )

    with pytest.raises(
        RuntimeError, match="belongs to provider test, not other"
    ):
        (
            Dispatcher().run(
                _workflow(
                    graph,
                    WorkflowConfiguration(
                        profile_arguments={
                            AgentProfileId("default"): _RecordingInvoker(),
                            AgentProfileId("other"): _OtherRecordingInvoker(),
                        }
                    ),
                ),
                None,
                output_dir=tmp_path / "output",
            )
        )


def test_persistent_agent_session_can_move_workspace_but_keeps_access(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    invoker = _RecordingInvoker()

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        context.invoke("first", workspace=workspace)
        context.invoke("second", workspace=other_workspace)
        context.invoke(
            "third",
            workspace=other_workspace,
            access=AgentAccess.WORKSPACE_WRITE,
        )
        return Success(output=None, state=state)

    with pytest.raises(RuntimeError, match="must keep one access mode"):
        (
            Dispatcher().run(
                _workflow(
                    _agent_graph(implementation),
                    WorkflowConfiguration(
                        profile_arguments={AgentProfileId("default"): invoker}
                    ),
                ),
                None,
                output_dir=tmp_path / "output",
            )
        )
    assert [request.workspace for request in invoker.requests] == [
        workspace.resolve(),
        other_workspace.resolve(),
    ]
    assert [request.provider_session_id for request in invoker.requests] == [
        None,
        ProviderSessionId("session-1"),
    ]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (object(), "did not return AgentReply"),
        (AgentReply(text=cast(str, 7)), "text must be a string"),
        (AgentReply(text="reply"), "returned no session id"),
        (
            AgentReply(text="reply", provider_session_id=ProviderSessionId("")),
            "must be a non-empty string",
        ),
        (
            AgentReply(
                text="reply", provider_session_id=cast(ProviderSessionId, 7)
            ),
            "must be a non-empty string",
        ),
    ],
)
def test_agent_rejects_invalid_replies(
    tmp_path: Path, reply: object, message: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    @dataclass(frozen=True, slots=True)
    class InvalidInvoker:
        session_provider = "test"

        def __call__(self, request: AgentRequest, /) -> AgentReply:
            return cast(AgentReply, reply)

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        response = context.invoke("prompt", workspace=workspace)
        return Success(output=response, state=state)

    with pytest.raises((RuntimeError, TypeError), match=message):
        (
            Dispatcher().run(
                _workflow(
                    _agent_graph(implementation),
                    WorkflowConfiguration(
                        profile_arguments={
                            AgentProfileId("default"): InvalidInvoker()
                        }
                    ),
                ),
                None,
                output_dir=tmp_path / "output",
            )
        )


def test_subroutines_share_the_owning_run_agent_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invoker = _RecordingInvoker()

    def child_agent(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        response = context.invoke(str(input), workspace=workspace)
        return Success(output=response, state=state)

    child = _agent_graph(
        child_agent,
        graph_id="runtime_test_project.main__child",
        session_id="child_session",
    )
    child = replace(
        child,
        profile_parameters=(
            *child.profile_parameters,
            AgentProfileParameter(
                id=AgentProfileId("fallback"), name="fallback"
            ),
        ),
        session_parameters=(
            *child.session_parameters,
            AgentSessionParameter(
                id=AgentSessionId("retry_session"),
                name="retry session",
            ),
        ),
    )
    child_definition = SubroutineDefinition(graph=child)
    project = _subroutine_project(tmp_path, monkeypatch, child_definition)
    child_module = _subroutine_module(child_definition)

    def adapter(
        input: object,
        state: EmptyState,
        context: CallContext[object, object, object, object],
        /,
    ) -> Success[object, EmptyState]:
        return Success(
            output=context.invoke(input),
            state=state,
        )

    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    calls = tuple(
        NodeDefinition[EmptyState, object](
            id=NodeId(f"call-{number}"),
            name=f"Call {number}",
            state_type=EmptyState,
            operation=SubroutineCall(
                definition_id=child.id,
                definition_module=child_module,
                params_types={(".", child.id): type(None)},
                profile_arguments={
                    AgentProfileId("default"): AgentProfileId("default"),
                    AgentProfileId("fallback"): AgentProfileId("fallback"),
                },
                session_arguments={
                    AgentSessionId("child_session"): AgentSessionId(
                        "conversation"
                    ),
                    AgentSessionId("retry_session"): AgentSessionId(
                        "retry_session"
                    ),
                },
            ),
        )
        for number in (1, 2)
    )
    parent: GraphDefinition[object, object, None, object] = GraphDefinition(
        id=GraphId("runtime_test_project.main"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=calls,
        profile_parameters=(
            AgentProfileParameter(id=AgentProfileId("default"), name="default"),
            AgentProfileParameter(
                id=AgentProfileId("fallback"), name="fallback"
            ),
        ),
        session_parameters=(
            AgentSessionParameter(
                id=AgentSessionId("conversation"),
                name="conversation",
            ),
            AgentSessionParameter(
                id=AgentSessionId("retry_session"),
                name="retry session",
            ),
        ),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter-call-1"),
                source=enter.id,
                target=calls[0].id,
                visit=CallVisitDefinition(implementation=adapter),
            ),
            EdgeDefinition(
                id=EdgeId("call-1-call-2"),
                source=calls[0].id,
                target=calls[1].id,
                visit=CallVisitDefinition(implementation=adapter),
            ),
            EdgeDefinition(
                id=EdgeId("call-2-exit"), source=calls[1].id, target=exit_.id
            ),
        ),
    )

    workflow = _workflow(
        parent,
        WorkflowConfiguration(
            profile_arguments={
                AgentProfileId("default"): invoker,
                AgentProfileId("fallback"): invoker,
            }
        ),
        workflow_id=GraphId("runtime_test_project.main"),
    )
    _register_root_workflow(monkeypatch, workflow)
    result = Dispatcher(project_root=project).run(
        workflow,
        "start",
        output_dir=tmp_path / "output",
    )
    assert isinstance(result, Success)
    assert [request.provider_session_id for request in invoker.requests] == [
        None,
        ProviderSessionId("session-1"),
    ]
    assert all(
        request.node_context.graph_id == child.id
        for request in invoker.requests
    )
    operation = cast(SubroutineCall, calls[0].operation)
    incomplete = replace(
        operation,
        profile_arguments={
            AgentProfileId("default"): AgentProfileId("default")
        },
    )
    incomplete_workflow = _workflow(
        replace(
            parent,
            nodes=(replace(calls[0], operation=incomplete), *calls[1:]),
        ),
        workflow.configuration,
        workflow_id=workflow.id,
    )
    _register_root_workflow(monkeypatch, incomplete_workflow)
    with pytest.raises(
        ValueError, match="profile resource arguments do not match"
    ):
        Dispatcher(project_root=project).run(
            incomplete_workflow,
            "start",
            output_dir=tmp_path / "incomplete-output",
        )

    wrong_kind = replace(
        operation,
        profile_arguments={
            AgentProfileId("default"): AgentProfileId("conversation"),
            AgentProfileId("fallback"): AgentProfileId("fallback"),
        },
    )
    wrong_kind_workflow = _workflow(
        replace(
            parent,
            nodes=(replace(calls[0], operation=wrong_kind), *calls[1:]),
        ),
        workflow.configuration,
        workflow_id=workflow.id,
    )
    _register_root_workflow(monkeypatch, wrong_kind_workflow)
    with pytest.raises(ValueError, match="caller profile resource is missing"):
        Dispatcher(project_root=project).run(
            wrong_kind_workflow,
            "start",
            output_dir=tmp_path / "wrong-kind-output",
        )

    ephemeral_workflow = replace(
        workflow,
        sessions=tuple(
            replace(session, persistent=False) for session in workflow.sessions
        ),
    )
    _register_root_workflow(monkeypatch, ephemeral_workflow)
    Dispatcher(project_root=project).run(
        ephemeral_workflow,
        "start",
        output_dir=tmp_path / "ephemeral-output",
    )
    assert [
        request.provider_session_id for request in invoker.requests[-2:]
    ] == [
        None,
        None,
    ]
    assert all(request.persistent is False for request in invoker.requests[-2:])


def test_local_agent_resources_are_created_per_subroutine_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invoker = _RecordingInvoker()
    initialized: list[object] = []

    def child_agent(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        first = context.invoke("first", workspace=workspace)
        second = context.invoke("second", workspace=workspace)
        return Success(output=(first, second), state=state)

    def profile(input: object, params: None, /) -> _RecordingInvoker:
        initialized.append(input)
        return invoker

    child = replace(
        _agent_graph(
            child_agent,
            graph_id="runtime_test_project.main__child",
            session_id="local_session",
        ),
        profiles=(
            AgentProfileDefinition(
                id=AgentProfileId("default"),
                name="default",
                implementation=profile,
            ),
        ),
        profile_parameters=(),
        sessions=(
            AgentSessionDefinition(
                id=AgentSessionId("local_session"),
                name="local session",
                persistent=True,
            ),
        ),
        session_parameters=(),
    )
    project = _subroutine_project(
        tmp_path, monkeypatch, SubroutineDefinition(graph=child)
    )
    child_module = _subroutine_module(SubroutineDefinition(graph=child))

    def adapter(
        input: object,
        state: EmptyState,
        context: CallContext[object, object, object, object],
        /,
    ) -> Success[object, EmptyState]:
        return Success(
            output=context.invoke(input),
            state=state,
        )

    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    failure = PortDefinition(id=NodeId("failure"))
    calls = tuple(
        NodeDefinition[EmptyState, object](
            id=NodeId(f"call-{number}"),
            name=f"Call {number}",
            state_type=EmptyState,
            operation=SubroutineCall(
                definition_id=child.id,
                definition_module=child_module,
                params_types={(".", child.id): type(None)},
                profile_arguments={},
                session_arguments={},
            ),
        )
        for number in (1, 2)
    )
    parent: GraphDefinition[object, object, None, object] = GraphDefinition(
        id=GraphId("runtime_test_project.main"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=failure,
        nodes=calls,
        edges=(
            EdgeDefinition(
                id=EdgeId("enter-call-1"),
                source=enter.id,
                target=calls[0].id,
                visit=CallVisitDefinition(implementation=adapter),
            ),
            EdgeDefinition(
                id=EdgeId("call-1-call-2"),
                source=calls[0].id,
                target=calls[1].id,
                visit=CallVisitDefinition(implementation=adapter),
            ),
            EdgeDefinition(
                id=EdgeId("call-2-exit"), source=calls[1].id, target=exit_.id
            ),
        ),
    )

    workflow = _workflow(
        parent,
        workflow_id=GraphId("runtime_test_project.main"),
    )
    _register_root_workflow(monkeypatch, workflow)
    result = Dispatcher(project_root=project).run(
        workflow,
        "start",
        output_dir=tmp_path / "output",
    )

    assert isinstance(result, Success)
    assert initialized == ["start", ("reply-1", "reply-2")]
    assert [request.provider_session_id for request in invoker.requests] == [
        None,
        ProviderSessionId("session-1"),
        None,
        ProviderSessionId("session-3"),
    ]


def _executable(path: Path, source: str, /) -> Path:
    path.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent(source), encoding="utf-8"
    )
    path.chmod(0o755)
    return path


def _provider_request(path: Path) -> AgentRequest:
    context: NodeContext[object] = NodeContext(
        run_id=RunId("run"),
        graph_id=GraphId("graph"),
        node_id=NodeId("agent"),
        edge_id=EdgeId("enter"),
        output_dir=path,
        params=None,
    )
    return AgentRequest(
        prompt="prompt",
        profile_id=AgentProfileId("profile"),
        session_id=AgentSessionId("session"),
        persistent=True,
        provider_session_id=None,
        node_context=context,
        workspace=path,
        access=AgentAccess.READ_ONLY,
        artifact_dir=path,
    )


@pytest.mark.parametrize("invoker", (CodexInvoker(), ClaudeInvoker()))
def test_builtin_agents_advertise_latest_session_forks(
    invoker: CodexInvoker | ClaudeInvoker,
) -> None:
    assert invoker.session_capabilities == AgentSessionCapabilities(
        fork_latest=True
    )


@pytest.mark.parametrize(
    ("last_message", "message"),
    [
        (None, "latest"),
        ("", ""),
        ("file response", "ignored"),
        (" ", "ignored"),
    ],
)
def test_codex_parses_events_once_and_preserves_response_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    last_message: str | None,
    message: str,
) -> None:
    records: tuple[object, ...] = (
        [],
        {"thread_id": 3, "session_id": "ignored"},
        {"thread_id": "", "session_id": "first", "conversation_id": "ignored"},
        {"item": {"type": "agent_message", "text": "earlier"}},
        {
            "type": "agent_reasoning",
            "text": " top ",
            "item": {"type": "reasoning", "text": "ignored"},
        },
        {"item": {"type": "reasoning", "summary": " nested "}},
        {
            "type": "agent_reasoning",
            "reasoning": " final thought ",
            "item": {"type": "agent_message", "text": message},
        },
        {"item": {"type": "agent_message", "text": 4}, "thread_id": "later"},
    )
    events = "not json\n" + "\n".join(json.dumps(record) for record in records)
    parsed: list[str] = []

    def parse(value: str, /) -> tuple[dict[str, object], ...]:
        parsed.append(value)
        return json_objects(value)

    def run(command: Sequence[str], _request: AgentRequest, /) -> CommandResult:
        if last_message is not None:
            Path(command[command.index("-o") + 1]).write_text(
                last_message, "utf-8"
            )
        return CommandResult(returncode=0, events=events, stderr="")

    monkeypatch.setattr("verdog_runtime.agents._command.json_objects", parse)
    monkeypatch.setattr("verdog_runtime.agents._command.run_command", run)
    reply = CodexInvoker()(_provider_request(tmp_path))

    assert parsed == [events]
    assert reply == AgentReply(
        text=last_message or message,
        provider_session_id=ProviderSessionId("first"),
    )
    assert (tmp_path / "reasoning.txt").read_text("utf-8") == (
        "top\n\nnested\n\nfinal thought"
    )


@pytest.mark.parametrize(
    ("terminal", "error"),
    [
        ({"result": "final"}, None),
        ({"result": ""}, None),
        ({"is_error": True, "result": "bad"}, "claude returned an error: bad"),
        ({"result": 7}, "claude result has no text"),
        (None, "claude returned no result event"),
    ],
)
def test_claude_uses_the_last_result_and_first_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: dict[str, object] | None,
    error: str | None,
) -> None:
    records: list[object] = [
        {"session_id": 4},
        {"session_id": "first"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    None,
                    {"type": "thinking", "thinking": " first thought "},
                    {"type": "thinking", "thinking": 3},
                    {"type": "text", "thinking": "ignored"},
                ]
            },
        },
        {"type": "assistant", "message": {"content": None}},
    ]
    if terminal is not None:
        records.extend(
            (
                {"type": "result", "is_error": True, "result": "earlier error"},
                {"type": "result", "result": 7},
                {"type": "result", "session_id": "later", **terminal},
            )
        )
    events = "\n".join(json.dumps(record) for record in records)
    parsed: list[str] = []

    def parse(value: str, /) -> tuple[dict[str, object], ...]:
        parsed.append(value)
        return json_objects(value)

    def run(
        _command: Sequence[str], _request: AgentRequest, /
    ) -> CommandResult:
        return CommandResult(returncode=0, events=events, stderr="")

    monkeypatch.setattr("verdog_runtime.agents._command.json_objects", parse)
    monkeypatch.setattr("verdog_runtime.agents._command.run_command", run)
    if error is not None:
        with pytest.raises(AgentInvocationError, match=error):
            ClaudeInvoker()(_provider_request(tmp_path))
    else:
        reply = ClaudeInvoker()(_provider_request(tmp_path))
        assert terminal is not None and reply.text == terminal["result"]
        assert reply.provider_session_id == ProviderSessionId("first")
        assert (tmp_path / "reasoning.txt").read_text(
            "utf-8"
        ) == "first thought"
    assert parsed == [events]


@pytest.mark.parametrize("provider", ("codex", "claude"))
def test_builtin_agents_fork_provider_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    commands: list[tuple[str, ...]] = []
    if provider == "codex":
        invoker: CodexInvoker | ClaudeInvoker = CodexInvoker()
        events = "\n".join(
            (
                json.dumps({"thread_id": "child"}),
                json.dumps(
                    {"item": {"type": "agent_message", "text": "reply"}}
                ),
            )
        )
    else:
        invoker = ClaudeInvoker()
        events = json.dumps(
            {
                "type": "result",
                "result": "reply",
                "session_id": "child",
            }
        )

    def run(command: Sequence[str], _request: AgentRequest, /) -> CommandResult:
        commands.append(tuple(command))
        return CommandResult(returncode=0, events=events, stderr="")

    monkeypatch.setattr("verdog_runtime.agents._command.run_command", run)
    request = replace(
        _provider_request(tmp_path),
        provider_session_id=ProviderSessionId("source"),
        provider_session_action=AgentSessionAction.FORK,
    )
    reply = invoker(request)

    assert reply == AgentReply(
        text="reply",
        provider_session_id=ProviderSessionId("child"),
    )
    assert len(commands) == 1
    command = commands[0]
    if provider == "codex":
        assert command[-3:] == ("fork", "source", "-")
        assert "resume" not in command
    else:
        resume = command.index("--resume")
        assert command[resume : resume + 3] == (
            "--resume",
            "source",
            "--fork-session",
        )
    metadata = cast(
        dict[str, object],
        json.loads((tmp_path / "metadata.json").read_text("utf-8")),
    )
    assert metadata["provider_session_action"] == "fork"
    assert metadata["provider_session_source"] == "source"
    assert metadata["provider_session"] == "child"


@pytest.mark.parametrize("provider", ("codex", "claude"))
@pytest.mark.parametrize(
    ("persistent", "source", "error"),
    (
        (True, None, "cannot fork without a provider session"),
        (False, ProviderSessionId("source"), "cannot fork a nonpersistent"),
    ),
)
def test_builtin_agents_reject_invalid_fork_requests(
    tmp_path: Path,
    provider: str,
    persistent: bool,
    source: ProviderSessionId | None,
    error: str,
) -> None:
    invoker: CodexInvoker | ClaudeInvoker = (
        CodexInvoker() if provider == "codex" else ClaudeInvoker()
    )
    request = replace(
        _provider_request(tmp_path),
        persistent=persistent,
        provider_session_id=source,
        provider_session_action=AgentSessionAction.FORK,
    )

    with pytest.raises(AgentInvocationError, match=error):
        invoker(request)


@pytest.mark.parametrize(
    "provider_session_id", (None, ProviderSessionId("source"))
)
def test_builtin_agent_forks_require_a_distinct_returned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_session_id: ProviderSessionId | None,
) -> None:
    def run(
        _command: Sequence[str], _request: AgentRequest, /
    ) -> CommandResult:
        event: dict[str, object] = {
            "item": {"type": "agent_message", "text": "reply"}
        }
        if provider_session_id is not None:
            event["thread_id"] = provider_session_id
        return CommandResult(returncode=0, events=json.dumps(event), stderr="")

    monkeypatch.setattr("verdog_runtime.agents._command.run_command", run)
    request = replace(
        _provider_request(tmp_path),
        provider_session_id=ProviderSessionId("source"),
        provider_session_action=AgentSessionAction.FORK,
    )
    error = "returned no provider session|returned its source provider session"

    with pytest.raises(AgentInvocationError, match=error):
        CodexInvoker()(request)

    metadata = cast(
        dict[str, object],
        json.loads((tmp_path / "metadata.json").read_text("utf-8")),
    )
    assert metadata["status"] == "failed"
    assert metadata["provider_session_action"] == "fork"
    assert metadata["provider_session_source"] == "source"


@pytest.mark.skipif(
    os.name == "nt", reason="test providers use executable scripts"
)
@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_builtin_agent_artifacts_and_commands(
    tmp_path: Path, provider: str
) -> None:
    workspaces = (tmp_path / "workspace", tmp_path / "resumed-workspace")
    for workspace in workspaces:
        workspace.mkdir()
    executable = _executable(
        tmp_path / provider,
        """
import json
from pathlib import Path
import sys

arguments = sys.argv[1:]
with Path("arguments.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")
prompt = sys.stdin.read()
if sys.argv[0].endswith("codex"):
    output = Path(arguments[arguments.index("-o") + 1])
    output.write_text("codex:" + prompt, encoding="utf-8")
    print(
        json.dumps(
            {"type": "thread.started", "thread_id": "codex-session"}
        )
    )
    print(
        json.dumps({"type": "agent_reasoning", "text": "codex thought"})
    )
else:
    print(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "claude thought",
                        }
                    ]
                },
            }
        )
    )
    print(
        json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": "claude:" + prompt,
                "session_id": "claude-session",
            }
        )
    )
print(provider + " stderr", file=sys.stderr)
""".replace("provider", repr(provider)),
    )
    profile_id = AgentProfileId(provider)
    if provider == "codex":
        invoker = CodexInvoker(
            executable=str(executable),
            model="provider-model",
            reasoning_effort="high",
            extra_args=("--color", "never"),
            web_search=True,
        )
        access = AgentAccess.READ_ONLY
        expected_session = ProviderSessionId("codex-session")
    else:
        invoker = ClaudeInvoker(
            executable=str(executable),
            model="provider-model",
            reasoning_effort="high",
            extra_args=("--max-budget-usd", "1"),
            web_search=True,
        )
        access = AgentAccess.READ_ONLY
        expected_session = ProviderSessionId("claude-session")

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        first = context.invoke("first", workspace=workspaces[0], access=access)
        second = context.invoke(
            "second", workspace=workspaces[1], access=access
        )
        return Success(output=(first, second), state=state)

    output = tmp_path / "output"
    result = Dispatcher().run(
        _workflow(
            _agent_graph(
                implementation,
                profile_id=provider,
                session_id=f"{provider}_session",
            ),
            WorkflowConfiguration(profile_arguments={profile_id: invoker}),
        ),
        None,
        output_dir=output,
    )
    assert isinstance(result, Success)
    assert result.output == (f"{provider}:first", f"{provider}:second")

    artifacts = sorted(output.glob("**/invocations/*"))
    assert len(artifacts) == 2
    for number, artifact in enumerate(artifacts, 1):
        assert (artifact / "prompt.txt").read_text("utf-8") == (
            "first" if number == 1 else "second"
        )
        assert (artifact / "response.txt").read_text("utf-8") == (
            f"{provider}:" + ("first" if number == 1 else "second")
        )
        assert (artifact / "events.jsonl").is_file()
        assert (artifact / "stderr.txt").read_text(
            "utf-8"
        ) == f"{provider} stderr\n"
        assert (artifact / "reasoning.txt").read_text(
            "utf-8"
        ) == f"{provider} thought"
        metadata = cast(
            dict[str, object],
            json.loads((artifact / "metadata.json").read_text("utf-8")),
        )
        assert metadata == {
            "access": "read-only",
            "duration": metadata["duration"],
            "model": "provider-model",
            "profile": provider,
            "provider": provider,
            "returncode": 0,
            "provider_session_action": "continue",
            "provider_session_source": (
                None if number == 1 else str(expected_session)
            ),
            "provider_session": str(expected_session),
            "session": f"{provider}_session",
            "status": "succeeded",
            "workspace": str(workspaces[number - 1].resolve()),
        }
        assert isinstance(metadata["duration"], float)
        assert not (artifact / "command.json").exists()
        assert not (artifact / "stdout.txt").exists()

    arguments = [
        cast(
            list[str],
            json.loads((workspace / "arguments.jsonl").read_text("utf-8")),
        )
        for workspace in workspaces
    ]
    assert len(arguments) == 2
    if provider == "codex":
        for index, command in enumerate(arguments):
            assert command[0] == "exec"
            assert command[1:3] == ["--color", "never"]
            assert command[-1] == "-"
            assert command[command.index("-C") + 1] == str(
                workspaces[index].resolve()
            )
            assert command.count("--sandbox") == 1
            sandbox = len(command) - 1 - command[::-1].index("--sandbox")
            assert command[sandbox + 1] == "read-only"
            assert "--search" in command
            if index == 0:
                assert "resume" not in command
            else:
                resume = command.index("resume")
                assert command[resume:] == ["resume", "codex-session", "-"]
                assert command.index("--json") < resume
                assert command.index("-o") < resume
                assert sandbox < resume
    else:
        for index, command in enumerate(arguments):
            assert command[1:3] == ["--max-budget-usd", "1"]
            assert command.count("--permission-mode") == 1
            assert command.count("--tools") == 1
            permission = (
                len(command) - 1 - command[::-1].index("--permission-mode")
            )
            tools = len(command) - 1 - command[::-1].index("--tools")
            assert command[permission + 1] == "plan"
            assert command[tools + 1] == "Read,Glob,Grep,WebSearch,WebFetch"
            assert command[command.index("--model") + 1] == "provider-model"
            assert command[command.index("--effort") + 1] == "high"
            if index == 0:
                assert "--resume" not in command
            else:
                assert (
                    command[command.index("--resume") + 1] == "claude-session"
                )


@pytest.mark.parametrize(
    ("constructor", "extra_args"),
    (
        (CodexInvoker, ("--sandbox", "danger-full-access")),
        (CodexInvoker, ("--cd", "/tmp")),
        (CodexInvoker, ("fork",)),
        (CodexInvoker, ("resume",)),
        (ClaudeInvoker, ("--fork-session",)),
        (ClaudeInvoker, ("--permission-mode=bypassPermissions",)),
    ),
)
def test_builtin_agents_reject_runtime_owned_extra_args(
    constructor: type[CodexInvoker] | type[ClaudeInvoker],
    extra_args: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="may not override runtime option"):
        constructor(extra_args=extra_args)


def _process_exists(pid: int, /) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until_gone(pid: int, /) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and _process_exists(pid):
        time.sleep(0.05)
    assert not _process_exists(pid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-tree assertion")
@pytest.mark.parametrize(
    "interruption",
    ("explicit", "deadline", "keyboard"),
)
def test_builtin_agent_cancellation_kills_and_reaps_process_tree(
    tmp_path: Path,
    interruption: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executable = _executable(
        tmp_path / "codex",
        """
from pathlib import Path
import subprocess
import sys
import time

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(30)"],
    start_new_session=True,
)
Path("pids").write_text(
    f"{__import__('os').getpid()} {child.pid}", encoding="ascii"
)
time.sleep(30)
""",
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    cancellation = CancellationToken.with_timeout(
        1.0 if interruption == "deadline" else 5.0
    )
    request = replace(
        _provider_request(artifact),
        workspace=workspace,
        cancellation=cancellation,
    )
    (artifact / "prompt.txt").write_text(request.prompt, "utf-8")
    pids_path = workspace / "pids"
    interrupter: Thread | None = None
    if interruption != "deadline":

        def interrupt_when_started() -> None:
            deadline = time.monotonic() + 5
            while not pids_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            if pids_path.exists():
                if interruption == "explicit":
                    cancellation.cancel()
                else:
                    os.kill(os.getpid(), signal.SIGINT)

        interrupter = Thread(target=interrupt_when_started, daemon=True)
        interrupter.start()

    try:
        if interruption == "keyboard":
            with pytest.raises(KeyboardInterrupt):
                CodexInvoker(executable=str(executable))(request)
        else:
            message = "cancelled" if interruption == "explicit" else "deadline"
            with pytest.raises(ExecutionCancelled, match=message):
                CodexInvoker(executable=str(executable))(request)
    finally:
        if interrupter is not None:
            interrupter.join(timeout=1)
    if interrupter is not None:
        assert not interrupter.is_alive()
    assert pids_path.exists()
    parent, child = (
        int(value) for value in pids_path.read_text("ascii").split()
    )
    _wait_until_gone(parent)
    _wait_until_gone(child)
    metadata = cast(
        dict[str, object],
        json.loads((artifact / "metadata.json").read_text("utf-8")),
    )
    assert metadata["status"] == "cancelled"
    assert (artifact / "prompt.txt").read_text("utf-8") == "prompt"
    assert (artifact / "events.jsonl").is_file()
    assert (artifact / "stderr.txt").is_file()
    assert not (artifact / "response.txt").exists()


def test_neutral_agent_package_does_not_import_providers() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import verdog_runtime.agents; "
            "assert 'verdog_runtime.agents.codex' not in sys.modules; "
            "assert 'verdog_runtime.agents.claude' not in sys.modules",
        ],
        check=True,
    )


def test_exact_resume_refuses_an_ambiguous_agent_request_without_retry(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invoker = _InterruptingInvoker()

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        return Success(
            output=context.invoke("prompt", workspace=workspace),
            state=state,
        )

    definition = _workflow(
        _agent_graph(implementation),
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    output = tmp_path / "output"
    with pytest.raises(KeyboardInterrupt):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                None,
                output_dir=output,
                checkpointing=CheckpointPolicy.AUTO,
            )
        )
    assert len(invoker.requests) == 1

    with pytest.raises(InvocationJournalError) as captured:
        (
            Dispatcher(project_root=tmp_path).resume(
                definition,
                output_dir=output,
            )
        )
    assert captured.value.code == "invocation.ambiguous"
    assert "--retry-incomplete" in str(captured.value)
    assert len(invoker.requests) == 1

    resumed = Dispatcher(project_root=tmp_path).resume(
        definition,
        output_dir=output,
        retry_incomplete=True,
    )
    assert resumed.output == "reply-2"
    assert len(invoker.requests) == 2


def test_exact_resume_replays_a_durably_completed_agent_reply(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invoker = _RecordingInvoker()
    node_attempts = 0

    def implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        nonlocal node_attempts
        response = context.invoke("prompt", workspace=workspace)
        node_attempts += 1
        if node_attempts == 1:
            raise KeyboardInterrupt
        return Success(output=response, state=state)

    definition = _workflow(
        _agent_graph(implementation),
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    output = tmp_path / "output"
    with pytest.raises(KeyboardInterrupt):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                None,
                output_dir=output,
                checkpointing=CheckpointPolicy.AUTO,
            )
        )
    assert len(invoker.requests) == 1

    resumed = Dispatcher(project_root=tmp_path).resume(
        definition,
        output_dir=output,
    )
    assert resumed.output == "reply-1"
    assert node_attempts == 2
    assert len(invoker.requests) == 1
    attempts = sorted(output.glob("agent/*/invocations/000001/response.txt"))
    assert [path.read_text("utf-8") for path in attempts] == [
        "reply-1",
        "reply-1",
    ]


def test_replayed_reply_advances_a_restored_copy_on_write_session_once(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    invoker = _InterruptingInvoker(interrupt_first=False)
    second_attempts = 0

    def first_implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        context.invoke("first", workspace=workspace)
        return Success(output=input, state=state)

    def second_implementation(
        input: object,
        state: EmptyState,
        context: AgentNodeContext,
        /,
    ) -> Success[object, EmptyState]:
        nonlocal second_attempts
        response = context.invoke("second", workspace=workspace)
        second_attempts += 1
        if second_attempts == 1:
            raise KeyboardInterrupt
        return Success(output=response, state=state)

    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    first = NodeDefinition[EmptyState, object](
        id=NodeId("first"),
        name="First",
        state_type=EmptyState,
        operation=Agent(
            profile=AgentProfileId("default"),
            session=AgentSessionId("conversation"),
        ),
    )
    second = replace(first, id=NodeId("second"), name="Second")
    graph = GraphDefinition[object, object, None, object](
        id=GraphId("agent.cow_replay"),
        params_type=type(None),
        enter=enter,
        exit=exit_,
        failure=PortDefinition(id=NodeId("failure")),
        nodes=(first, second),
        profile_parameters=(
            AgentProfileParameter(id=AgentProfileId("default"), name="default"),
        ),
        session_parameters=(
            AgentSessionParameter(
                id=AgentSessionId("conversation"),
                name="conversation",
            ),
        ),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter-first"),
                source=enter.id,
                target=first.id,
                visit=VisitDefinition(implementation=first_implementation),
            ),
            EdgeDefinition(
                id=EdgeId("first-second"),
                source=first.id,
                target=second.id,
                visit=VisitDefinition(implementation=second_implementation),
            ),
            EdgeDefinition(
                id=EdgeId("second-exit"),
                source=second.id,
                target=exit_.id,
            ),
        ),
    )
    definition = _workflow(
        graph,
        WorkflowConfiguration(
            profile_arguments={AgentProfileId("default"): invoker}
        ),
    )
    output = tmp_path / "output"
    with pytest.raises(KeyboardInterrupt):
        (
            Dispatcher(project_root=tmp_path).run(
                definition,
                None,
                output_dir=output,
                checkpointing=CheckpointPolicy.REQUIRED,
            )
        )

    assert len(invoker.requests) == 2
    assert invoker.requests[0].provider_session_id is None
    assert (
        invoker.requests[0].provider_session_action
        is AgentSessionAction.CONTINUE
    )
    assert invoker.requests[1].provider_session_id == ProviderSessionId(
        "session-1"
    )
    assert (
        invoker.requests[1].provider_session_action is AgentSessionAction.FORK
    )

    resumed = Dispatcher(project_root=tmp_path).resume(
        definition,
        output_dir=output,
    )
    assert resumed.output == "reply-2"
    assert second_attempts == 2
    assert len(invoker.requests) == 2


def test_web_search_is_off_unless_the_profile_opts_in(tmp_path: Path) -> None:
    request = _provider_request(tmp_path)
    claude = ClaudeInvoker()._command(request)  # pyright: ignore[reportPrivateUsage]
    assert claude[claude.index("--tools") + 1] == "Read,Glob,Grep"
    searching = ClaudeInvoker(web_search=True)._command(request)  # pyright: ignore[reportPrivateUsage]
    assert (
        searching[searching.index("--tools") + 1]
        == "Read,Glob,Grep,WebSearch,WebFetch"
    )
    codex = CodexInvoker()._command(request, tmp_path / "last.txt")  # pyright: ignore[reportPrivateUsage]
    assert "--search" not in codex
    searching_codex = CodexInvoker(web_search=True)._command(  # pyright: ignore[reportPrivateUsage]
        request, tmp_path / "last.txt"
    )
    assert "--search" in searching_codex
