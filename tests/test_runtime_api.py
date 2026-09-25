import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run(tmp_path: Path, source: str) -> None:
    """Exercise the runtime as a project does: importing the installed
    `verdog_runtime`, not a copy staged into a fake package."""

    package = tmp_path / "sample"
    package.mkdir()
    (package / "__init__.py").touch()
    environment = os.environ.copy()
    runtime_source = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(runtime_source), str(runtime_source / "tests"))
    )
    preamble = """\
from dataclasses import dataclass
from itertools import count
from pathlib import Path
import sys
from types import ModuleType
from report_helpers import call_reports, table_rows
from verdog_runtime.declarations import (
    SubroutineCall,
    SubroutineDefinition,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import GraphId

_output_ids = count()
_definition_module_ids = count()
import sample


@dataclass(frozen=True, slots=True)
class EmptyState:
    pass


def output_dir():
    return Path("outputs") / str(next(_output_ids))


def trace_paths(path):
    return [
        line.split("] START ", 1)[1]
        for line in path.read_text("utf-8").splitlines()
        if "] START " in line
    ]


def definition(graph):
    subroutine = SubroutineDefinition(graph=graph)
    module = install_definition_module(subroutine)
    params_types = {(".", graph.id): graph.params_type}
    for node in graph.nodes:
        if isinstance(node.operation, SubroutineCall):
            params_types.update(node.operation.params_types)
    return WorkflowDefinition(
        id=GraphId(f"{graph.id}_workflow"),
        input_type=object,
        entry=SubroutineCall(
            definition_id=graph.id,
            definition_module=module,
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


def install_definition_module(definition):
    name = f"sample.subroutine_{next(_definition_module_ids)}"
    declaration = ModuleType(name)
    declaration.definition = lambda: definition
    sys.modules[name] = declaration
    return name


"""
    subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(source)],
        cwd=tmp_path,
        env=environment,
        check=True,
    )


def test_direct_state_effects_and_atomic_routing(tmp_path: Path) -> None:
    _run(
        tmp_path,
        """
        from dataclasses import dataclass, replace

        from verdog_runtime.declarations import (
            EdgeDefinition,
            Feature,
            FeatureDefinition,
            FeatureKind,
            FeatureNodeDefinition,
            FeatureState,
            FeatureSuccess,
            GraphDefinition,
            NodeDefinition,
            NumericalConditionObservation,
            NumericalEffectObservation,
            NumericalFeatureCondition,
            NumericalFeatureEffect,
            PortDefinition,
            Python,
            Success,
            VisitDefinition,
        )
        from verdog_runtime.declarations.ids import EdgeId, FeatureId, GraphId, NodeId, RunId
        from verdog_runtime.interpreter import (
            Dispatcher,
            EdgeExecution,
            ExecutionStatus,
            NodeExecution,
            initial_workflow_state,
            validate_graph,
        )


        @dataclass(frozen=True, slots=True)
        class Payload:
            number: int


        @dataclass(frozen=True, slots=True)
        class Runs:
            count: int = 0


        calls = 0
        mode = "normal"


        def ref(symbol):
            return globals()[symbol]


        initialized_with = []


        def initialize_remaining(input, state, context, /):
            global calls
            calls += 1
            initialized_with.append(("remaining", input))
            return FeatureSuccess(
                state=state.replace(REMAINING, True if mode == "invalid" else 1)
            )


        REMAINING = FeatureDefinition[int, object](
            id=FeatureId("remaining"),
            label="Remaining",
            description="Budget",
            kind=FeatureKind.INTEGER,
        )
        DEFERRED = FeatureDefinition[int, object](
            id=FeatureId("deferred"),
            label="Deferred",
            description="Initialized by a node",
            kind=FeatureKind.INTEGER,
        )


        ENTER = PortDefinition(id=NodeId("enter"))
        EXIT = PortDefinition(id=NodeId("exit"))
        FAILURE = PortDefinition(id=NodeId("failure"))


        def work_run(input, state, context, /):
            assert context.edge_id == EdgeId("enter_work")
            return Success(output=input, state=Runs(state.count + 1))


        def update_features(input, state, context, /):
            assert state.get(WORK) == Runs(1)
            remaining = state.get(REMAINING)
            assert remaining == 1
            try:
                state.replace(WORK, Runs(99))
            except TypeError:
                pass
            else:
                raise AssertionError("FeatureState replaced ordinary node state")
            if mode == "touch_other":
                raw = state._as_workflow_state()._replace(WORK, Runs(99))
                return FeatureSuccess(
                    state=FeatureState._from_workflow_state(raw)
                )
            return FeatureSuccess(
                state=state.replace(REMAINING, remaining - 1).replace(
                    DEFERRED, input.number
                )
            )


        INITIALIZE = FeatureNodeDefinition(
            id=NodeId("initialize"),
            name="Initialize remaining",
            operation=Feature(),
        )
        ENTER_INITIALIZE = EdgeDefinition(
            id=EdgeId("enter_initialize"),
            source=ENTER.id,
            target=INITIALIZE.id,
            visit=VisitDefinition(implementation=initialize_remaining),
        )
        WORK = NodeDefinition(
            id=NodeId("work"),
            name="Work",
            operation=Python(),
            state_type=Runs,
        )
        UPDATE = FeatureNodeDefinition(
            id=NodeId("update"),
            name="Update features",
            operation=Feature(),
        )
        ENTER_WORK = EdgeDefinition(
            id=EdgeId("enter_work"),
            source=INITIALIZE.id,
            target=WORK.id,
            effects=(
                NumericalFeatureEffect(
                    feature_id=REMAINING.id,
                    observation=NumericalEffectObservation.UNCONSTRAINED,
                ),
            ),
            visit=VisitDefinition(
                implementation=ref("work_run"),
            ),
        )
        WORK_UPDATE = EdgeDefinition(
            id=EdgeId("work_update"),
            source=WORK.id,
            target=UPDATE.id,
            visit=VisitDefinition(
                implementation=ref("update_features"),
            ),
        )
        UPDATE_EXIT = EdgeDefinition(
            id=EdgeId("update_exit"),
            source=UPDATE.id,
            target=EXIT.id,
            conditions=(
                NumericalFeatureCondition(
                    feature_id=REMAINING.id,
                    observation=NumericalConditionObservation.GREATER_ZERO,
                ),
            ),
            effects=(
                NumericalFeatureEffect(
                    feature_id=REMAINING.id,
                    observation=NumericalEffectObservation.DECREASES,
                ),
                NumericalFeatureEffect(
                    feature_id=DEFERRED.id,
                    observation=NumericalEffectObservation.UNCONSTRAINED,
                ),
            ),
        )
        GRAPH = GraphDefinition(
            id=GraphId("stateful"),
            params_type=type(None),
            enter=ENTER,
            exit=EXIT,
            failure=FAILURE,
            nodes=(INITIALIZE, WORK, UPDATE),
            edges=(ENTER_INITIALIZE, ENTER_WORK, WORK_UPDATE, UPDATE_EXIT),
            features=(REMAINING, DEFERRED),
        )

        validate_graph(GRAPH)
        try:
            validate_graph(
                replace(
                    GRAPH,
                    edges=(
                        ENTER_INITIALIZE,
                        replace(ENTER_WORK, visit=None),
                        WORK_UPDATE,
                        UPDATE_EXIT,
                    ),
                )
            )
        except ValueError as error:
            assert "must declare a visit" in str(error)
        else:
            raise AssertionError("an executable target accepted an edge without a visit")
        try:
            validate_graph(
                replace(
                    GRAPH,
                    edges=(
                        ENTER_INITIALIZE,
                        ENTER_WORK,
                        WORK_UPDATE,
                        replace(UPDATE_EXIT, visit=ENTER_WORK.visit),
                    ),
                )
            )
        except ValueError as error:
            assert "targeting a port must not declare a visit" in str(error)
        else:
            raise AssertionError("a terminal edge accepted a visit")
        initial = initial_workflow_state(GRAPH)
        assert initial.get(WORK) == Runs()
        assert initial.get(REMAINING) is None
        assert initial.get(DEFERRED) is None
        assert calls == 0
        assert initialized_with == []
        assert not hasattr(initial, "replace")
        assert not hasattr(UPDATE, "state_key")

        illegal_effect = replace(
            WORK_UPDATE,
            effects=(
                NumericalFeatureEffect(
                    feature_id=REMAINING.id,
                    observation=NumericalEffectObservation.UNCONSTRAINED,
                ),
            ),
        )
        try:
            validate_graph(
                replace(
                    GRAPH,
                    edges=(ENTER_INITIALIZE, ENTER_WORK, illegal_effect, UPDATE_EXIT),
                )
            )
        except ValueError as error:
            assert "source is not a feature node" in str(error)
        else:
            raise AssertionError("an ordinary-source edge carried an effect")

        original = Payload(7)
        events = []
        output = output_dir()
        result = (
            Dispatcher(execution_handler=events.append).run(
                definition(GRAPH),
                original,
                output_dir=output,
                run_id=RunId("run"),
            )
        )
        assert calls == 1
        assert initialized_with == [("remaining", original)]
        assert initialized_with[-1][1] is original
        assert isinstance(result, Success)
        assert result.output is original
        assert result.state.get(WORK) == Runs(1)
        assert result.state.get(REMAINING) == 0
        assert result.state.get(DEFERRED) == 7
        statistics = (output / "stats.md").read_text("utf-8")
        assert statistics.splitlines()[0] == "## Summary"
        rows = table_rows(output / "stats.md", "Nodes")
        for node_id, node_type in (("initialize", "feature"), ("work", "python"),
                                   ("update", "feature")):
            assert any(row[:5] == [".", str(GRAPH.id), node_id, node_type, "1"]
                       for row in rows)
        fired = next(
            event
            for event in events
            if isinstance(event, EdgeExecution)
            and event.edge_id == UPDATE_EXIT.id
        )
        assert fired.state == result.state
        assert [
            event.status
            for event in events
            if isinstance(event, NodeExecution) and event.node_id == WORK.id
        ] == [ExecutionStatus.RUNNING, ExecutionStatus.SUCCEEDED]

        def run_failure(graph, error_type, message, *, value=original, output=None,
                        failed_node=None):
            output = output_dir() if output is None else output
            caught = None
            failure_events = []
            try:
                (
                    Dispatcher(execution_handler=failure_events.append).run(
                        definition(graph), value, output_dir=output
                    )
                )
            except error_type as error:
                assert message in str(error)
                caught = error
            else:
                raise AssertionError(f"{error_type.__name__} was not raised")
            trace = trace_paths(output / "trace.log")
            assert trace[-1] == "failure/000001"
            stack = (output / trace[-1] / "stacktrace.txt").read_text("utf-8")
            assert caught is not None
            assert type(caught).__name__ in stack
            assert message in stack
            if failed_node is not None:
                assert [
                    (event.node_id, event.status, event.state)
                    for event in failure_events if isinstance(event, NodeExecution)
                ][-4:] == [
                    (failed_node.id, ExecutionStatus.RUNNING, None),
                    (failed_node.id, ExecutionStatus.FAILED, None),
                    (FAILURE.id, ExecutionStatus.RUNNING, None),
                    (FAILURE.id, ExecutionStatus.FAILED, None),
                ]
            return caught

        duplicate = replace(UPDATE_EXIT, id=EdgeId("duplicate"))
        ambiguous = replace(GRAPH, edges=(*GRAPH.edges, duplicate))
        run_failure(
            ambiguous,
            RuntimeError,
            "2 compatible outgoing edges; expected 1",
        )

        no_effect = replace(UPDATE_EXIT, effects=())
        run_failure(
            replace(
                GRAPH,
                edges=(ENTER_INITIALIZE, ENTER_WORK, WORK_UPDATE, no_effect),
            ),
            RuntimeError,
            "0 compatible outgoing edges; expected 1",
        )

        mode = "touch_other"
        run_failure(
            GRAPH,
            RuntimeError,
            "changed state ('node', 'work')",
            failed_node=UPDATE,
        )
        mode = "normal"
        assert not hasattr(EXIT, "state_key")

        mode = "invalid"
        run_failure(
            GRAPH,
            ValueError,
            "must be a non-negative int",
            failed_node=INITIALIZE,
        )
        mode = "normal"

        occupied = output_dir()
        occupied.mkdir(parents=True)
        marker = occupied / "trace"
        marker.write_text("keep\\n", encoding="utf-8")
        try:
            (
                Dispatcher().run(definition(GRAPH), original, output_dir=occupied)
            )
        except ValueError as error:
            assert "must be empty" in str(error)
        else:
            raise AssertionError("a non-empty output directory was accepted")
        assert marker.read_text("utf-8") == "keep\\n"

        output_file = output_dir()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("keep", encoding="utf-8")
        try:
            (
                Dispatcher().run(definition(GRAPH), original, output_dir=output_file)
            )
        except ValueError as error:
            assert "must be a directory" in str(error)
        else:
            raise AssertionError("an output file was accepted as a directory")
        assert output_file.read_text("utf-8") == "keep"

        same_id = WorkflowDefinition(
            id=GRAPH.id,
            input_type=Payload,
            entry=SubroutineCall(
                definition_id=GRAPH.id,
                definition_module=install_definition_module(
                    SubroutineDefinition(graph=GRAPH)
                ),
                params_types={(".", GRAPH.id): GRAPH.params_type},
                profile_arguments={},
                session_arguments={},
            ),
            configuration=WorkflowConfiguration(),
        )
        assert same_id.id == same_id.entry.definition_id
        """,
    )


def test_incoming_edges_select_distinct_visits_with_shared_node_state(
    tmp_path: Path,
) -> None:
    _run(
        tmp_path,
        """

        from verdog_runtime.declarations import (
            EdgeDefinition,
            Feature,
            FeatureDefinition,
            FeatureKind,
            FeatureNodeDefinition,
            FeatureSuccess,
            GraphDefinition,
            NodeDefinition,
            NumericalConditionObservation,
            NumericalEffectObservation,
            NumericalFeatureCondition,
            NumericalFeatureEffect,
            PortDefinition,
            Python,
            Success,
            VisitDefinition,
        )
        from verdog_runtime.declarations.ids import EdgeId, FeatureId, GraphId, NodeId
        from verdog_runtime.interpreter import Dispatcher


        def initialize(input, state, context, /):
            return FeatureSuccess(state=state.replace(REMAINING, 1))


        REMAINING = FeatureDefinition[int, object](
            id=FeatureId("remaining"),
            label="Remaining",
            description="One revisit",
            kind=FeatureKind.INTEGER,
        )


        def first(input, state, context, /):
            assert context.edge_id == EdgeId("enter_shared")
            assert state == SharedState()
            return Success(output=input, state=SharedState(("first",)))


        def second(input, state, context, /):
            assert context.edge_id == EdgeId("route_shared")
            assert state == SharedState(("first",))
            return Success(
                output=input,
                state=SharedState((*state.visits, "second")),
            )


        def decrement(input, state, context, /):
            remaining = state.get(REMAINING)
            return FeatureSuccess(
                state=state.replace(REMAINING, max(0, remaining - 1))
            )


        ENTER = PortDefinition(id=NodeId("enter"))
        EXIT = PortDefinition(id=NodeId("exit"))
        FAILURE = PortDefinition(id=NodeId("failure"))


        @dataclass(frozen=True, slots=True)
        class SharedState:
            visits: tuple[str, ...] = ()


        INITIALIZE = FeatureNodeDefinition(
            id=NodeId("initialize"), name="Initialize", operation=Feature(),
        )
        SHARED = NodeDefinition(
            id=NodeId("shared"),
            name="Shared",
            operation=Python(),
            state_type=SharedState,
        )
        ROUTE = FeatureNodeDefinition(
            id=NodeId("route"),
            name="Route",
            operation=Feature(),
        )
        GRAPH = GraphDefinition(
            id=GraphId("visits"),
            params_type=type(None),
            enter=ENTER,
            exit=EXIT,
            failure=FAILURE,
            nodes=(INITIALIZE, SHARED, ROUTE),
            edges=(
                EdgeDefinition(
                    id=EdgeId("enter_initialize"),
                    source=ENTER.id,
                    target=INITIALIZE.id,
                    visit=VisitDefinition(implementation=initialize),
                ),
                EdgeDefinition(
                    id=EdgeId("enter_shared"),
                    source=INITIALIZE.id,
                    target=SHARED.id,
                    effects=(
                        NumericalFeatureEffect(
                            feature_id=REMAINING.id,
                            observation=NumericalEffectObservation.UNCONSTRAINED,
                        ),
                    ),
                    visit=VisitDefinition(implementation=first),
                ),
                EdgeDefinition(
                    id=EdgeId("shared_route"),
                    source=SHARED.id,
                    target=ROUTE.id,
                    visit=VisitDefinition(implementation=decrement),
                ),
                EdgeDefinition(
                    id=EdgeId("route_shared"),
                    source=ROUTE.id,
                    target=SHARED.id,
                    conditions=(
                        NumericalFeatureCondition(
                            feature_id=REMAINING.id,
                            observation=NumericalConditionObservation.GREATER_ZERO,
                        ),
                    ),
                    effects=(
                        NumericalFeatureEffect(
                            feature_id=REMAINING.id,
                            observation=NumericalEffectObservation.DECREASES,
                        ),
                    ),
                    visit=VisitDefinition(implementation=second),
                ),
                EdgeDefinition(
                    id=EdgeId("route_exit"),
                    source=ROUTE.id,
                    target=EXIT.id,
                    conditions=(
                        NumericalFeatureCondition(
                            feature_id=REMAINING.id,
                            observation=NumericalConditionObservation.EQUAL_ZERO,
                        ),
                    ),
                ),
            ),
            features=(REMAINING,),
        )

        result = (
            Dispatcher().run(definition(GRAPH), 7, output_dir=output_dir())
        )
        assert result.output == 7
        assert result.state.get(SHARED) == SharedState(("first", "second"))
        """,
    )


def test_feature_saturation_uses_pre_state_conditions_and_effect_frame(
    tmp_path: Path,
) -> None:
    _run(
        tmp_path,
        """

        from verdog_runtime.declarations import (
            EdgeDefinition,
            Feature,
            FeatureDefinition,
            FeatureKind,
            FeatureNodeDefinition,
            FeatureSuccess,
            GraphDefinition,
            NumericalConditionObservation,
            NumericalEffectObservation,
            NumericalFeatureCondition,
            NumericalFeatureEffect,
            PortDefinition,
            Success,
            VisitDefinition,
        )
        from verdog_runtime.declarations.ids import EdgeId, FeatureId, GraphId, NodeId
        from verdog_runtime.interpreter import Dispatcher, EdgeExecution


        initial = 1


        BUDGET = FeatureDefinition[int, object](
            id=FeatureId("budget"),
            label="Budget",
            description="Saturating budget",
            kind=FeatureKind.INTEGER,
        )


        def initialize(input, state, context, /):
            return FeatureSuccess(state=state.replace(BUDGET, initial))


        def decrement(input, state, context, /):
            before = state.get(BUDGET)
            return FeatureSuccess(
                state=state.replace(BUDGET, max(0, before - 1))
            )


        ENTER = PortDefinition(id=NodeId("enter"))
        EXIT = PortDefinition(id=NodeId("exit"))
        FAILURE = PortDefinition(id=NodeId("failure"))
        INITIALIZE = FeatureNodeDefinition(
            id=NodeId("initialize"), name="Initialize", operation=Feature(),
        )
        DECREMENT = FeatureNodeDefinition(
            id=NodeId("decrement"),
            name="Decrement",
            operation=Feature(),
        )
        POSITIVE = EdgeDefinition(
            id=EdgeId("positive"),
            source=DECREMENT.id,
            target=EXIT.id,
            conditions=(
                NumericalFeatureCondition(
                    feature_id=BUDGET.id,
                    observation=NumericalConditionObservation.GREATER_ZERO,
                ),
            ),
            effects=(
                NumericalFeatureEffect(
                    feature_id=BUDGET.id,
                    observation=NumericalEffectObservation.DECREASES,
                ),
            ),
        )
        ZERO = EdgeDefinition(
            id=EdgeId("zero"),
            source=DECREMENT.id,
            target=EXIT.id,
            conditions=(
                NumericalFeatureCondition(
                    feature_id=BUDGET.id,
                    observation=NumericalConditionObservation.EQUAL_ZERO,
                ),
            ),
        )
        GRAPH = GraphDefinition(
            id=GraphId("saturating_feature"),
            params_type=type(None),
            enter=ENTER,
            exit=EXIT,
            failure=FAILURE,
            nodes=(INITIALIZE, DECREMENT),
            edges=(
                EdgeDefinition(
                    id=EdgeId("enter_initialize"),
                    source=ENTER.id,
                    target=INITIALIZE.id,
                    visit=VisitDefinition(implementation=initialize),
                ),
                EdgeDefinition(
                    id=EdgeId("enter_decrement"),
                    source=INITIALIZE.id,
                    target=DECREMENT.id,
                    effects=(
                        NumericalFeatureEffect(
                            feature_id=BUDGET.id,
                            observation=NumericalEffectObservation.UNCONSTRAINED,
                        ),
                    ),
                    visit=VisitDefinition(
                        implementation=decrement,
                    ),
                ),
                POSITIVE,
                ZERO,
            ),
            features=(BUDGET,),
        )


        def run(initial_value, expected_edge):
            global initial
            initial = initial_value
            events = []
            result = (
                Dispatcher(execution_handler=events.append).run(
                    definition(GRAPH), 7, output_dir=output_dir()
                )
            )
            assert isinstance(result, Success)
            assert result.output == 7
            assert result.state.get(BUDGET) == 0
            assert [
                event.edge_id
                for event in events
                if isinstance(event, EdgeExecution)
                and event.edge_id in (POSITIVE.id, ZERO.id)
            ] == [expected_edge]


        # Conditions observe sigma, while effects compare sigma with sigma-prime.
        run(1, POSITIVE.id)
        # With no explicit effect, the inferred frame requires the saturated value
        # to remain unchanged.
        run(0, ZERO.id)
        """,
    )


def test_explicit_failure_edge_raises_and_writes_native_stacktrace(
    tmp_path: Path,
) -> None:
    _run(
        tmp_path,
        """

        from verdog_runtime.declarations import (
            EdgeDefinition,
            GraphDefinition,
            NodeDefinition,
            PortDefinition,
            Python,
            Success,
            VisitDefinition,
        )
        from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId
        from verdog_runtime.interpreter import (
            Dispatcher,
            EdgeExecution,
            ExecutionStatus,
            NodeExecution,
        )


        def accept(*arguments):
            return True


        class FailurePayload:
            def __repr__(self):
                raise AssertionError("a failure-port value must be discarded")


        def fail(input, state, context, /):
            return Success(
                output=FailurePayload(),
                state=FailureState(state.count + 1),
            )


        ENTER = PortDefinition(id=NodeId("enter"))
        EXIT = PortDefinition(id=NodeId("exit"))
        FAILURE = PortDefinition(id=NodeId("failure"))


        @dataclass(frozen=True, slots=True)
        class FailureState:
            count: int = 0


        WORK = NodeDefinition(
            id=NodeId("work"),
            name="Work",
            state_type=FailureState,
            operation=Python(),
        )
        TO_WORK = EdgeDefinition(
            id=EdgeId("to_work"),
            source=ENTER.id,
            target=WORK.id,
            visit=VisitDefinition(implementation=fail),
        )
        TO_FAILURE = EdgeDefinition(
            id=EdgeId("to_failure"), source=WORK.id, target=FAILURE.id
        )
        GRAPH = GraphDefinition(
            id=GraphId("declared_failure"),
            params_type=type(None),
            enter=ENTER,
            exit=EXIT,
            failure=FAILURE,
            nodes=(WORK,),
            edges=(TO_WORK, TO_FAILURE),
        )

        output = output_dir()
        events = []
        caught = None
        try:
            (
                Dispatcher(execution_handler=events.append).run(
                    definition(GRAPH), 7, output_dir=output
                )
            )
        except RuntimeError as error:
            caught = error
        else:
            raise AssertionError("a failure edge returned normally")

        assert caught is not None
        assert str(caught) == "subroutine declared_failure reached its failure port"
        notes = "\\n".join(caught.__notes__)
        assert "graph=declared_failure source=work" in notes
        assert "graph=declared_failure edge=to_failure" in notes
        assert "failure=failure" in notes
        assert not hasattr(ENTER, "implementation")
        assert not hasattr(ENTER, "state_key")
        assert [
            (event.status, event.state)
            for event in events
            if isinstance(event, NodeExecution) and event.node_id == WORK.id
        ] == [
            (ExecutionStatus.RUNNING, FailureState()),
            (ExecutionStatus.SUCCEEDED, FailureState(1)),
        ]
        assert [
            event.status
            for event in events
            if isinstance(event, NodeExecution) and event.node_id == FAILURE.id
        ] == [ExecutionStatus.RUNNING, ExecutionStatus.FAILED]
        fired = next(
            event
            for event in events
            if isinstance(event, EdgeExecution) and event.edge_id == TO_FAILURE.id
        )
        assert fired.state.get(WORK) == FailureState(1)

        trace = trace_paths(output / "trace.log")
        assert trace == [
            "enter/000001",
            "work/000001",
            "failure/000001",
        ]
        stack = (output / trace[-1] / "stacktrace.txt").read_text("utf-8")
        assert "RuntimeError: subroutine declared_failure reached its failure port" in stack
        assert (output / "stats.md").is_file()
        assert "END failure/000001 status=failed" in (
            output / "trace.log"
        ).read_text("utf-8")
        assert "Verdog failure source:" in stack
        assert "Verdog failure edge:" in stack
        assert "Verdog failure boundary:" in stack
        """,
    )


def test_sequential_execution_nested_workflow_and_transition_limit(
    tmp_path: Path,
) -> None:
    _run(
        tmp_path,
        """
        from dataclasses import dataclass, replace

        from verdog_runtime.declarations import (
            Agent,
            AgentProfileParameter,
            AgentSessionParameter,
            CallVisitDefinition,
            EdgeDefinition,
            Feature,
            FeatureDefinition,
            FeatureKind,
            FeatureNodeDefinition,
            FeatureSuccess,
            GraphDefinition,
            NodeDefinition,
            NumericalEffectObservation,
            NumericalFeatureEffect,
            PortDefinition,
            Python,
            SubroutineCall,
            SubroutineDefinition,
            Success,
            VisitDefinition,
        )
        from verdog_runtime.declarations.ids import (
            AgentProfileId,
            AgentSessionId,
            EdgeId,
            FeatureId,
            GraphId,
            NodeId,
        )
        from verdog_runtime.interpreter import (
            Dispatcher,
            EdgeExecution,
            ExecutionStatus,
            NodeExecution,
            validate_graph,
        )


        @dataclass(frozen=True, slots=True)
        class Runs:
            count: int = 0


        @dataclass(frozen=True, slots=True)
        class AdapterState:
            invocations: int = 0
            child_output: int | None = None


        def ref(symbol):
            return globals()[symbol]


        BRANCH_COUNT = FeatureDefinition[int, object](
            id=FeatureId("branch_count"),
            label="Branch count",
            description="Branch-local feature",
            kind=FeatureKind.INTEGER,
        )


        def worker_a(input, state, context, /):
            return Success(
                output=input + 1,
                state=Runs(state.count + 1),
            )


        def increment_count(input, state, context, /):
            count = state.get(BRANCH_COUNT)
            assert count is None
            return FeatureSuccess(state=state.replace(BRANCH_COUNT, 1))


        def worker_b(input, state, context, /):
            return Success(
                output=input + 2,
                state=Runs(state.count + 1),
            )


        def collect(input, state, context, /):
            return Success(
                output=input,
                state=Runs(state.count + 1),
            )


        ENTER = PortDefinition(id=NodeId("enter"))
        EXIT = PortDefinition(id=NodeId("exit"))
        FAILURE = PortDefinition(id=NodeId("failure"))
        WORKER_A = NodeDefinition(
            id=NodeId("worker_a"),
            name="A",
            state_type=Runs,
            operation=Python(),
        )
        WORKER_B = NodeDefinition(
            id=NodeId("worker_b"),
            name="B",
            state_type=Runs,
            operation=Python(),
        )
        INCREMENT = FeatureNodeDefinition(
            id=NodeId("increment"),
            name="Increment branch count",
            operation=Feature(),
        )
        COLLECT = NodeDefinition(
            id=NodeId("collect"),
            name="Collect",
            state_type=Runs,
            operation=Python(),
        )


        def edge(
            edge_id,
            source,
            target,
            implementation=None,
            *,
            call=False,
            **kwargs,
        ):
            return EdgeDefinition(
                id=EdgeId(edge_id),
                source=source.id,
                target=target.id,
                visit=(
                    None
                    if implementation is None
                    else (
                        CallVisitDefinition(
                            implementation=ref(implementation),
                        )
                        if call
                        else VisitDefinition(
                            implementation=ref(implementation),
                        )
                    )
                ),
                **kwargs,
            )


        EDGES = (
            edge("enter_a", ENTER, WORKER_A, "worker_a"),
            edge("a_increment", WORKER_A, INCREMENT, "increment_count"),
            edge(
                "increment_b",
                INCREMENT,
                WORKER_B,
                "worker_b",
                effects=(
                    NumericalFeatureEffect(
                        feature_id=BRANCH_COUNT.id,
                        observation=NumericalEffectObservation.UNCONSTRAINED,
                    ),
                ),
            ),
            edge("b_collect", WORKER_B, COLLECT, "collect"),
            edge("collect_exit", COLLECT, EXIT),
        )
        GRAPH = GraphDefinition(
            id=GraphId("sequence"),
            params_type=type(None),
            enter=ENTER,
            exit=EXIT,
            failure=FAILURE,
            nodes=(WORKER_A, INCREMENT, WORKER_B, COLLECT),
            edges=EDGES,
            features=(BRANCH_COUNT,),
        )
        # Two agent nodes may share explicit profile and session resources.
        def agent(node, profile="default"):
            return replace(
                node,
                operation=Agent(
                    profile=AgentProfileId(profile),
                    session=AgentSessionId("shared"),
                ),
            )

        AGENT_GRAPH = replace(
            GRAPH,
            nodes=(
                agent(WORKER_A),
                INCREMENT,
                agent(WORKER_B),
                COLLECT,
            ),
            profile_parameters=(
                AgentProfileParameter(
                    id=AgentProfileId("default"), name="Default"
                ),
            ),
            session_parameters=(
                AgentSessionParameter(
                    id=AgentSessionId("shared"),
                    name="Shared",
                ),
            )
        )
        validate_graph(AGENT_GRAPH)
        try:
            validate_graph(
                replace(
                    AGENT_GRAPH,
                    nodes=(
                        agent(WORKER_A),
                        INCREMENT,
                        agent(WORKER_B, "other"),
                        COLLECT,
                    ),
                )
            )
        except ValueError as error:
            assert "references unknown profile: other" in str(error)
        else:
            raise AssertionError("an undeclared profile was accepted")

        validate_graph(GRAPH)
        events = []
        result = (
            Dispatcher(execution_handler=events.append).run(
                definition(GRAPH), 10, output_dir=output_dir()
            )
        )
        assert isinstance(result, Success)
        # 10 -> worker_a adds one -> worker_b adds two -> collect passes it through.
        assert result.output == 13
        # Sequential, so every node's state and the feature all persist into the final
        # workflow state. The old assertions here said the opposite -- they existed
        # because a fork's branch states were never merged back.
        assert result.state.get(BRANCH_COUNT) == 1
        assert result.state.get(WORKER_A) == Runs(1)
        assert result.state.get(WORKER_B) == Runs(1)
        assert result.state.get(COLLECT) == Runs(1)
        for edge_id in ("enter_a", "a_increment", "increment_b", "b_collect"):
            assert any(
                isinstance(event, EdgeExecution)
                and event.edge_id == EdgeId(edge_id)
                for event in events
            )

        CHILD_ENTER = PortDefinition(id=NodeId("child_enter"))
        CHILD_EXIT = PortDefinition(id=NodeId("child_exit"))
        CHILD_FAILURE = PortDefinition(id=NodeId("child_failure"))


        class ChildProblem(LookupError):
            pass


        child_mode = "normal"
        raised_errors = []
        raised_causes = []


        def child_run(input, state, context, /):
            if child_mode == "fail":
                cause = ValueError("underlying child cause")
                error = ChildProblem("child implementation broke")
                raised_causes.append(cause)
                raised_errors.append(error)
                try:
                    raise cause
                except ValueError as caught:
                    raise error from caught
            return Success(
                output=input + 1,
                state=Runs(state.count + 1),
            )


        CHILD_WORK = NodeDefinition(
            id=NodeId("child_work"),
            name="Child work",
            state_type=Runs,
            operation=Python(),
        )
        CHILD = GraphDefinition(
            id=GraphId("parent__child"),
            params_type=type(None),
            enter=CHILD_ENTER,
            exit=CHILD_EXIT,
            failure=CHILD_FAILURE,
            nodes=(CHILD_WORK,),
            edges=(
                edge("child_in", CHILD_ENTER, CHILD_WORK, "child_run"),
                edge("child_out", CHILD_WORK, CHILD_EXIT),
            ),
        )
        CHILD_MODULE = install_definition_module(SubroutineDefinition(graph=CHILD))


        adapter_mode = "normal"
        recovered_errors = []


        def workflow_adapter(input, state, context, /):
            if adapter_mode == "no_invoke":
                return Success(output="not invoked", state=state)
            if adapter_mode == "catch":
                try:
                    context.invoke(input)
                except ChildProblem as error:
                    recovered_errors.append(error)
                    return Success(
                        output="recovered",
                        state=AdapterState(invocations=state.invocations + 1),
                    )
                raise AssertionError("the child failure was not raised")
            child_output = context.invoke(input)
            if adapter_mode == "double_invoke":
                context.invoke(input)
            assert isinstance(child_output, int)
            if adapter_mode == "invalid_state_type":
                return Success(
                    output=f"child={child_output}",
                    state="not adapter state",
                )
            return Success(
                output=f"child={child_output}",
                state=AdapterState(
                    invocations=state.invocations + 1,
                    child_output=child_output,
                ),
            )


        PARENT_ENTER = PortDefinition(id=NodeId("parent_enter"))
        PARENT_EXIT = PortDefinition(id=NodeId("parent_exit"))
        PARENT_FAILURE = PortDefinition(id=NodeId("parent_failure"))
        CHILD_NODE = NodeDefinition(
            id=NodeId("child_node"),
            name="Child adapter",
            state_type=AdapterState,
            operation=SubroutineCall(
                definition_id=CHILD.id,
                definition_module=CHILD_MODULE,
                params_types={(".", CHILD.id): type(None)},
                profile_arguments={},
                session_arguments={},
            ),
        )
        PARENT = GraphDefinition(
            id=GraphId("parent"),
            params_type=type(None),
            enter=PARENT_ENTER,
            exit=PARENT_EXIT,
            failure=PARENT_FAILURE,
            nodes=(CHILD_NODE,),
            edges=(
                edge(
                    "parent_in",
                    PARENT_ENTER,
                    CHILD_NODE,
                    "workflow_adapter",
                    call=True,
                ),
                edge("parent_out", CHILD_NODE, PARENT_EXIT),
            ),
        )
        validate_graph(PARENT)
        nested_events = []
        nested_output = output_dir()
        result = (
            Dispatcher(execution_handler=nested_events.append).run(
                definition(PARENT), 4, output_dir=nested_output
            )
        )
        assert isinstance(result, Success)
        assert result.output == "child=5"
        assert result.state.get(CHILD_NODE) == AdapterState(1, 5)
        rows = table_rows(nested_output / "stats.md", "Nodes")
        assert {row[1] for row in rows} == {str(PARENT.id)}
        assert any(row[1:5] == [str(PARENT.id), str(CHILD_NODE.id), "subroutine_call", "1"]
                   for row in rows)
        child_output = nested_output / "child_node/000001"
        rows = table_rows(child_output / "stats.md", "Nodes")
        assert {row[1] for row in rows} == {str(CHILD.id)}
        assert any(row[1:5] == [str(CHILD.id), str(CHILD_WORK.id), "python", "1"]
                   for row in rows)
        for name in ("config.md", "stats.md"):
            assert call_reports(nested_output / name) == [(child_output / name).resolve()]
            assert call_reports(child_output / name) == []
            assert len(list(nested_output.rglob(name))) == 2
        try:
            result.state.get(CHILD_WORK)
        except KeyError:
            pass
        else:
            raise AssertionError("child state leaked into the parent universe")
        assert any(
            isinstance(event, NodeExecution) and event.graph_id == CHILD.id
            for event in nested_events
        )

        adapter_mode = "catch"
        child_mode = "fail"
        recovery_output = output_dir()
        recovery_events = []
        result = (
            Dispatcher(execution_handler=recovery_events.append).run(
                definition(PARENT), 4, output_dir=recovery_output
            )
        )
        assert isinstance(result, Success)
        assert result.output == "recovered"
        assert result.state.get(CHILD_NODE) == AdapterState(1, None)
        assert recovered_errors[-1] is raised_errors[-1]
        assert recovered_errors[-1].__cause__ is raised_causes[-1]
        recovery_trace = trace_paths(recovery_output / "trace.log")
        child_failures = [
            path for path in recovery_trace if path.endswith("/child_failure/000001")
        ]
        assert len(child_failures) == 1
        assert "parent_failure/000001" not in recovery_trace
        child_stack = (
            recovery_output / child_failures[0] / "stacktrace.txt"
        ).read_text("utf-8")
        assert "ValueError: underlying child cause" in child_stack
        assert "ChildProblem: child implementation broke" in child_stack
        assert [
            event.status
            for event in recovery_events
            if isinstance(event, NodeExecution) and event.node_id == CHILD_FAILURE.id
        ] == [ExecutionStatus.RUNNING, ExecutionStatus.FAILED]
        assert [
            event.status
            for event in recovery_events
            if isinstance(event, NodeExecution) and event.node_id == CHILD_NODE.id
        ] == [ExecutionStatus.RUNNING, ExecutionStatus.SUCCEEDED]
        child_events = [
            event for event in recovery_events
            if isinstance(event, NodeExecution) and event.graph_id == CHILD.id
        ]
        assert [
            (event.node_id, event.status, event.state) for event in child_events[-4:]
        ] == [
            (CHILD_WORK.id, ExecutionStatus.RUNNING, Runs()),
            (CHILD_WORK.id, ExecutionStatus.FAILED, Runs()),
            (CHILD_FAILURE.id, ExecutionStatus.RUNNING, None),
            (CHILD_FAILURE.id, ExecutionStatus.FAILED, None),
        ]
        assert child_events[-4].state is child_events[-3].state
        child_output = recovery_output / "child_node/000001"
        for name in ("config.md", "stats.md"):
            assert call_reports(recovery_output / name) == [(child_output / name).resolve()]
            assert call_reports(child_output / name) == []
        assert any(row[2:5] == ["child_failure", "failure", "1"]
                   for row in table_rows(child_output / "stats.md", "Nodes"))

        def run_error(
            graph,
            error_type,
            message,
            *,
            value=4,
            dispatcher=None,
        ):
            output = output_dir()
            caught = None
            try:
                (
                    (dispatcher or Dispatcher()).run(
                        definition(graph), value, output_dir=output
                    )
                )
            except error_type as error:
                assert message in str(error)
                caught = error
            else:
                raise AssertionError(f"{error_type.__name__} was not raised")
            trace = trace_paths(output / "trace.log")
            stack = (output / trace[-1] / "stacktrace.txt").read_text("utf-8")
            assert message in stack
            assert caught is not None
            return caught, output, trace


        adapter_mode = "normal"
        propagated, propagated_output, propagated_trace = run_error(
            PARENT,
            ChildProblem,
            "child implementation broke",
        )
        assert propagated is raised_errors[-1]
        assert propagated.__cause__ is raised_causes[-1]
        assert propagated.__context__ is raised_causes[-1]
        names = []
        traceback_cursor = propagated.__traceback__
        while traceback_cursor is not None:
            names.append(traceback_cursor.tb_frame.f_code.co_name)
            traceback_cursor = traceback_cursor.tb_next
        assert "child_run" in names
        notes = "\\n".join(propagated.__notes__)
        assert "graph=parent__child node=child_work edge=child_in" in notes
        assert "graph=parent__child failure=child_failure" in notes
        assert "graph=parent node=child_node edge=parent_in" in notes
        assert "graph=parent failure=parent_failure" in notes
        failure_paths = [
            path
            for path in propagated_trace
            if path.endswith(
                ("/child_failure/000001", "parent_failure/000001")
            )
        ]
        assert len(failure_paths) == 2
        for path in failure_paths:
            assert (propagated_output / path / "stacktrace.txt").is_file()

        child_mode = "normal"
        adapter_mode = "no_invoke"
        run_error(
            PARENT,
            RuntimeError,
            "a call node must invoke its child exactly once",
        )

        adapter_mode = "double_invoke"
        run_error(
            PARENT,
            RuntimeError,
            "a call node must invoke its child exactly once",
        )

        adapter_mode = "invalid_state_type"
        run_error(PARENT, RuntimeError, "entity state has the wrong type")
        adapter_mode = "normal"


        def loop_run(input, state, context, /):
            return Success(output=input, state=state)


        LOOP_ENTER = PortDefinition(id=NodeId("loop_enter"))
        LOOP_EXIT = PortDefinition(id=NodeId("loop_exit"))
        LOOP_FAILURE = PortDefinition(id=NodeId("loop_failure"))
        LOOP = NodeDefinition(
            id=NodeId("loop"),
            name="Loop",
            state_type=EmptyState,
            operation=Python(),
        )
        LOOP_GRAPH = GraphDefinition(
            id=GraphId("loop_graph"),
            params_type=type(None),
            enter=LOOP_ENTER,
            exit=LOOP_EXIT,
            failure=LOOP_FAILURE,
            nodes=(LOOP,),
            edges=(
                edge("loop_in", LOOP_ENTER, LOOP, "loop_run"),
                edge("loop_again", LOOP, LOOP, "loop_run"),
            ),
        )
        loop_output = output_dir()
        try:
            (
                Dispatcher(transition_limit=2).run(
                    definition(LOOP_GRAPH), 1, output_dir=loop_output
                )
            )
        except RuntimeError as error:
            assert "workflow transition limit exceeded" in str(error)
            assert error.__notes__[0] == "Verdog edge: loop_again"
        else:
            raise AssertionError("the transition limit was not enforced")
        loop_trace = trace_paths(loop_output / "trace.log")
        assert loop_trace == [
            "loop_enter/000001",
            "loop/000001",
            "loop/000002",
            "loop_failure/000001",
        ]
        assert (loop_output / loop_trace[-1] / "stacktrace.txt").is_file()
        """,
    )


def test_an_operation_the_runtime_does_not_know_is_refused_rather_than_run(
    tmp_path: Path,
) -> None:
    """Workflow calls used to be the unguarded fall-through of the dispatch chain.

    So an operation the chain did not recognise was executed as a child-workflow call:
    handed a `CallNodeContext` it never asked for, and then judged against the "must
    invoke its child exactly once" rule -- which means the failure that surfaced described
    a workflow node the author had not written. `validate_graph` does not catch it either;
    it only constrains the operations it knows. This is the one place that can, and the
    exception it raises names the operation.
    """

    _run(
        tmp_path,
        """
        from dataclasses import dataclass

        from verdog_runtime.declarations import (
            EdgeDefinition,
            GraphDefinition,
            NodeDefinition,
            PortDefinition,
            VisitDefinition,
        )
        from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId
        from verdog_runtime.interpreter import (
            Dispatcher,
            ExecutionStatus,
            NodeExecution,
            validate_graph,
        )


        @dataclass(frozen=True, slots=True)
        class Bespoke:
            \"\"\"Shaped like an operation, and not one of the four.\"\"\"

            pass


        def never_called(input, state, context, /):
            raise AssertionError("an unknown operation must not be invoked")


        ENTER = PortDefinition(id=NodeId("enter"))
        EXIT = PortDefinition(id=NodeId("exit"))
        FAILURE = PortDefinition(id=NodeId("failure"))
        BESPOKE = NodeDefinition(
            id=NodeId("bespoke"),
            name="Bespoke",
            state_type=EmptyState,
            operation=Bespoke(),
        )
        GRAPH = GraphDefinition(
            id=GraphId("unknown_operation"),
            params_type=type(None),
            enter=ENTER,
            exit=EXIT,
            failure=FAILURE,
            nodes=(BESPOKE,),
            edges=(
                EdgeDefinition(
                    id=EdgeId("enter_bespoke"),
                    source=ENTER.id,
                    target=BESPOKE.id,
                    visit=VisitDefinition(
                        implementation=never_called,
                    ),
                ),
                EdgeDefinition(id=EdgeId("bespoke_exit"), source=BESPOKE.id, target=EXIT.id),
            ),
            features=(),
        )

        # Graph validation is silent about it, which is why the guard has to be in the
        # interpreter rather than here.
        validate_graph(GRAPH)

        output = output_dir()
        events = []
        try:
            (
                Dispatcher(execution_handler=events.append).run(
                    definition(GRAPH), 1, output_dir=output
                )
            )
        except RuntimeError as error:
            assert "Bespoke is not an operation this runtime executes" in str(error)
            assert "[unsupported_operation]" in str(error)
            assert any("Verdog entity: bespoke" in note for note in error.__notes__)
        else:
            raise AssertionError("an unknown operation was accepted")
        assert [
            event.status
            for event in events
            if isinstance(event, NodeExecution) and event.node_id == BESPOKE.id
        ] == [ExecutionStatus.RUNNING, ExecutionStatus.FAILED]
        trace = trace_paths(output / "trace.log")
        assert trace[-1] == "failure/000001"
        stack = (output / trace[-1] / "stacktrace.txt").read_text("utf-8")
        assert "RuntimeError: Bespoke is not an operation" in stack
        """,
    )
