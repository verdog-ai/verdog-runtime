from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import ModuleType, NoneType

import pytest

from verdog_runtime.cli import WorkflowArguments, parse_arguments
from verdog_runtime.declarations import (
    EdgeDefinition,
    GraphDefinition,
    PortDefinition,
    SubroutineCall,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import EdgeId, GraphId, NodeId
from verdog_runtime.entry import (
    _arguments,  # pyright: ignore[reportPrivateUsage]
)


class Backend(StrEnum):
    CODEX = "codex"
    CLAUDE = "claude"


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorOptions:
    iterations: int = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class Input:
    domain: str
    tasks: tuple[Path, ...]
    generator: GeneratorOptions = field(default_factory=GeneratorOptions)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeOptions:
    backend: Backend = Backend.CODEX
    model: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Params:
    iterations: int = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildParams:
    attempts: int = 2


def test_typed_arguments_are_namespaced_by_role_and_definition() -> None:
    parsed = parse_arguments(
        Input,
        (
            "--input.domain",
            "blocks",
            "--input.tasks",
            "one.pddl",
            "two.pddl",
            "--input.generator.iterations",
            "8",
            "--params.main.iterations",
            "7",
            "--params.verdog-ai.child.worker.attempts",
            "4",
            "--runtime.backend",
            "claude",
            "--runtime.model",
            "sonnet",
        ),
        params_types={
            (".", GraphId("sample.main")): Params,
            (
                "external/verdog-ai/child",
                GraphId("sample.worker"),
            ): ChildParams,
        },
        runtime_options=RuntimeOptions,
        prog="example",
    )

    assert parsed == WorkflowArguments(
        input=Input(
            domain="blocks",
            tasks=(Path("one.pddl"), Path("two.pddl")),
            generator=GeneratorOptions(iterations=8),
        ),
        runtime=RuntimeOptions(backend=Backend.CLAUDE, model="sonnet"),
        params={
            (".", GraphId("sample.main")): Params(iterations=7),
            (
                "external/verdog-ai/child",
                GraphId("sample.worker"),
            ): ChildParams(attempts=4),
        },
    )


def test_workflow_declaration_owns_runtime_options_and_configuration() -> None:
    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    graph: GraphDefinition[object, object, object, object] = GraphDefinition(
        id=GraphId("sample.main_body"),
        params_type=NoneType,
        enter=enter,
        exit=exit_,
        failure=PortDefinition(id=NodeId("failure")),
        nodes=(),
        edges=(
            EdgeDefinition(
                id=EdgeId("enter_exit"),
                source=enter.id,
                target=exit_.id,
            ),
        ),
    )
    definition: WorkflowDefinition[object, object, object, object] = (
        WorkflowDefinition(
            id=GraphId("sample.main"),
            input_type=NoneType,
            entry=SubroutineCall(
                definition_id=graph.id,
                definition_module="sample.subroutines.main_body",
                params_types={(".", graph.id): graph.params_type},
                profile_arguments={},
                session_arguments={},
            ),
            configuration=WorkflowConfiguration(),
        )
    )
    declaration = ModuleType("sample.workflows.main")
    configured = WorkflowConfiguration()

    def configure(options: RuntimeOptions, /) -> WorkflowConfiguration:
        assert options == RuntimeOptions(backend=Backend.CLAUDE, model="sonnet")
        return configured

    declaration.__dict__["RuntimeOptions"] = RuntimeOptions
    declaration.__dict__["configure"] = configure

    selected, parsed = _arguments(
        definition,
        declaration,
        ("--runtime.backend", "claude", "--runtime.model", "sonnet"),
    )

    assert selected.configuration is configured
    assert parsed.input is None
    assert parsed.params == {}
    assert parsed.runtime == RuntimeOptions(
        backend=Backend.CLAUDE, model="sonnet"
    )

    with pytest.raises(RuntimeError, match="RuntimeOptions must be a type"):
        _arguments(definition, ModuleType("sample.workflows.missing"), ())


def test_object_input_is_the_parameterless_workflow_interface() -> None:
    assert type(parse_arguments(object, ()).input) is object
    assert parse_arguments(NoneType, ()).input is None
    assert (
        type(parse_arguments(NoneType, (), runtime_options=object).runtime)
        is object
    )
    assert (
        parse_arguments(NoneType, (), runtime_options=NoneType).runtime is None
    )
    with pytest.raises(SystemExit) as stopped:
        parse_arguments(object, ("--help",), prog="empty")
    assert stopped.value.code == 0


def test_parameter_defaults_and_cli_address_collisions() -> None:
    parsed = parse_arguments(
        NoneType,
        (),
        params_types={(".", GraphId("sample.main")): Params},
    )
    assert parsed.params == {
        (".", GraphId("sample.main")): Params(),
    }
    assert (
        parse_arguments(
            NoneType,
            (),
            params_types={
                (".", GraphId("sample.empty")): NoneType,
                (".", GraphId("sample.opaque")): object,
            },
        ).params
        == {}
    )

    with pytest.raises(
        ValueError, match="duplicate parameter command-line address"
    ):
        parse_arguments(
            NoneType,
            (),
            params_types={
                (".", GraphId("one.main")): Params,
                (".", GraphId("two.main")): Params,
            },
        )
