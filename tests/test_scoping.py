from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from verdog_runtime.child import (
    _local_definition_id,  # pyright: ignore[reportPrivateUsage]
)
from verdog_runtime.declarations import (
    EdgeDefinition,
    GraphDefinition,
    PortDefinition,
    SubroutineCall,
    SubroutineDefinition,
    WorkflowCall,
    WorkflowConfiguration,
    WorkflowDefinition,
)
from verdog_runtime.declarations.ids import (
    EdgeId,
    GraphId,
    NodeId,
    is_valid_definition_id,
    is_valid_entity_id,
)
from verdog_runtime.interpreter._calls import (
    CallScope,
    local_subroutine,
    require_local_workflow,
    subroutine_scope,
    workflow_scope,
)


def test_definition_identifiers_use_ascii_non_keyword_leaves() -> None:
    for value in ("main", "child_2", "main__child_2"):
        assert is_valid_definition_id(value)
    for value in (
        "",
        "2child",
        "Child",
        "chíld",
        "child_",
        "child___nested",
        "main__case",
        "main__for",
        "main__match",
        "main__type",
    ):
        assert not is_valid_definition_id(value)
    assert is_valid_entity_id("child_2")
    assert not is_valid_entity_id("child__nested")


def _definition(
    graph_id: str,
) -> SubroutineDefinition[object, object, None, object]:
    enter = PortDefinition(id=NodeId("enter"))
    exit_ = PortDefinition(id=NodeId("exit"))
    return SubroutineDefinition(
        graph=GraphDefinition(
            id=GraphId(graph_id),
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
        )
    )


def _module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    definition: SubroutineDefinition[object, object, None, object],
) -> None:
    module = ModuleType(name)
    module.__dict__["definition"] = lambda: definition
    monkeypatch.setitem(sys.modules, name, module)


def _subroutine_call(definition_id: str, module: str) -> SubroutineCall:
    return SubroutineCall(
        definition_id=GraphId(definition_id),
        definition_module=module,
        params_types={},
        profile_arguments={},
        session_arguments={},
    )


def _workflow_call(definition_id: str, module: str) -> WorkflowCall:
    return WorkflowCall(
        definition_id=GraphId(definition_id), definition_module=module
    )


def test_calls_import_exact_modules_and_enforce_lexical_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules = {
        "scoped.main__child": "scoped.subroutines.main.subroutines.child",
        "scoped.main__child__grandchild": (
            "scoped.subroutines.main.subroutines.child.subroutines.grandchild"
        ),
        "scoped.main__sibling": "scoped.subroutines.main.subroutines.sibling",
    }
    definitions = {name: _definition(name) for name in modules}
    for name, module in modules.items():
        _module(monkeypatch, module, definitions[name])

    scope = CallScope(
        GraphId("scoped.main"), GraphId("scoped.main"), GraphId("scoped.main")
    )
    child, scope = local_subroutine(
        tmp_path,
        scope,
        _subroutine_call("scoped.main__child", modules["scoped.main__child"]),
    )
    assert child is definitions["scoped.main__child"]
    grandchild, scope = local_subroutine(
        tmp_path,
        scope,
        _subroutine_call(
            "scoped.main__child__grandchild",
            modules["scoped.main__child__grandchild"],
        ),
    )
    assert grandchild is definitions["scoped.main__child__grandchild"]
    sibling, sibling_scope = local_subroutine(
        tmp_path,
        scope,
        _subroutine_call(
            "scoped.main__sibling", modules["scoped.main__sibling"]
        ),
    )
    assert sibling is definitions["scoped.main__sibling"]
    with pytest.raises(LookupError, match="not lexically visible"):
        local_subroutine(
            tmp_path,
            sibling_scope,
            _subroutine_call(
                "scoped.main__child__grandchild",
                modules["scoped.main__child__grandchild"],
            ),
        )

    require_local_workflow(
        scope,
        _workflow_call(
            "scoped.main__child__review",
            "scoped.subroutines.main.subroutines.child.workflows.review",
        ),
    )
    with pytest.raises(LookupError, match="not lexically visible"):
        require_local_workflow(
            scope,
            _workflow_call(
                "scoped.main__sibling__review",
                "scoped.subroutines.main.subroutines.sibling.workflows.review",
            ),
        )


def test_external_subroutine_needs_no_project_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _definition("scoped.main__child")
    module = "scoped.subroutines.main.subroutines.child"
    _module(monkeypatch, module, definition)
    scope, loaded = subroutine_scope(
        tmp_path, _subroutine_call("scoped.main__child", module)
    )
    assert loaded is definition
    assert scope == CallScope(
        GraphId("scoped.main__child"), GraphId("scoped.main")
    )
    assert not (tmp_path / "project.json").exists()


def test_workflow_scope_is_derived_from_the_loaded_definition() -> None:
    subroutine = _definition("scoped.main__child")
    workflow: WorkflowDefinition[object, object, None, object] = (
        WorkflowDefinition(
            id=GraphId("scoped.main__child"),
            input_type=object,
            entry=_subroutine_call(
                str(subroutine.graph.id),
                "scoped.subroutines.main.subroutines.child",
            ),
            configuration=WorkflowConfiguration(),
        )
    )
    assert workflow_scope(workflow) == CallScope(
        GraphId("scoped.main__child"),
        GraphId("scoped.main"),
        GraphId("scoped.main__child"),
    )


def test_child_environment_requires_matching_module_and_definition() -> None:
    assert (
        _local_definition_id(
            GraphId("scoped.main__child_2"),
            "scoped.subroutines.main.subroutines.child_2",
        )
        == "main__child_2"
    )
    for definition_id, module in (
        ("scoped.main__match", "scoped.subroutines.main"),
        ("scoped.main__chíld", "scoped.subroutines.main"),
        ("other.main", "scoped.workflows.main"),
    ):
        with pytest.raises(ValueError):
            _local_definition_id(GraphId(definition_id), module)


@pytest.mark.parametrize(
    "module", ["", ".relative", "scope..child", "scope.Child", "scope.match"]
)
def test_call_modules_are_absolute_python_modules(module: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        _workflow_call("scoped.main", module)
