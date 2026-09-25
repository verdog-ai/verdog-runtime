import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import assert_type

import pytest

from verdog_runtime._definitions import load_definition
from verdog_runtime.declarations.ids import GraphId


@dataclass(frozen=True, slots=True)
class LoadedDefinition:
    identifier: GraphId


def test_definition_loader_retains_type_and_checks_before_reading_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration = ModuleType("runtime_test_definition_loader")
    expected = LoadedDefinition(identifier=GraphId("loaded"))
    declaration.__dict__["definition"] = lambda: expected
    monkeypatch.setitem(sys.modules, declaration.__name__, declaration)
    monkeypatch.setattr(sys, "path", list(sys.path))
    identities: list[LoadedDefinition] = []

    def identity(definition: LoadedDefinition) -> GraphId:
        identities.append(definition)
        return definition.identifier

    loaded, module = load_definition(
        tmp_path,
        declaration.__name__,
        expected.identifier,
        LoadedDefinition,
        identity,
    )
    assert_type(loaded, LoadedDefinition)
    assert loaded is expected
    assert module is declaration
    assert identities == [expected]

    with pytest.raises(ValueError, match="definition id does not match"):
        load_definition(
            tmp_path,
            declaration.__name__,
            GraphId("other"),
            LoadedDefinition,
            identity,
        )
    assert identities == [expected, expected]

    identities.clear()
    declaration.__dict__["definition"] = lambda: object()
    with pytest.raises(
        TypeError, match=r"definition\(\) is not LoadedDefinition"
    ):
        load_definition(
            tmp_path,
            declaration.__name__,
            expected.identifier,
            LoadedDefinition,
            identity,
        )
    assert identities == []

    declaration.__dict__["definition"] = None
    with pytest.raises(TypeError, match="definition is not callable"):
        load_definition(
            tmp_path,
            declaration.__name__,
            expected.identifier,
            LoadedDefinition,
            identity,
        )
    assert identities == []
