"""Load generated workflow declarations with verified identities."""

from __future__ import annotations

import importlib
import pathlib
import sys
import types
from collections.abc import Callable
from typing import TypeVar

from verdog_runtime.declarations import ids

DefinitionT = TypeVar("DefinitionT")


def load_definition(
    project_root: pathlib.Path,
    module: str,
    definition_id: ids.GraphId,
    definition_type: type[DefinitionT],
    identity: Callable[[DefinitionT], ids.GraphId],
    /,
) -> tuple[DefinitionT, types.ModuleType]:
    owner = project_root.resolve()
    source = (owner / "src").resolve()
    if not source.is_relative_to(owner):
        raise ValueError("project source root escapes its project")
    for path in reversed((str(source), str(owner))):
        if path not in sys.path:
            sys.path.insert(0, path)

    declaration = importlib.import_module(module)
    factory: object = getattr(declaration, "definition", None)
    if not callable(factory):
        raise TypeError(f"{module}.definition is not callable")
    definition: object = factory()
    if not isinstance(definition, definition_type):
        raise TypeError(
            f"{module}.definition() is not {definition_type.__name__}"
        )
    actual_id = identity(definition)
    if actual_id != definition_id:
        raise ValueError(
            f"definition id does not match: {module}.definition() has id "
            f"{actual_id!s}, expected {definition_id}"
        )
    return definition, declaration
