"""Identifiers in the workflow declaration language."""

import keyword
import re
from typing import NewType, TypeAlias


NodeId = NewType("NodeId", str)
EdgeId = NewType("EdgeId", str)
FeatureId = NewType("FeatureId", str)
GraphId = NewType("GraphId", str)
ParameterAddress: TypeAlias = tuple[str, GraphId]
RunId = NewType("RunId", str)
AgentProfileId = NewType("AgentProfileId", str)
AgentSessionId = NewType("AgentSessionId", str)
ProviderSessionId = NewType("ProviderSessionId", str)


_ENTITY_ID = re.compile(r"[a-z](?:[a-z0-9_]*[a-z0-9])?")


def is_valid_entity_id(value: str, /) -> bool:
    """Whether *value* is one ASCII definition leaf."""

    return (
        _ENTITY_ID.fullmatch(value) is not None
        and "__" not in value
        and not keyword.iskeyword(value)
        and not keyword.issoftkeyword(value)
    )


def is_valid_definition_id(value: str, /) -> bool:
    """Whether *value* is a canonical ``__``-separated definition path."""

    return bool(value) and all(is_valid_entity_id(part) for part in value.split("__"))
