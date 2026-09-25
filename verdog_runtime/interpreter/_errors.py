from typing import Never

from ..declarations.ids import EdgeId, FeatureId, NodeId


def fault(entity_id: NodeId | EdgeId | FeatureId, code: str, message: str) -> Never:
    error = RuntimeError(f"{message} [{code}]")
    error.add_note(f"Verdog entity: {entity_id}")
    raise error
