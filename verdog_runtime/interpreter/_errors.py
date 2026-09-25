"""Attach graph and node context to execution failures."""

from typing import Never

from verdog_runtime.declarations import ids


def fault(
    entity_id: ids.NodeId | ids.EdgeId | ids.FeatureId, code: str, message: str
) -> Never:
    error = RuntimeError(f"{message} [{code}]")
    error.add_note(f"Verdog entity: {entity_id}")
    raise error
