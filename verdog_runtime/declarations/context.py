"""Per-visit contexts for parameters, agent access, and child calls."""

from __future__ import annotations

import dataclasses
import enum
import pathlib
from collections.abc import Callable
from typing import Generic, cast, overload

from typing_extensions import TypeVar as TypeVarWithDefault

from verdog_runtime.declarations import ids
from verdog_runtime.declarations.ids import RunId as RunId

ParamsT = TypeVarWithDefault("ParamsT", default=None)
ChildParamsT = TypeVarWithDefault("ChildParamsT", default=None)
ChildInputT = TypeVarWithDefault("ChildInputT", default=object)
ChildOutputT = TypeVarWithDefault("ChildOutputT", default=object)
_USE_CHILD_PARAMS = object()


class AgentAccess(enum.StrEnum):
    """Filesystem access permitted for agent invocations."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class NodeContext(Generic[ParamsT]):
    """Identifiers, output directory, and parameters for one node visit."""

    run_id: RunId
    graph_id: ids.GraphId
    node_id: ids.NodeId
    edge_id: ids.EdgeId
    output_dir: pathlib.Path
    params: ParamsT


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentNodeContext(NodeContext[ParamsT], Generic[ParamsT]):
    """A node context that also provides the bound agent invocation."""

    _invoke: Callable[[str, pathlib.Path, AgentAccess], str] = (
        dataclasses.field(repr=False)
    )

    def invoke(
        self,
        prompt: str,
        /,
        *,
        workspace: pathlib.Path,
        access: AgentAccess = AgentAccess.READ_ONLY,
    ) -> str:
        """Invoke the bound agent and return its final response text."""
        return self._invoke(prompt, workspace, access)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CallContext(
    NodeContext[ParamsT],
    Generic[
        ParamsT,
        ChildParamsT,
        ChildInputT,
        ChildOutputT,
    ],
):
    """A node context that invokes its child with typed arguments."""

    child_params: ChildParamsT
    _invoke: Callable[[ChildInputT, ChildParamsT], ChildOutputT] = (
        dataclasses.field(repr=False)
    )

    @overload
    def invoke(self, input: ChildInputT, /) -> ChildOutputT: ...

    @overload
    def invoke(
        self,
        input: ChildInputT,
        /,
        *,
        params: ChildParamsT,
    ) -> ChildOutputT: ...

    def invoke(
        self,
        input: ChildInputT,
        /,
        *,
        params: ChildParamsT | object = _USE_CHILD_PARAMS,
    ) -> ChildOutputT:
        """Run the child, optionally overriding its configured parameters."""
        child_params = (
            self.child_params
            if params is _USE_CHILD_PARAMS
            else cast(ChildParamsT, params)
        )
        return self._invoke(input, child_params)
