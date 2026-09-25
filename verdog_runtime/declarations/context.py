from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Generic, cast, overload

from typing_extensions import TypeVar as TypeVarWithDefault

from .ids import EdgeId, GraphId, NodeId, RunId as RunId


ParamsT = TypeVarWithDefault("ParamsT", default=None)
ChildParamsT = TypeVarWithDefault("ChildParamsT", default=None)
ChildInputT = TypeVarWithDefault("ChildInputT", default=object)
ChildOutputT = TypeVarWithDefault("ChildOutputT", default=object)
_USE_CHILD_PARAMS = object()


class AgentAccess(StrEnum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeContext(Generic[ParamsT]):
    run_id: RunId
    graph_id: GraphId
    node_id: NodeId
    edge_id: EdgeId
    output_dir: Path
    params: ParamsT


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentNodeContext(NodeContext[ParamsT], Generic[ParamsT]):
    _invoke: Callable[[str, Path, AgentAccess], str] = field(repr=False)

    def invoke(
        self,
        prompt: str,
        /,
        *,
        workspace: Path,
        access: AgentAccess = AgentAccess.READ_ONLY,
    ) -> str:
        return self._invoke(prompt, workspace, access)


@dataclass(frozen=True, slots=True, kw_only=True)
class CallContext(
    NodeContext[ParamsT],
    Generic[
        ParamsT,
        ChildParamsT,
        ChildInputT,
        ChildOutputT,
    ],
):
    child_params: ChildParamsT
    _invoke: Callable[[ChildInputT, ChildParamsT], ChildOutputT] = field(repr=False)

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
        child_params = (
            self.child_params
            if params is _USE_CHILD_PARAMS
            else cast(ChildParamsT, params)
        )
        return self._invoke(input, child_params)
