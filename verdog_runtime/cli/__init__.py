"""Typed command-line arguments for a workflow envelope."""

from __future__ import annotations

import dataclasses
import pathlib
import types
from collections.abc import Mapping
from typing import Annotated, Generic, TypeAlias, TypeVar, cast, overload

import tyro
from typing_extensions import TypeForm

from verdog_runtime import _process
from verdog_runtime.declarations import ids

InputT = TypeVar("InputT", covariant=True)
RuntimeOptionsT = TypeVar("RuntimeOptionsT", covariant=True)
_InputType: TypeAlias = TypeForm[InputT] | types.UnionType
_CONFIG = (tyro.conf.EnumChoicesFromValues,)
_NO_PARAMETERS: Mapping[ids.ParameterAddress, object] = types.MappingProxyType(
    {}
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowArguments(Generic[InputT, RuntimeOptionsT]):
    """The semantic input, graph parameters, and direct-launch options."""

    input: InputT
    runtime: RuntimeOptionsT
    params: Mapping[ids.ParameterAddress, object] = dataclasses.field(
        default_factory=lambda: _NO_PARAMETERS
    )


def _address_name(address: ids.ParameterAddress, /) -> str:
    project_path, graph_id = address
    parts = (
        list(pathlib.PurePosixPath(project_path).parts)
        if project_path != "."
        else []
    )
    aliases: list[str] = []
    while parts:
        if parts.pop(0) != "external":
            raise ValueError(
                f"unsupported parameter project path: {project_path}"
            )
        package: list[str] = []
        while parts and parts[0] != "external":
            package.append(parts.pop(0))
        if not package:
            raise ValueError(f"invalid parameter project path: {project_path}")
        aliases.append(".".join(package))
    name = str(graph_id).rsplit(".", maxsplit=1)[-1]
    if not name:
        raise ValueError("parameter graph id must not be empty")
    return ".".join((*aliases, name))


def _default_field(value_type: object, /) -> dataclasses.Field[object]:
    """Assemble a field for make_dataclass with an optional empty default."""
    if value_type is types.NoneType:
        default = None
    elif value_type is object:
        default = object()
    else:
        return dataclasses.field()  # pylint: disable=invalid-field-call
    return cast(
        dataclasses.Field[object],
        dataclasses.field(default=default),  # pylint: disable=invalid-field-call
    )


def _annotation(
    value_type: object,
    /,
    *,
    name: str | None = None,
) -> object:
    return Annotated[
        object,
        tyro.conf.arg(constructor=cast(type[object], value_type), name=name),
    ]


def _params_type(
    params_types: Mapping[ids.ParameterAddress, object], /
) -> tuple[type[object], tuple[tuple[str, ids.ParameterAddress], ...]]:
    fields: list[tuple[str, object, dataclasses.Field[object]]] = []
    bindings: list[tuple[str, ids.ParameterAddress]] = []
    names: set[str] = set()
    for index, (raw_address, value_type) in enumerate(params_types.items()):
        address = _process.normalize_parameter_address(raw_address)
        if value_type is types.NoneType or value_type is object:
            continue
        name = _address_name(address)
        if name in names:
            raise ValueError(
                f"duplicate parameter command-line address: {name}"
            )
        names.add(name)
        internal = f"p{index}"
        fields.append(
            (internal, _annotation(value_type, name=name), dataclasses.field())  # pylint: disable=invalid-field-call
        )
        bindings.append((internal, address))
    return (
        cast(
            type[object],
            dataclasses.make_dataclass(
                "_WorkflowParams",
                fields,
                frozen=True,
                slots=True,
                kw_only=True,
            ),
        ),
        tuple(bindings),
    )


@overload
def parse_arguments(
    input_type: _InputType[InputT],
    arguments: tuple[str, ...],
    /,
    *,
    params_types: Mapping[ids.ParameterAddress, object] = _NO_PARAMETERS,
    runtime_options: None = None,
    prog: str | None = None,
) -> WorkflowArguments[InputT, None]: ...


@overload
def parse_arguments(
    input_type: _InputType[InputT],
    arguments: tuple[str, ...],
    /,
    *,
    params_types: Mapping[ids.ParameterAddress, object] = _NO_PARAMETERS,
    runtime_options: type[RuntimeOptionsT],
    prog: str | None = None,
) -> WorkflowArguments[InputT, RuntimeOptionsT]: ...


def parse_arguments(
    input_type: _InputType[InputT],
    arguments: tuple[str, ...],
    /,
    *,
    params_types: Mapping[ids.ParameterAddress, object] = _NO_PARAMETERS,
    runtime_options: type[RuntimeOptionsT] | None = None,
    prog: str | None = None,
) -> WorkflowArguments[InputT, RuntimeOptionsT | None]:
    """Populate the input, addressed graph parameters, and runtime options."""
    params_class, bindings = _params_type(params_types)
    envelope_fields: list[tuple[str, object, dataclasses.Field[object]]] = []
    envelope_fields.append(
        ("input", _annotation(input_type), _default_field(input_type))
    )
    envelope_fields.append(("params", params_class, dataclasses.field()))  # pylint: disable=invalid-field-call
    if runtime_options is not None:
        envelope_fields.append(
            (
                "runtime",
                _annotation(runtime_options),
                _default_field(runtime_options),
            )
        )
    envelope = cast(
        type[object],
        dataclasses.make_dataclass(
            "_WorkflowCli",
            envelope_fields,
            frozen=True,
            slots=True,
            kw_only=True,
        ),
    )
    parsed = tyro.cli(envelope, args=arguments, prog=prog, config=_CONFIG)
    params_value = getattr(parsed, "params")  # noqa: B009 - dynamic dataclass
    params = types.MappingProxyType(
        {
            address: cast(object, getattr(params_value, internal))
            for internal, address in bindings
        }
    )
    return WorkflowArguments(
        input=cast(InputT, getattr(parsed, "input")),  # noqa: B009 - dynamic dataclass
        runtime=(
            None
            if runtime_options is None
            else cast(RuntimeOptionsT, getattr(parsed, "runtime"))  # noqa: B009 - dynamic dataclass
        ),
        params=params,
    )
