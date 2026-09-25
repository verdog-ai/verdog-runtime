"""Safe envelope for opaque continuations owned by isolated child runtimes.

The parent process may inspect and rewrite this JSON envelope, but it must never
deserialize ``runtime``.  Authored child state is decoded only after the child
has checked its own source and environment compatibility fingerprint.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import types
from collections.abc import Mapping
from typing import Literal, cast

from verdog_runtime import _encoding

FORMAT_VERSION = 2
_MAX_NESTING = 128
ForkSessionPolicy = Literal["branch", "fresh"]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PendingFork:
    run_id: str
    source_output: str
    target_output: str
    sessions: ForkSessionPolicy


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ChildCheckpointBundle:
    compatibility: Mapping[str, str]
    runtime: bytes
    shards: Mapping[str, bytes]
    pending_forks: tuple[PendingFork, ...] = ()


def _binary(value: bytes, /) -> str:
    return _encoding.encode_base64(value)


def _require_bytes(value: object, label: str, /) -> bytes:
    if not isinstance(value, bytes):
        raise ValueError(f"child checkpoint {label} is not bytes")
    return value


def _decode_binary(value: object, label: str, /) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"child checkpoint {label} is not encoded bytes")
    try:
        return _encoding.decode_base64(value)
    except ValueError as error:
        raise ValueError(
            f"child checkpoint {label} is not encoded bytes"
        ) from error


def _shard_name(value: object, /) -> str:
    if not isinstance(value, str):
        raise ValueError("child checkpoint shard name is invalid")
    path = pathlib.PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("child checkpoint shard name is invalid")
    return value


def _compatibility(value: object, /) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise ValueError("child checkpoint compatibility is invalid")
    raw = cast(dict[object, object], value)
    if not raw or not all(
        isinstance(key, str) and key and isinstance(item, str) and item
        for key, item in raw.items()
    ):
        raise ValueError("child checkpoint compatibility is invalid")
    return types.MappingProxyType(
        dict(sorted(cast(dict[str, str], raw).items()))
    )


def _absolute_path(value: object, label: str, /) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not pathlib.Path(value).is_absolute()
    ):
        raise ValueError(f"child checkpoint fork {label} is invalid")
    return value


def _pending_fork(value: object, /) -> PendingFork:
    if not isinstance(value, dict):
        raise ValueError("child checkpoint fork is invalid")
    raw = cast(dict[object, object], value)
    if set(raw) != {"run_id", "source_output", "target_output", "sessions"}:
        raise ValueError("child checkpoint fork is invalid")
    run_id = raw["run_id"]
    sessions = raw["sessions"]
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(sessions, str)
        or sessions not in {"branch", "fresh"}
    ):
        raise ValueError("child checkpoint fork is invalid")
    source = _absolute_path(raw["source_output"], "source output")
    target = _absolute_path(raw["target_output"], "target output")
    if pathlib.Path(source).resolve() == pathlib.Path(target).resolve():
        raise ValueError("child checkpoint fork outputs must be distinct")
    return PendingFork(
        run_id=run_id,
        source_output=source,
        target_output=target,
        sessions=cast(ForkSessionPolicy, sessions),
    )


def encode_child_checkpoint(bundle: ChildCheckpointBundle, /) -> bytes:
    compatibility = _compatibility(dict(bundle.compatibility))
    runtime = _require_bytes(bundle.runtime, "runtime")
    shards: dict[str, bytes] = {}
    for raw_name, raw_payload in bundle.shards.items():
        name = _shard_name(raw_name)
        payload = _require_bytes(raw_payload, "shard payload")
        if name in shards:
            raise ValueError("child checkpoint shard is duplicated")
        shards[name] = payload
    pending_forks = tuple(
        _pending_fork(
            {
                "run_id": pending.run_id,
                "source_output": pending.source_output,
                "target_output": pending.target_output,
                "sessions": pending.sessions,
            }
        )
        for pending in bundle.pending_forks
    )
    value = {
        "version": FORMAT_VERSION,
        "compatibility": dict(compatibility),
        "runtime": _binary(runtime),
        "shards": [
            {"name": name, "payload": _binary(payload)}
            for name, payload in sorted(shards.items())
        ],
        "pending_forks": [
            {
                "run_id": pending.run_id,
                "source_output": pending.source_output,
                "target_output": pending.target_output,
                "sessions": pending.sessions,
            }
            for pending in pending_forks
        ],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def decode_child_checkpoint(payload: bytes, /) -> ChildCheckpointBundle:
    try:
        value: object = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("child checkpoint bundle is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("child checkpoint bundle is not an object")
    raw = cast(dict[object, object], value)
    if (
        set(raw)
        != {
            "version",
            "compatibility",
            "runtime",
            "shards",
            "pending_forks",
        }
        or raw["version"] != FORMAT_VERSION
    ):
        raise ValueError("unsupported child checkpoint bundle format")
    shard_values = raw["shards"]
    pending_values = raw["pending_forks"]
    if not isinstance(shard_values, list) or not isinstance(
        pending_values, list
    ):
        raise ValueError("child checkpoint bundle collections are invalid")
    shards: dict[str, bytes] = {}
    for value in cast(list[object], shard_values):
        if not isinstance(value, dict):
            raise ValueError("child checkpoint shard is invalid")
        shard = cast(dict[object, object], value)
        if set(shard) != {"name", "payload"}:
            raise ValueError("child checkpoint shard is invalid")
        name = _shard_name(shard["name"])
        if name in shards:
            raise ValueError("child checkpoint shard is duplicated")
        shards[name] = _decode_binary(shard["payload"], "shard payload")
    pending_forks = tuple(
        _pending_fork(value) for value in cast(list[object], pending_values)
    )
    return ChildCheckpointBundle(
        compatibility=_compatibility(raw["compatibility"]),
        runtime=_decode_binary(raw["runtime"], "runtime"),
        shards=types.MappingProxyType(shards),
        pending_forks=pending_forks,
    )


def mark_child_checkpoint_fork(
    payload: bytes,
    /,
    *,
    run_id: str,
    source_output: pathlib.Path,
    target_output: pathlib.Path,
    sessions: ForkSessionPolicy,
    _depth: int = 0,
) -> bytes:
    """Append a transform without deserializing child state."""
    if _depth >= _MAX_NESTING:
        raise ValueError("child checkpoint nesting is too deep")
    source = source_output.resolve()
    target = target_output.resolve()
    pending = PendingFork(
        run_id=run_id,
        source_output=str(source),
        target_output=str(target),
        sessions=sessions,
    )
    bundle = decode_child_checkpoint(payload)
    shards = {
        name: (
            mark_child_checkpoint_fork(
                value,
                run_id=run_id,
                source_output=source,
                target_output=target,
                sessions=sessions,
                _depth=_depth + 1,
            )
            if name.startswith("children/")
            else value
        )
        for name, value in bundle.shards.items()
    }
    return encode_child_checkpoint(
        ChildCheckpointBundle(
            compatibility=bundle.compatibility,
            runtime=bundle.runtime,
            shards=shards,
            pending_forks=(*bundle.pending_forks, pending),
        )
    )


__all__ = [
    "ChildCheckpointBundle",
    "ForkSessionPolicy",
    "PendingFork",
    "decode_child_checkpoint",
    "encode_child_checkpoint",
    "mark_child_checkpoint_fork",
]
