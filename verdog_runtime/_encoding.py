"""Shared byte encodings used by versioned runtime envelopes."""

from __future__ import annotations

import base64


def encode_base64(value: bytes, /) -> str:
    return base64.b64encode(value).decode("ascii")


def decode_base64(value: str, /) -> bytes:
    return base64.b64decode(value, validate=True)
