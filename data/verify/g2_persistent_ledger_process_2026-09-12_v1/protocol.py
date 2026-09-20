"""frozen親でも使える標準libraryだけの会計通信規約。"""
from __future__ import annotations
import hashlib
import json
from typing import Any

SCHEMA = 'g2-ledger-process/v1'
MAX_BYTES = 32 * 1024 * 1024
TIMEOUT = 180.0
IDENTITY_KEYS = ('source_video_id', 'build_id', 'attempt_id')


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


def invalid_constant(value: str) -> None:
    raise ValueError('nonfinite_json:' + value)


def decoded(raw: bytes) -> Any:
    value = json.loads(raw, parse_constant=invalid_constant)
    encoded(value)
    return value


def event_sha(prefix: Any) -> str:
    return hashlib.sha256(b''.join(encoded(batch) for batch in prefix)).hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('ledger_process:' + reason)
