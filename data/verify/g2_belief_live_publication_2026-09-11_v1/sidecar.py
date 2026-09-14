"""暫定評価と全確率候補を専用票へ保存。整数採録/会計へは公開しない。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any
import live_binding as L
import serialization as S

SCHEMA = 'belief-joint-evaluation-sidecar/v1'


def packet(bound: L.Bound, result: Any, model_artifact_sha256: str) -> dict[str, Any]:
    L.C.require(type(bound) is L.Bound and type(result) is L.J.Result, 'sidecar_types')
    L.C.require(type(model_artifact_sha256) is str and len(model_artifact_sha256) == 64
                and all(c in '0123456789abcdef' for c in model_artifact_sha256), 'model_digest')
    L.C.require(result.frame == bound.frame and result.world_counts == tuple(len(v.worlds) for v in bound.values), 'sidecar_result_scope')
    states = [S.encode(value) for value in bound.values]
    L.C.require(all(S.decode(state) == original for state, original in zip(states, bound.values, strict=True)), 'sidecar_joint_roundtrip')
    return dict(schema=SCHEMA, frame=bound.frame, context_digest=bound.digest,
        journal_tokens=list(bound.tokens), states=states, evaluation=asdict(result),
        model_artifact_sha256=model_artifact_sha256, supported=False,
        ledger_connection='NOT_CONNECTED', queues=bound.inputs.queues.tolist(),
        ledger_values=bound.inputs.ledger_values.tolist(), ledger_availability=bound.inputs.ledger_availability.tolist(),
        provisional=True, accounting_permission=False, integer_current_permission=False,
        training_permission=False, quality_gate_clear=False)


def save(path: Path, bound: L.Bound, result: Any, model_artifact_sha256: str) -> str:
    value = packet(bound, result, model_artifact_sha256)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
    # 排他作成。途中失敗の残存票も上書きせず、上位finish不成立として保持する。
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
    L.C.require(path.read_bytes() == raw, 'sidecar_saved_bytes')
    return hashlib.sha256(raw).hexdigest()
