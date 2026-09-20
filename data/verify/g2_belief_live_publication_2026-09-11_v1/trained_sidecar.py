"""修復capture→固定三seed評価→全joint候補とモデル来歴の専用保存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable
import sidecar as S
import trained_connection as T


def run(capture: Callable[[], Any], members: Any, path: Path, *,
        sample_count: int = T.L.J.DEFAULT_SAMPLES, seed: int = 0) -> dict[str, Any]:
    bound, result, details = T.evaluate(capture, members, sample_count=sample_count, seed=seed)
    manifest = json.dumps(details['members'], sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    packet = S.packet(bound, result, digest)
    packet.update(model_artifact_kind='ensemble_manifest', trained_details=details)
    raw = json.dumps(packet, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
    fresh = capture()
    T.L.C.require(fresh.digest == bound.digest and fresh.tokens == bound.tokens
        and all(a is b for a,b in zip(fresh.values,bound.values,strict=True)), 'save_context_changed')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
    T.L.C.require(path.read_bytes() == raw, 'trained_saved_bytes')
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), frame=bound.frame,
        model_manifest_sha256=digest, probability_p1=result.probability_p1,
        actual_video=False, quality_gate_clear=False)
