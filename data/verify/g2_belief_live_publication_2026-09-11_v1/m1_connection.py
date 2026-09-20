"""検証済み生contextを既存M1へ接続。未知会計のresidualは有効化しない。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Callable
import live_binding as L

BRIDGE = Path(__file__).resolve().parent.parent/'g2_hidden_probability_provisional_2026-09-08_v1/model_bridge.py'
BRIDGE_SHA = 'd7f1a0658aff0a8bc42386d1377e320a5835aa8d9b9ee418e9e24ce0328f1022'


def evaluate(capture: Callable[[], L.Bound], model: Any, *, sample_count: int = L.J.DEFAULT_SAMPLES,
             seed: int = 0, device: str = 'cpu') -> tuple[L.Bound, Any]:
    L.C.require(hashlib.sha256(BRIDGE.read_bytes()).hexdigest() == BRIDGE_SHA, 'M1_bridge_source')
    spec = importlib.util.spec_from_file_location('_belief_publication_m1_bridge', BRIDGE)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    def scorer(inputs: Any) -> Any:
        return bridge.M1ProbabilityScorer(model, inputs, supported=False, integrity_valid=True, device=device)
    return L.evaluate(capture, scorer, sample_count=sample_count, seed=seed)
