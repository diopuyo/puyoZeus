"""固定fold6三seedの既存loader/較正器へjoint候補を接続する。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any, Callable
import live_binding as L

ROOT = Path(__file__).resolve().parent.parent/'g2_trained_context_score_2026-09-09_v1'
PINS = {'score.py':'90550cf638f331e8eb9ade46607262519e78503a943d2730d890068cf0cc6e06',
        'loader.py':'2fd2f8cefc9d0bd513480e884b05164e232b52862b7552a9b22a52dfd8670d0f'}
ALIAS = '_belief_trained_score'


def backend() -> Any:
    L.C.require(all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha for name,sha in PINS.items()), 'trained_source')
    if ALIAS not in sys.modules:
        spec = importlib.util.spec_from_file_location(ALIAS, ROOT/'score.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[ALIAS] = module
        spec.loader.exec_module(module)
    module = sys.modules[ALIAS]
    L.C.require(Path(module.__file__).resolve() == ROOT/'score.py', 'trained_alias')
    module.fixed_sources()
    return module


def evaluate(capture: Callable[[], L.Bound], members: Any, *,
             sample_count: int = L.J.DEFAULT_SAMPLES, seed: int = 0) -> tuple[Any, Any, dict[str, Any]]:
    score = backend()
    first = capture()
    source = first.values[0].scope[0]
    score.members_ready(source, members, 'cpu')
    scorers = []
    def make(inputs: Any) -> Any:
        value = score.BatchScorer(N(inputs=inputs, supported=False, integrity_valid=True), members, 'cpu')
        scorers.append(value)
        return value
    bound, calibrated = L.evaluate(capture, make, sample_count=sample_count, seed=seed)
    L.C.require(first.digest == bound.digest and all(a is b for a,b in zip(first.values,bound.values,strict=True)), 'trained_initial_changed')
    scorer = scorers[0]
    raw = L.J.evaluate(bound.values, bound.frame, ('STABLE','STABLE'), bound.observed,
                       scorer.raw_cached, sample_count=sample_count, seed=seed)
    score.members_ready(source, members, 'cpu')
    after = capture()
    L.C.require(after.digest == bound.digest and all(a is b for a,b in zip(after.values,bound.values,strict=True)), 'trained_final_changed')
    details = dict(raw=asdict(raw), sample_digest=scorer.sample_digest,
        seed_raw={str(k):v.tolist() for k,v in scorer.raw.items()},
        seed_calibrated={str(k):v.tolist() for k,v in scorer.calibrated.items()},
        members=[dict(seed=m.seed, fold=m.fold, slope=m.slope, checkpoint_sha256=m.checkpoint_sha256,
                      model_state_sha256=m.model_state_sha256) for m in members],
        trained_weights=True, supported=False, quality_gate_clear=False)
    return bound, calibrated, details
