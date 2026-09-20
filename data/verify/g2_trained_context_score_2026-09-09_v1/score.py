"""同じ隠し段sampleで三seedを個別較正し、既存サンプラへ渡す。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BINDING_ROOT = ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1'
sys.path.insert(0, str(BINDING_ROOT))
import binding as B
from scripts.advantage_m1_outer_fold_baseline_contract_v1 import equal_seed_probability_mean, M1_SEEDS
from scripts.train_advantage_m1_causal_ledger_v1 import calibrate

FIXED = {BINDING_ROOT / 'binding.py': '401c1e1a5c5f4965c20750a9c47ecd210deab083f958f3ef4b29bf5c3db1aa2d',
    B.BRIDGE: B.BRIDGE_SHA,
    PROJECT / 'scripts/advantage_m1_outer_fold_baseline_contract_v1.py':
        'bb7e170d1421fa5fa13a75e7ebab1082852231776ccc3fb9cdbf568520b6279f',
    PROJECT / 'scripts/train_advantage_m1_causal_ledger_v1.py':
        '9becc13b80a39f101909b2d7db15d569d4e1369e67191d84cde3247c6c8f2b47'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_sources() -> None:
    B.require(all(sha(p) == h for p, h in FIXED.items()), 'score_fixed_source_changed')


def load_loader() -> Any:
    """親/子が同じ実classを共有し、異なるbare aliasの混入は拒否する。"""
    path = ROOT / 'loader.py'
    if 'loader' in sys.modules:
        module = sys.modules['loader']
        B.require(Path(module.__file__).resolve() == path, 'loader_alias_collision')
        return module
    spec = importlib.util.spec_from_file_location('loader', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def members_ready(source_id: str, members: Sequence[Any], device: str) -> None:
    load_loader().verify_members(source_id, members, device=device)


def fresh_bound(bound: Any) -> tuple[Any, dict[str, Any]]:
    """元evaluate_boundと同じ再bindを維持し、callerの制御boolを信用しない。"""
    fixed_sources()
    B.require(type(bound) is B.BoundContext, 'bound_type')
    payload = json.loads(bound.source_json)
    fresh = B.bind_provisional_context(payload['row'], payload['registration'])
    B.require(bound.context_digest == fresh.context_digest and bound.supported is False
        and bound.integrity_valid is True and bound.reason == fresh.reason, 'bound_tampered')
    for key in ('boards', 'queues', 'ledger_values', 'ledger_availability'):
        B.require(np.array_equal(getattr(bound.inputs, key), getattr(fresh.inputs, key)), 'bound_inputs_changed')
    return fresh, payload


def bridge_module() -> Any:
    fixed_sources()
    spec = importlib.util.spec_from_file_location('_trained_context_probability_bridge', B.BRIDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchScorer:
    """モデル出力rawの意味を変えず、別配列として較正する。"""
    def __init__(self, fresh: Any, members: Sequence[Any], device: str) -> None:
        B.require(tuple(m.seed for m in members) == M1_SEEDS, 'seed_order')
        bridge = bridge_module()
        self.members = tuple(members)
        self.scorers = {m.seed: bridge.M1ProbabilityScorer(m.model, fresh.inputs,
            supported=fresh.supported, integrity_valid=fresh.integrity_valid, device=device) for m in members}
        self.sample_digest: str | None = None
        self.raw: dict[int, np.ndarray] = {}
        self.calibrated: dict[int, np.ndarray] = {}

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        B.require(self.sample_digest is None, 'unexpected_repeated_model_batch')
        self.sample_digest = B.digest(samples.tolist())
        for member in self.members:
            raw = self.scorers[member.seed](samples)
            self.raw[member.seed] = raw.copy()
            self.calibrated[member.seed] = calibrate(raw, member.slope)
        B.require(B.digest(samples.tolist()) == self.sample_digest, 'sample_mutated')
        return equal_seed_probability_mean(self.calibrated)

    def raw_cached(self, samples: np.ndarray) -> np.ndarray:
        B.require(self.sample_digest is not None and B.digest(samples.tolist()) == self.sample_digest,
            'raw_and_calibrated_samples_differ')
        return equal_seed_probability_mean(self.raw)


def evaluate(bound: Any, members: Sequence[Any], *, sample_count: int = 256,
             seed: int = 0, device: str = 'cpu') -> dict[str, Any]:
    fresh, payload = fresh_bound(bound)
    B.require(type(sample_count) is int and sample_count > 0 and type(seed) is int and seed >= 0,
        'sample_count_or_seed')
    members_ready(payload['registration']['source_id'], members, device)
    scorer = BatchScorer(fresh, members, device)
    pair = [B._candidate(payload['row'], side) for side in B.SIDES]
    calibrated = B.C.evaluate_pair(*pair, scorer, sample_count=sample_count, seed=seed)
    # 同じ既存サンプラを同seedで再生成し、モデルは再呼出せずraw保存列だけ計算。
    raw = B.C.evaluate_pair(*pair, scorer.raw_cached, sample_count=sample_count, seed=seed)
    return {'context_digest': fresh.context_digest, 'raw': asdict(raw), 'calibrated': asdict(calibrated),
        'sample_digest': scorer.sample_digest, 'seed_raw': {str(k): v.tolist() for k, v in scorer.raw.items()},
        'seed_calibrated': {str(k): v.tolist() for k, v in scorer.calibrated.items()},
        'model_sample_batches': len(members), 'supported': False, 'integrity_valid': True,
        'evaluation_mode': 'trained_m0_only_with_hidden_distribution', 'quality_gate_clear': False,
        'calibration_quality_measured': False, 'production_permission': False}
