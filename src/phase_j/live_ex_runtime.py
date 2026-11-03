"""EX専用子プロセスの資産読込。既存C2の設定は変更しない。"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .live_ex_engine import LiveExEngine

SEED_SHA256 = '5aec04e78d2e3128b4706978969c669b5cb27ee79f7e22ea9f7d40f219d4f851'
POLICY_SHA256 = 'b32374fb8bfd07ee739f9559f9e6f38b88d5ac3c000e3a71dedb9fcb66c309c4'


def create_engine(settings: dict, epoch: int) -> LiveExEngine:
    """固定したnative・種表・policyを明示指定して読み込む。"""
    import numpy as np
    from .live_chain_split_config import set_split_env
    set_split_env(settings.get('chain_split_place', False))
    if 'work_limit' in settings:
        os.environ['PUYO_EX_WORK_BUDGET'] = '1'
        os.environ['PUYO_EX_WORK_LIMIT'] = str(settings['work_limit'])
    os.environ['PUYO_PREFIRE_V49F_DETERMINISTIC'] = '1'
    sys.path.insert(0, settings['native_dir'])
    native = importlib.import_module('puyo_core')
    from .live_ex_engine import LiveExEngine
    from ._live_ex_frozen.src.prefire_tsumo_v411 import SeedLookup
    from ._live_ex_frozen.src.tsumo_seed_identifier import SeedTable
    for name in ('ama_search_py', 'simulate_chain_trace_py', 'enumerate_and_simulate_placements_py'):
        if not hasattr(native, name):
            raise RuntimeError(f'EX native API不足: {name}')
    from .live_ex_native import verify_native
    receipt = verify_native(native)
    policy = json.loads(Path(settings['policy']).read_text(encoding='utf-8-sig'))
    policy_sha = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    seed_sha = hashlib.sha256(Path(settings['seed_table']).read_bytes()).hexdigest()
    if policy_sha != POLICY_SHA256 or seed_sha != SEED_SHA256:
        raise ValueError(f'EX固定資産と不一致: policy={policy_sha == POLICY_SHA256}, seed={seed_sha == SEED_SHA256}')
    with np.load(settings['seed_table'], allow_pickle=False) as data:
        if int(data['version']) != 1:
            raise ValueError('EX種表の版が違う')
        lookup = SeedLookup(SeedTable(data['cmap'].copy(), data['pool'].copy()))
    engine = LiveExEngine(policy, native, lookup, epoch, connect_ledger,
                          tape_next=settings.get('tape_next', False), stable_key=settings.get('stable_key', False),
                          xc_same_time=settings.get('xc_same_time', False))
    if settings.get('ex_speed', False):
        from .live_ex_speed import matching_positions
        engine.engine.ex_speed = True
        engine.engine.retr_seed_matcher = matching_positions
    engine.receipt = dict(native=receipt, policy_sha256=policy_sha, seed_sha256=seed_sha)
    return engine


def connect_ledger(engine: Any) -> None:
    """正式OBS抽出器と同じ疎な得点窓で台帳を進める。"""
    from .live_ex_ledger import ObsLedger
    if not isinstance(engine.state.sl.core, ObsLedger):
        engine.state.sl.core = ObsLedger(newest_first_offset=True,
            defer_all_clear=True, all_clear_native=engine.native)


def settings_from_env(defaults: dict | None = None) -> dict:
    names = dict(policy='PUYO_LIVE_EX_POLICY', seed_table='PUYO_LIVE_EX_SEED_TABLE',
                 native_dir='PUYO_LIVE_EX_NATIVE_DIR')
    result = {}
    for name, env in names.items():
        value = os.environ.get(env, (defaults or {}).get(name))
        if value is None:
            raise KeyError(env)
        result[name] = value
    for name in ('policy', 'seed_table'):
        path = Path(result[name]).resolve(strict=True)
        result[name] = str(path)
        result[name + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['native_dir'] = str(Path(result['native_dir']).resolve(strict=True))
    return result
