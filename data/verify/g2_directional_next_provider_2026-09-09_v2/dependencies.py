"""既存の計測・候補・packet投影を固定内容で再用する。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
FIXED = {
    'candidate': ('g2_directional_next_candidate_2026-09-09_v1/candidate.py',
        'c224066f92f43c6fa095d7cb53e4da578d070500d3b6907169b62a0f0b121667'),
    'observables': ('g2_next_motion_observables_2026-09-09_v1/observables.py',
        '31b29d44669cc62727c18dac0bbd235a7382eba25139ed224995edba50a6606c'),
    'packet': ('g2_directional_next_parent_holdout_2026-09-09_v1/replay_candidate.py',
        'fec515b6402bf1314bef5bc52e224aeaf6f723f8bb99ae8f280069ffd4af3d29'),
}


OWNED: dict[str, ModuleType] = {}


def load(name: str) -> Any:
    relative, expected = FIXED[name]
    path = ROOT.parent / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('dependency_changed:' + name)
    alias = '_parent_motion_provider_' + name
    if alias in sys.modules:
        cached = sys.modules[alias]
        if cached is not OWNED.get(name) or type(cached) is not ModuleType:
            raise ValueError('foreign_dependency_alias:' + name)
        if Path(vars(cached).get('__file__', '')).resolve() != path.resolve():
            raise ValueError('cached_dependency_origin:' + name)
        return cached
    if name in OWNED:
        raise ValueError('owned_dependency_alias_removed:' + name)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    OWNED[name] = module
    return module
