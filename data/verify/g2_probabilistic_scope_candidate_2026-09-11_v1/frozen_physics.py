"""実v12と同じ凍結済み連鎖シミュレータを選ぶ。原ファイルは編集しない。"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[3]
SOURCE=PROJECT/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/chain.py'
SHA='911c400dfa5c0881839a7ba20a688f16da6c195acc71adfc693efed63b23d57c'
ALIAS='_g2_probability_scope_frozen_physics'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA,'probability_physics_source'
assert ALIAS not in sys.modules,'probability_physics_foreign_module'
spec=importlib.util.spec_from_file_location(ALIAS,SOURCE)
module=importlib.util.module_from_spec(spec)
sys.modules[ALIAS]=module
spec.loader.exec_module(module)
ChainSimulator=module.ChainSimulator
