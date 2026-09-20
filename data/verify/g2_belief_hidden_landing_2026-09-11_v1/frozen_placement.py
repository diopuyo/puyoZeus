"""実物理の着地候補列挙を原snapshotから固定して借りる。原ファイルは読むだけ。

新しい落下シミュレータは作らない。列挙・色配分・書き込みはすべて原
placement_inferrer の API をそのまま使う。live src/placement_inferrer.py は
本日時点で snapshot と byte 同一であることも assert する。
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[3]
SOURCE=PROJECT/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/placement_inferrer.py'
LIVE=PROJECT/'src/placement_inferrer.py'
SHA='412a15120db0d6e9c10f38ad8d5fa8dff1dfa05f05f3d1a14b71f4ce10610b82'
ALIAS='_g2_hidden_landing_frozen_placement'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA,'hidden_landing_placement_source'
assert hashlib.sha256(LIVE.read_bytes()).hexdigest()==SHA,'hidden_landing_placement_live_drift'
assert ALIAS not in sys.modules,'hidden_landing_placement_foreign_module'
spec=importlib.util.spec_from_file_location(ALIAS,SOURCE)
module=importlib.util.module_from_spec(spec)
sys.modules[ALIAS]=module
spec.loader.exec_module(module)

LandingPattern=module.LandingPattern
enumerate_landing_patterns=module.enumerate_landing_patterns
enumerate_color_assignments=module.enumerate_color_assignments
materialize_pattern=module.materialize_pattern
