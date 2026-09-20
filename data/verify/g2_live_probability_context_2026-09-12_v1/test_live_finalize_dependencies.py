"""原終了部品の整数前提を最小再現。実live終了の合格試験ではない。"""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1'
sys.path.insert(0, str(BASE / 'qa_retired_v1'))
from test_retired import carrier
import retired_completion as R
import runtime_world as W


def test_integer_owner_normal_control() -> None:
    lease, factory, old, state, key = carrier()
    assert R.retired_owner(lease, factory, old, state, state, key)


def test_retired_owner_rejects_probability_baseline_zero() -> None:
    lease, factory, old, state, key = carrier()
    lease.recovery.baseline_count = 0
    with pytest.raises(AssertionError):
        R.retired_owner(lease, factory, old, state, state, key)


def test_world_authority_requires_integer_new_baseline() -> None:
    lease, factory, old, state, key = carrier()
    lease.recovery.baseline_count = 0
    lease.recovery.rows = []
    with pytest.raises(AssertionError, match='unowned_reset_window'):
        W.scope_authority(lease, dict(retired=0, active=0), factory, lease.new_scope, 102)


def test_waiting_rejects_saved_v67_probability_rows() -> None:
    path = BASE / 'publication_boundary/verify.py'
    spec = importlib.util.spec_from_file_location('_probability_finish_boundary_repro', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = json.loads((BASE / 'prefix_cpu_v67_core_activation/LIVE_EMPTY_RESET.json').read_bytes())
    assert saved['error'] is None and saved['recovery']
    assert not any(row['kind'] == 'new_baseline' for row in saved['recovery'])
    with pytest.raises(AssertionError, match='baseline_once'):
        module.waiting({}, [], [], saved['recovery'])
