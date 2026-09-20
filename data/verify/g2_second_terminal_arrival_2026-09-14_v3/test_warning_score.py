"""A36の原保存列を変えず、落下得点の誤拒否と実連鎖/相殺負例を検査する。"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any
import pytest
import warning_source as W

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'g2_second_terminal_arrival_2026-09-14_v1/warning_source.py'
SAVED = ROOT.parent / 'video38_second_prefix_candidate_v36/SECOND_WARNING_932e60a0e8d4762e.jsonl'
CURRENT, LAST = 36678, 36820


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def rows() -> tuple:
    value = [json.loads(line) for line in SAVED.read_text().splitlines()]
    return tuple(row for row in value if CURRENT < row['frame'] <= LAST)


def verify(value: tuple) -> dict:
    return W.select(require, value, tuple(value[0]['scope']), CURRENT, LAST)


def test_actual_false_rejection_and_repaired_contract() -> None:
    before = hashlib.sha256(SAVED.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('_original_warning_score_test', ORIGINAL)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    saved = rows()
    with pytest.raises(ValueError, match='warning_cancellation_or_unknown'):
        old.select(require, saved, tuple(saved[0]['scope']), CURRENT, LAST)
    result = verify(saved)
    assert result['first_positive'] == 36768 and result['first_confirmed_ojama'] == 36804
    assert result['score_evidence']['total_delta'] == 6
    assert len(result['score_evidence']['positive_calls']) == 6
    assert result['conditional_drop_amount'] == 30
    assert not result['calibrated'] and not result['future_landing_guaranteed']
    assert before == hashlib.sha256(SAVED.read_bytes()).hexdigest()
    packet = dict(classification='inspection_false_rejection_user_confirmed_soft_drop',
        original_exit_preserved=1, actual_saved_warning_rows=True, result=result,
        source_sha256=before, live_pipeline_resumed=False, quality_gate_clear=False)
    with (ROOT / 'A36_WARNING_REPLAY_v1.json').open('x') as stream:
        json.dump(packet, stream, indent=2)


@pytest.mark.parametrize('case', ['all_zero', 'tail_small_drop'])
def test_normal_controls(case: str) -> None:
    value = deepcopy(rows())
    for row in value:
        row['own_score_delta'] = 0
    if case == 'tail_small_drop':
        value[-2]['own_score_delta'] = 1
    assert verify(value)['conditional_drop_amount'] == 30


@pytest.mark.parametrize('case', ['chain40', 'chain80', 'mixed41', 'pure10', 'accumulated40',
    'negative', 'nan', 'fraction', 'chain', 'origin', 'event', 'unknown', 'interrupted', 'gap'])
def test_reject_erasure_cancellation_and_missing_evidence(case: str) -> None:
    value = list(deepcopy(rows()))
    if case == 'accumulated40':
        for row in value:
            row['own_score_delta'] = 0
        for row in value[:W.MIN_ERASURE_SCORE]:
            row['own_score_delta'] = 1
    elif case in ('chain40', 'chain80', 'mixed41', 'pure10', 'negative', 'nan', 'fraction'):
        value[-2]['own_score_delta'] = dict(chain40=40, chain80=80, mixed41=41,
            pure10=10, negative=-1, nan=float('nan'), fraction=0.5)[case]
    elif case == 'interrupted':
        value[-2]['observation']['status'] = 'NOT_POSITIVE'
    elif case == 'gap':
        value.pop(0)
    else:
        field, setting = dict(chain=('own_chain_active', True), origin=('active_origin', True),
            event=('chain_event', True), unknown=('context_known', False))[case]
        value[-2][field] = setting
    with pytest.raises(ValueError, match='warning_'):
        verify(tuple(value))
