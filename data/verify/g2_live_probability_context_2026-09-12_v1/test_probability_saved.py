"""既存v67静止/v64通常手の原保存票を再用。生きたfactory終了の合格ではない。"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
import pytest
import probability_saved as P

VERIFY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VERIFY / 'g2_probabilistic_scope_candidate_2026-09-11_v1'))
import serialization as S
import native_consumption as N

BASE = VERIFY / 'g2_empty_tail_reset_integration_2026-09-11_v1'


def saved(case: str) -> tuple[Any, Any, Any]:
    output = BASE / case
    initial = S.decode(P.read(output, 'PROBABILISTIC_BASIS.jsonl')[0]['state'])
    tracking = P.read(output, 'PROBABILISTIC_TRACKING.jsonl')
    journal = P.read(output, 'atomic_journal.jsonl')
    steps = {row['token']: row for row in journal if row['kind'] == 'step'}
    return tracking, initial, steps


@pytest.mark.parametrize('case', ('prefix_cpu_v67_core_activation', 'prefix_cpu_v64_joint_saved'))
def test_original_static_and_normal_hand_saved(case: str) -> None:
    tracking, initial, steps = saved(case)
    current, events, receipts = P.transitions(tracking, initial, S, N, steps)
    assert not events['pending'] and current.scope == initial.scope
    if case == 'prefix_cpu_v67_core_activation':
        assert current == initial and not events['used'] and not receipts
    else:
        assert events['used'] and receipts and current.frame > initial.frame


@pytest.mark.parametrize('case', ('native', 'pending', 'call'))
def test_static_saved_corruption_rejected(case: str) -> None:
    tracking, initial, steps = saved('prefix_cpu_v67_core_activation')
    if case == 'native': tracking[0]['native_consumption'] = dict(invented=True)
    elif case == 'pending': tracking[0]['pending_occurrences'] = ['invented']
    else: steps[tracking[0]['journal_token']]['frame_idx'] += 2
    with pytest.raises(AssertionError):
        P.transitions(tracking, initial, S, N, steps)
