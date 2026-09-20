"""保存済み実境界を使い、片側欠測・別run・古frame・世代違いを拒否する。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import pytest
import check_saved_inputs as C
import journal_context as J


@pytest.fixture(scope='module')
def evidence() -> tuple[Any, ...]:
    value = C.S.decode([r for r in C.rows('PROBABILISTIC_TRACKING.jsonl')
                       if r['physical_transition_applied']][-1]['transition']['state'])
    row = next(r for r in C.rows('provisional_context.jsonl') if r['frame_idx'] == value.frame)
    steps = [r for r in C.rows('atomic_journal.jsonl')
             if r['kind'] == 'step' and r['frame_idx'] == value.frame]
    return row, steps, value.scope


def test_saved_two_sides(evidence: tuple[Any, ...]) -> None:
    assert J.join(*evidence) == ('step:598', 'step:599')
    assert evidence[1][0]['software_reset'] != evidence[1][1]['software_reset']


@pytest.mark.parametrize('case', ['missing', 'order', 'run', 'frame', 'generation', 'exception', 'token', 'gap', 'malformed'])
def test_reject(evidence: tuple[Any, ...], case: str) -> None:
    row, steps, scope = deepcopy(evidence)
    if case == 'missing': steps.pop()
    elif case == 'order': steps.reverse()
    elif case == 'run': steps[1]['run_id'] = 'foreign'
    elif case == 'frame': steps[1]['frame_idx'] -= 2
    elif case == 'generation': steps[1]['generation_after']['reset_epoch'] += 1
    elif case == 'exception': steps[1]['exception'] = 'failed'
    elif case == 'token': steps[1]['token'] = steps[0]['token']
    elif case == 'gap': steps[1]['token'] = 'step:601'
    elif case == 'malformed': steps[1]['token'] = 'step:bad'
    with pytest.raises(ValueError):
        J.join(row, steps, scope)
