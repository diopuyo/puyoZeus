"""新しい段階照合だけの人工正常・改変対照。元資格判定は実prefix検査で別途通す。"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import sys

import pytest


@pytest.fixture
def module(monkeypatch: Any) -> Any:
    def require(ok: bool, reason: str) -> None:
        if not ok:
            raise ValueError(reason)
    old = N(BASE=None, L=N(require=require), capture=None, grid=None, validate=lambda v: None)
    monkeypatch.setitem(sys.modules, 'old_stable', old)
    path = Path(__file__).with_name('stable_capture_v2.py')
    spec = importlib.util.spec_from_file_location('_late_stable_test', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def packet() -> tuple[dict, dict, dict]:
    grid = [[0] * 6 for _ in range(13)]
    board = dict(grid=grid, object_id=1, sha256='人工同一盤面')
    scope = dict(frame_idx=10, time_sec=10/60, side='1P', source_id='source',
                 run_id='run', pipe_object_id=2, generation=dict(reset_epoch=3))
    events = [dict(stage=name, line=line, state='STABLE', confirmed=deepcopy(board))
              for name, line in zip(('publication_before', 'publication_after'), (8632, 8637))]
    step = dict(scope, token='step:1', software_reset=3, events=events, status='returned',
                exception=None, returned=dict(state='STABLE', confirmed=deepcopy(board)))
    value = dict(source_call_token='step:1', scope=scope, software_reset=3,
                 raw=grid, cnn=grid, sm=grid, returned=grid)
    row = dict(journal_token='step:1', stable_qualification=deepcopy(value))
    side = dict(pb=dict(raw=deepcopy(board)), sm=dict(input_cnn=deepcopy(board),
                context_after=dict(confirmed=deepcopy(board))),
                before_hold=dict(confirmed=deepcopy(board)))
    return row, step, dict(scope, sides={'1P': side})


def test_original_early_sm_can_differ_from_late_sm(module: Any) -> None:
    row, step, context = packet()
    context['sides']['1P']['sm']['context_after']['confirmed']['grid'][2][3] = 5
    assert module.verify(row, step, context) is True


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'line', 'state', 'board', 'identity',
                                    'raw', 'late_both', 'returned', 'scope', 'epoch', 'exception'])
def test_rejects_missing_or_changed_evidence(module: Any, mutation: str) -> None:
    row, step, context = packet()
    events = step['events']
    if mutation == 'missing': events.pop()
    elif mutation == 'duplicate': events.append(deepcopy(events[-1]))
    elif mutation == 'line': events[-1]['line'] += 1
    elif mutation == 'state': events[-1]['state'] = 'CHAIN'
    elif mutation == 'board': events[-1]['confirmed']['grid'][2][3] = 5
    elif mutation == 'identity': events[-1]['confirmed']['object_id'] += 1
    elif mutation == 'raw': context['sides']['1P']['pb']['raw']['grid'][2][3] = 5
    elif mutation == 'late_both':
        for event in events: event['confirmed']['grid'][2][3] = 5
    elif mutation == 'returned': step['returned']['confirmed']['grid'][2][3] = 5
    elif mutation == 'scope': context['run_id'] = 'foreign'
    elif mutation == 'epoch': step['software_reset'] += 1
    elif mutation == 'exception': step['exception'] = 'original failure'
    with pytest.raises(ValueError): module.verify(row, step, context)
