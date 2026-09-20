"""原extract使用の人工台帳対照。隠し段/物理分布や実動画成功は証明しない。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest
import check_saved_inputs  # 既存belief依存の通常src選択。
import second_pending_replay as R

SCOPE = ['source', 'run', 0, 1, 2, 0, '2P']


def step(frame: int, consume: bool = False) -> dict:
    events = []
    if consume:
        events = [dict(stage='fifo_before', accounting=dict(pending_tsumo=[[1, 2]]), fifo_occurrence_tokens=['q']),
                  dict(stage='fifo_after', accounting=dict(pending_tsumo=[]), committed=[1, 2], enqueue_occurrence_token='q')]
    return dict(frame_idx=frame, token=f'step:{frame}', events=events, side='2P', source_id='source', run_id='run',
                software_reset=0, pipe_object_id=1, generation=dict(reset_epoch=0), status='returned', exception=None)


def fixture() -> tuple[dict, dict]:
    mode = dict(initial=dict(state=dict(scope=SCOPE, frame=10), source_call_token='step:10'), retired=None,
        applied=[dict(source_call_token='step:14', applied_frame=14, occurrence_token='q', consumed_frame=12)])
    return mode, {f: step(f, f == 12) for f in (10, 12, 14, 16)}


@pytest.mark.parametrize('case', ['normal', 'missing', 'duplicate', 'wrong_receipt', 'foreign', 'failure', 'off_grid'])
def test_delayed_once(case: str) -> None:
    mode, steps = fixture()
    if case == 'missing': steps.pop(12)
    elif case == 'duplicate': steps[16] = step(16, True)
    elif case == 'wrong_receipt': mode['applied'][0]['occurrence_token'] = 'other'
    elif case == 'foreign': steps[12]['run_id'] = 'other'
    elif case == 'failure': steps[12]['exception'] = 'failed'
    elif case == 'off_grid': steps[13] = step(13, True)
    if case == 'normal': assert R.timeline(mode, steps, 16) == {10: (), 12: ('q',), 14: (), 16: ()}
    else:
        with pytest.raises((ValueError, KeyError)): R.timeline(mode, steps, 16)


@pytest.mark.parametrize('corrupt', [False, True])
def test_retire_keeps_pending_and_separates_boundary(corrupt: bool) -> None:
    mode, steps = fixture()
    mode['applied'] = []
    event = R.N.extract(dict(scope=dict(frame_idx=12), token='step:12', events=steps[12]['events']))
    mode['retired'] = dict(frame=14, pending=[] if corrupt else [R.encoded(event)], pending_discarded=False,
                           source_call_token='step:14', reset_call_consumption=None,
                           old_state=dict(scope=SCOPE), new_scope=SCOPE[:5] + [1, '2P'], start_epoch=0, code_sha256='code')
    steps[14].update(generation_after=dict(reset_epoch=1), code_sha256='code')
    if corrupt:
        with pytest.raises(ValueError, match='retirement_pending'): R.timeline(mode, steps, 16)
    else: assert R.timeline(mode, steps, 16) == {10: (), 12: ('q',)}
