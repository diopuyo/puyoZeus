"""2Pの原completeを人工frameで駆動し、実保存PBの捕捉と復元を検査する。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import inspect
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_live_binding import saved, live
import test_journal_witness as T
import second_observation as O


def generated(journal: Any, pipe: Any, result: Any, row: Any) -> None:
    side, frame_idx, time_sec = '2P', row['frame_idx'], row['time_sec']
    sm = pipe._sm_2p
    signals = N(is_match_active=True, effect_gate_window_active=False)
    item = dict(frame=inspect.currentframe(), pipe=pipe, scope=journal.scope(pipe,side,frame_idx,time_sec),
                epoch=0, token='step:599', events=[], return_line=None)
    journal.complete_step(item, result, None, sys.getprofile())
    assert item['frame'] is None


@pytest.fixture
def context(live: Any) -> Any:
    row, pipe = live.rec.rows[-1], live.pipe
    pipe._landing_grace_2p = None
    journal = T.recorder()
    journal.active, journal.pipe, journal.codes = None, pipe, {generated.__code__}
    scope = {k:live.steps[1][k] for k in ('frame_idx','time_sec','side','source_id','run_id','pipe_object_id','generation')}
    journal.scope = lambda *args: deepcopy(scope)
    journal.epoch = lambda *args: 0
    journal.tracker = N(generation=lambda side:T.Generation(**scope['generation']))
    state = dict(hidden_probability_observer=N(active=None, failures=[], rows=[deepcopy(row['sides']['2P']['pb'])]),
        postcommit_current_receiver=N(rec=N(side_value=lambda result:deepcopy(row['sides']['2P']['before_hold']))))
    board = pipe._sm_2p.context.confirmed_board
    result = N(confirmed_board=board, inferred_board=board, state=N(name='STABLE'))
    return N(journal=journal, pipe=pipe, state=state, row=row, result=result)


def test_capture_before_original_cleanup(context: Any) -> None:
    c = context
    original = c.journal.complete_step
    with ExitStack() as stack:
        evidence = O.install(stack, c.journal, c.state)
        generated(c.journal, c.pipe, c.result, c.row)
        assert evidence.latest['state'] == 'stable' and evidence.latest['scope'][-1] == '2P'
        assert evidence.latest['probability']['cells'] == c.state['hidden_probability_observer'].rows[0]['probability']['cells']
        assert not evidence.latest['basis_registered'] and c.journal.count == 1
    assert evidence.closed and c.journal.complete_step == original


def test_wrong_PB_frame_rejected_after_original(context: Any) -> None:
    c = context
    c.state['hidden_probability_observer'].rows[0]['frame_idx'] -= 2
    with ExitStack() as stack:
        evidence = O.install(stack, c.journal, c.state)
        with pytest.raises(ValueError, match='second_PB_scope'):
            generated(c.journal, c.pipe, c.result, c.row)
        assert evidence.latest is None and c.journal.count == 1


@pytest.mark.parametrize('field',['action_revision','reset_epoch'])
def test_in_call_generation_change(context: Any,field: str) -> None:
    c = context
    before = c.journal.scope()
    after = deepcopy(before)
    before['generation'][field] = before['generation'][field] or 0
    after['generation'][field] = before['generation'][field]+1
    calls = []
    def scope(*args: Any) -> Any:
        calls.append(None)
        return deepcopy(before if len(calls)==1 else after)
    c.journal.scope = scope
    c.journal.tracker = N(generation=lambda side:T.Generation(**after['generation']))
    with ExitStack() as stack:
        evidence = O.install(stack,c.journal,c.state)
        if field=='reset_epoch':
            with pytest.raises(ValueError,match='second_generation'):
                generated(c.journal,c.pipe,c.result,c.row)
            assert evidence.error is not None
        else:
            generated(c.journal,c.pipe,c.result,c.row)
            assert evidence.error is None and evidence.latest is not None
        assert c.journal.count==1
