"""元PB保存fixtureで30fpsの捕捉/同step resetを確認。人工再配置でありGTでない。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace as N
from typing import Any
import pytest
from scripts import g3_video39_early_clock as G
from tests.test_g3_reset_scope import context, live, saved, generated, T
from tests.test_g3_video39_early_clock import ASYNC, PUB, replace

FRAME = 100


@pytest.mark.parametrize('reset', [False, True])
def test_probability_clock(context: Any, reset: bool) -> None:
    c = context
    before = c.journal.scope()
    before.update(frame_idx=FRAME, time_sec=FRAME / 30)
    after = deepcopy(before)
    if reset:
        after['generation']['reset_epoch'] += 1
    calls = []
    def scope(*args: Any) -> dict:
        calls.append(None)
        return deepcopy(before if len(calls) == 1 else after)
    c.journal.scope = scope
    c.journal.tracker = N(generation=lambda side: T.Generation(**after['generation']))
    c.row = dict(c.row, frame_idx=FRAME, time_sec=FRAME / 30)
    c.pipe._sm_2p.context.frame_idx = 0 if reset else FRAME
    c.state['hidden_probability_observer'].rows[0].update(frame_idx=FRAME, time_sec=FRAME / 30)
    with ExitStack() as stack:
        reader = stack.enter_context(G.G.P.loaded(ASYNC / 'journal_pair_reader.py', G.READER_SHA))
        history = stack.enter_context(G.G.P.loaded(ASYNC / 'early_origin_history.py', G.HISTORY_SHA))
        observation = stack.enter_context(G.G.P.loaded(PUB / 'second_observation.py', G.R.SOURCE_SHA))
        original = observation.capture
        proof = G.prepare(stack, reader, history, observation, replace)
        assert [row['divisions_changed'] for row in proof['functions']] == [1, 1, 2]
        evidence = observation.install(stack, c.journal, c.state)
        generated(c.journal, c.pipe, c.result, c.row)
        value = evidence.latest
        assert value['frame'] == FRAME and not value['basis_registered']
        if reset:
            assert value['hold_reason'] == 'generation_changed' and value['observed_sm_frame'] == 0
        else:
            assert value['clock'] == FRAME / 30 and value['state'] == 'stable'
        assert c.journal.count == 1
    assert observation.capture is original and evidence.closed
