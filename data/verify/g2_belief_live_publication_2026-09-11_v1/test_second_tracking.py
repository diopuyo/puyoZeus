"""人工frameで初回登録→次の原J採録→復元。物理反映の合格ではない。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import inspect
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_second_basis import saved, live, context, policy, observed
import second_tracking as T
import test_journal_witness as J


def subsequent(journal: Any, pipe: Any, result: Any, events: list[Any]) -> None:
    side, frame_idx = '2P', pipe._sm_2p.context.frame_idx
    time_sec = frame_idx/60
    sm = pipe._sm_2p
    signals = N(is_match_active=True, effect_gate_window_active=False)
    item = dict(frame=inspect.currentframe(), pipe=pipe, scope=journal.scope(pipe,side,frame_idx,time_sec),
                epoch=0, token='step:601', events=events, return_line=None)
    journal.complete_step(item, result, None, sys.getprofile())
    assert item['frame'] is None


def advance(o: Any) -> Any:
    journal = o.witness.journal
    scope = deepcopy(journal.scope())
    scope['frame_idx'] += T.V.FRAME_STRIDE
    scope['time_sec'] = scope['frame_idx']/60
    o.pipe._sm_2p.context.frame_idx = scope['frame_idx']
    journal.scope = lambda *args: deepcopy(scope)
    journal.tracker = N(generation=lambda side:J.Generation(**scope['generation']))
    journal.codes.add(subsequent.__code__)
    pb = o.evidence.state['hidden_probability_observer'].rows[-1]
    pb['frame_idx'], pb['time_sec'] = scope['frame_idx'], scope['time_sec']
    board = o.pipe._sm_2p.context.confirmed_board
    return N(confirmed_board=board, inferred_board=board, state=N(name='STABLE'))


def test_next_J_observed_not_physical_update(observed: Any, policy: Any) -> None:
    o = observed
    mode = T.Mode(o.evidence,o.witness,o.pipe,o.registry,o.factory,35410,policy)
    initial = o.registry.current(mode.connection.binding)
    result = advance(o)
    original = o.witness.journal.complete_step
    with ExitStack() as stack:
        T.install(stack,mode)
        subsequent(o.witness.journal,o.pipe,result,[])
        assert mode.native.last_frame == 35372 and mode.native.seen_calls == {'step:601'}
        assert not mode.native.pending and not mode.applied
        assert o.registry.current(mode.connection.binding) is initial
    assert mode.closed and o.witness.journal.complete_step == original


def test_single_consumption_remains_pending(observed: Any, policy: Any) -> None:
    o = observed
    mode = T.Mode(o.evidence,o.witness,o.pipe,o.registry,o.factory,35410,policy)
    scope = mode.connection.binding.scope
    token = f'{scope[0]}:{scope[1]}:reset:{scope[2]}:2P:enqueue:fixture'
    events = [dict(stage='fifo_before',accounting={'pending_tsumo':[[5,5]]},fifo_occurrence_tokens=[token]),
        dict(stage='fifo_after',accounting={'pending_tsumo':[]},committed=[5,5],enqueue_occurrence_token=token)]
    result = advance(o)
    with ExitStack() as stack:
        T.install(stack,mode)
        subsequent(o.witness.journal,o.pipe,result,events)
        assert len(mode.native.pending) == 1 and mode.native.pending[0].pair == (5,5)
        assert not mode.native.pending[0].physical_landing_certified and not mode.applied


def test_gap_rejected_after_original_cleanup(observed: Any, policy: Any) -> None:
    o = observed
    mode = T.Mode(o.evidence,o.witness,o.pipe,o.registry,o.factory,35410,policy)
    advance(o)
    result = advance(o)
    with ExitStack() as stack:
        T.install(stack,mode)
        with pytest.raises(ValueError,match='native_frame_bound'):
            subsequent(o.witness.journal,o.pipe,result,[])
        assert mode.error is not None and o.witness.journal.count == 3
        assert not mode.native.seen_calls
