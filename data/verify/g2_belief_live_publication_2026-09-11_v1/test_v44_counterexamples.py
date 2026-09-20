"""Fable v44の仮説を原native/物理処理で再現。人工frameの限定反例。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from typing import Any
import pytest
from test_second_physical import saved,live,context,policy,observed,physical,setup,step,consumption
from test_second_tracking import advance
import second_tracking as T
import reflection as R


@pytest.mark.parametrize('next_hand',[False,True])
def test_zero_support_retention_and_next_hand(observed: Any,policy: Any,physical: Any,next_hand: bool) -> None:
    o=observed
    mode=setup(o,policy,physical)
    initial=o.registry.current(mode.connection.binding)
    empty=o.pipe._sm_2p.context.confirmed_board.copy()
    correct=empty.copy()
    correct.set(12,0,5)
    correct.set(11,0,5)
    advance(o)
    with ExitStack() as stack:
        T.install(stack,mode)
        step(o,empty,'step:601',consumption(mode))
        assert mode.latest['reason']=='stable_observation_zero_support'
        assert len(mode.native.pending)==1 and o.registry.current(mode.connection.binding) is initial
        with pytest.raises(ValueError,match='reflection_pending'):
            R.verify(mode,initial,'step:601',35372)
        advance(o)
        if next_hand:
            events=deepcopy(consumption(mode))
            events[0]['fifo_occurrence_tokens'][0]+=':second'
            events[1]['enqueue_occurrence_token']+=':second'
            with pytest.raises(ValueError,match='physical_pending_ambiguous'):
                step(o,empty,'step:603',events)
            assert len(mode.native.pending)==2 and mode.error is not None
            assert o.registry.current(mode.connection.binding) is initial and not mode.applied
        else:
            step(o,correct,'step:603',[])
            assert len(mode.applied)==1 and not mode.native.pending


def test_reset_after_registration_stops_old_scope(observed: Any,policy: Any,physical: Any,monkeypatch: Any) -> None:
    # 退役分岐追加前の経路を維持した反例。修復版は別の退役テストで検収する。
    import second_retirement
    monkeypatch.setattr(second_retirement,'prepare',lambda *args:None)
    o=observed
    mode=setup(o,policy,physical)
    advance(o)
    scope=deepcopy(o.witness.journal.scope())
    scope['generation']['reset_epoch']+=1
    o.witness.journal.scope=lambda *args:deepcopy(scope)
    board=o.pipe._sm_2p.context.confirmed_board.copy()
    with ExitStack() as stack:
        T.install(stack,mode)
        with pytest.raises(ValueError,match='native_live_scope'):
            step(o,board,'step:601',[])
        assert mode.error is not None and not mode.applied
