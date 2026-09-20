"""同型人工輸送で元Sへの初期化と原J caller拒否を検査する。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import replace
from typing import Any
import sys
import pytest
import fixture as F


def prepared(stack: Any) -> Any:
    item = F.fixture()
    item.provider.attach(stack,item.control)
    F.enqueue(item,F.FIRST,True)
    F.enqueue(item,F.BASELINE,False)
    item.view = F.view(item)
    return item


def test_adoption_preserves_original_fifo_and_state_with_artificial_transport() -> None:
    with ExitStack() as stack:
        item = prepared(stack)
        old_pair,old_counter,old_board = item.view.refs[0],dict(item.pipe.counter),item.sm.context.confirmed_board
        binding = item.control.baseline(item.sm,item.view,item.grid,F.SIDE)
        assert sum(binding.owner.state.counter)==55 and binding.owner.state.current is None
        assert binding.owner.state.action==1 and binding.next_started==(F.BASELINE,F.BASELINE/60)
        assert binding.next_token==item.view.tokens[0] and item.view.queue[0] is old_pair
        assert dict(item.pipe.counter)==old_counter and item.sm.context.confirmed_board is old_board
        assert not item.witness.adoptions[0]['placement_permission']
    assert item.witness.closed and 'baseline' not in vars(item.control)


@pytest.mark.parametrize('fault',['missing','token','reference','segment','reset','ambiguous','not_quiet',
    'source','run','future','discard','new_cycle','pending'])
def test_unbound_head_is_rejected(fault: str) -> None:
    with ExitStack() as stack:
        item = prepared(stack)
        if fault=='missing':
            item.witness.slots.clear()
        elif fault=='token':
            item.view = replace(item.view,tokens=('foreign',))
        elif fault=='reference':
            item.view = replace(item.view,refs=(tuple([4,3]),))
        elif fault=='segment':
            item.provider.link.latest[F.SIDE]['segment'] = 'new-segment'
        elif fault=='reset':
            item.witness.slots[F.SIDE]['scope']['generation']['reset_epoch'] = 3
        elif fault=='ambiguous':
            item.provider.link.basis = lambda view:((3,4),(5,5))
        elif fault in ('source','run'):
            item.witness.slots[F.SIDE]['scope'][fault+'_id'] = 'foreign'
        elif fault=='future':
            item.witness.slots[F.SIDE]['clock'] = item.view.clock+1
        elif fault=='discard':
            item.provider.owner(item.pipe,F.SIDE,2)['discarded_tokens'].append('discarded')
        elif fault=='new_cycle':
            item.provider.link.latest[F.SIDE]['baseline_frame'] = F.BASELINE
        elif fault=='pending':
            item.provider.link.latest[F.SIDE]['pending'] = object()
        else:
            item.view = replace(item.view,quiet=False)
        with pytest.raises(RuntimeError):
            item.control.baseline(item.sm,item.view,item.grid,F.SIDE)
        assert not item.control.history and not item.witness.adoptions


@pytest.mark.parametrize('fault',['raw_missing','nonstable','multiple'])
def test_baseline_preconditions_remain_required(fault: str) -> None:
    with ExitStack() as stack:
        item = prepared(stack)
        raw = item.grid
        if fault=='raw_missing':
            raw = None
        elif fault=='nonstable':
            item.sm.context.state = F.M.State.OJAMA_FALL
        else:
            item.view = replace(item.view,refs=item.view.refs*2,tokens=item.view.tokens*2)
        with pytest.raises(RuntimeError):
            item.control.baseline(item.sm,item.view,raw,F.SIDE)
        assert not item.control.history and not item.witness.adoptions


def test_forged_direct_emit_is_rejected() -> None:
    with ExitStack() as stack:
        item = prepared(stack)
        first = item.rec.rows[0]
        with pytest.raises(RuntimeError,match='caller'):
            item.witness.capture(first,sys._getframe())


def test_empty_fifo_keeps_original_baseline() -> None:
    with ExitStack() as stack:
        item = F.fixture()
        item.provider.attach(stack,item.control)
        binding = item.control.baseline(item.sm,item.view,item.grid,F.SIDE)
        assert binding.owner.state.action==0 and binding.next_token is None
        assert not item.witness.adoptions


def test_outside_native_scope_never_creates_adoption_witness() -> None:
    with ExitStack() as stack:
        item = F.fixture()
        item.provider.link.adapter.native.FIRST_FRAME = F.BASELINE
        item.provider.attach(stack,item.control)
        F.enqueue(item,F.FIRST,True)
        assert len(item.view.queue)==1 and not item.witness.slots
        assert item.witness.error is None and len(item.witness.unwitnessed_appends)==1
        F.enqueue(item,F.BASELINE,False)
        with pytest.raises(RuntimeError,match='unwitnessed'):
            item.control.baseline(item.sm,F.view(item),item.grid,F.SIDE)


def test_exception_receipt_does_not_replace_primary_error() -> None:
    with ExitStack() as stack:
        item = F.fixture()
        item.provider.attach(stack,item.control)
        item.provider.link.adapter.enabled = False
        with pytest.raises(RuntimeError,match='adapter_disabled'):
            F.enqueue(item,F.FIRST,True)
        assert 'adapter_disabled' in item.witness.error
        assert 'adoption_witness_state' not in item.witness.error
        assert item.rec.rows[-1]['status']=='exception'


@pytest.mark.parametrize('armed',[None,F.FIRST+2,F.BASELINE])
def test_quiet_rearming_preserves_same_unresolved_slot(armed: Any) -> None:
    with ExitStack() as stack:
        item = prepared(stack)
        item.provider.link.latest[F.SIDE]['armed_at'] = armed
        pair,counter,board = item.view.refs[0],dict(item.pipe.counter),item.sm.context.confirmed_board
        binding = item.control.baseline(item.sm,item.view,item.grid,F.SIDE)
        assert binding.owner.state.current is None and sum(binding.owner.state.counter)==55
        assert item.witness.adoptions[0]['waiting_rearmed_at']==armed
        assert item.view.queue[0] is pair and dict(item.pipe.counter)==counter
        assert item.sm.context.confirmed_board is board


@pytest.mark.parametrize('fault',['arm_bool','arm_float','arm_old','arm_future','event',
    'reason','committed'])
def test_rearming_never_allows_changed_episode_or_invalid_clock(fault: str) -> None:
    with ExitStack() as stack:
        item = prepared(stack)
        live = item.provider.link.latest[F.SIDE]
        if fault.startswith('arm_'):
            live['armed_at'] = dict(arm_bool=True,arm_float=float(F.FIRST+2),
                arm_old=F.FIRST,arm_future=F.BASELINE+2)[fault]
        elif fault=='event':
            live['event'] = replace(live['event'],sequence_number=40)
        elif fault=='reason':
            live['reason'] = 'await_fresh_quiet'
        else:
            live['committed'] = True
        with pytest.raises(RuntimeError):
            item.control.baseline(item.sm,item.view,item.grid,F.SIDE)
        assert not item.control.history and not item.witness.adoptions
