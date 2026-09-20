"""2P reset原J成功→旧binding退役→新scope登録。原世代は人工駆動。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_second_physical import saved,live,context,policy,observed,physical,setup,step,consumption
from test_second_tracking import advance
import test_journal_witness as J
import second_tracking as T
import second_basis as BASIS


def first_side(o: Any,token: str) -> None:
    journal=o.witness.journal
    scope=dict(journal.scope(),side='1P')
    item=dict(scope=scope,token=token,epoch=0,events=[],return_line=None,frame=None)
    journal.complete_step(item,None,None,sys.getprofile())


@pytest.mark.parametrize('pending',[False,True])
def test_retire_preserve_and_rebind(observed: Any,policy: Any,physical: Any,pending: bool) -> None:
    o=observed
    mode=setup(o,policy,physical)
    binding=mode.connection.binding
    old=o.registry.current(binding)
    board=o.pipe._sm_2p.context.confirmed_board.copy()
    with ExitStack() as stack:
        T.install(stack,mode)
        if pending:
            advance(o)
            first_side(o,'step:600')
            step(o,board,'step:601',consumption(mode))
        advance(o)
        scope=deepcopy(o.witness.journal.scope())
        scope['generation']['reset_epoch']+=1
        o.witness.journal.scope=lambda *args:deepcopy(scope)
        o.witness.journal.tracker=N(generation=lambda side:J.Generation(**scope['generation']))
        first_side(o,'step:602')
        step(o,board,'step:603',[])
        assert mode.closed and mode.retired_receipt['retired'] and mode.error is None
        assert len(mode.retired_receipt['pending'])==int(pending)
        assert len(mode.native.pending)==int(pending) and not mode.retired_receipt['pending_discarded']
        with pytest.raises(ValueError,match='registry_stale_binding'):
            o.registry.current(binding)
        following=T.Mode(o.evidence,o.witness,o.pipe,o.registry,o.factory,35410,policy)
        value=o.registry.current(following.connection.binding)
        assert value.scope[5]==old.scope[5]+1 and value.frame>old.frame
        assert not following.native.pending and not following.applied
        # 退役済みhookは以後の原callを通す。新hookの下で旧scopeを再採録しない。
        advance(o)
        first_side(o,'step:604')
        with ExitStack() as inner:
            T.install(inner,following)
            step(o,board,'step:605',[])
            assert following.native.last_frame==o.pipe._sm_2p.context.frame_idx


@pytest.mark.parametrize('software',[False,True])
def test_reset_inside_original_call_holds_boundary(observed: Any,policy: Any,physical: Any,software: bool) -> None:
    o=observed
    mode=setup(o,policy,physical)
    advance(o)
    first_side(o,'step:600')
    journal=o.witness.journal
    before=deepcopy(journal.scope())
    after=deepcopy(before)
    if not software: after['generation']['reset_epoch']+=1
    calls=[]
    def scope(*args: Any) -> Any:
        calls.append(None)
        return deepcopy(before if len(calls)==1 else after)
    journal.scope=scope
    journal.epoch=lambda *args:int(software and len(calls)>1)
    journal.tracker=N(generation=lambda side:J.Generation(**after['generation']))
    board=o.pipe._sm_2p.context.confirmed_board.copy()
    with ExitStack() as stack:
        T.install(stack,mode)
        step(o,board,'step:601',[])
        assert mode.closed and mode.retired_receipt['retired'] and mode.error is None
        assert o.evidence.latest['hold_reason']=='generation_changed'
        with pytest.raises(BASIS.BasisHold):
            T.Mode(o.evidence,o.witness,o.pipe,o.registry,o.factory,35410,policy)


def test_retirement_call_consumption_is_preserved(observed: Any,policy: Any,physical: Any) -> None:
    o=observed
    mode=setup(o,policy,physical)
    advance(o)
    scope=deepcopy(o.witness.journal.scope())
    scope['generation']['reset_epoch']+=1
    journal=o.witness.journal
    journal.scope=lambda *args:deepcopy(scope)
    journal.tracker=N(generation=lambda side:J.Generation(**scope['generation']))
    events=consumption(mode)
    with ExitStack() as stack:
        T.install(stack,mode)
        first_side(o,'step:600')
        step(o,o.pipe._sm_2p.context.confirmed_board.copy(),'step:601',events)
        receipt=mode.retired_receipt
        assert receipt['reset_call_consumption']['occurrence_token']==events[1]['enqueue_occurrence_token']
        assert receipt['reset_call_consumption']['source_call_token']=='step:601'
        assert receipt['reset_call_consumption']['pair']==(5,5)
        assert receipt['reset_call_attribution']=='UNATTRIBUTED'
        assert not receipt['reset_call_accounting_permission'] and not receipt['pending_discarded']
