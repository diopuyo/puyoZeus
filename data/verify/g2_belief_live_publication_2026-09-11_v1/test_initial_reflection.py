"""原Jを観測した未変化2Pだけに初回票の利用を許す。人工pipeline fixture。"""
from __future__ import annotations

from contextlib import ExitStack
from typing import Any
import pytest
from test_second_physical import saved,live,context,policy,observed,physical,setup,step
from test_second_tracking import advance
import second_tracking as T
import reflection as OLD
import reflection_v3 as R


@pytest.fixture
def unchanged(observed: Any,policy: Any,physical: Any) -> Any:
    o=observed
    mode=setup(o,policy,physical)
    value=o.registry.current(mode.connection.binding)
    advance(o)
    with ExitStack() as stack:
        T.install(stack,mode)
        step(o,o.pipe._sm_2p.context.confirmed_board.copy(),'step:601',[])
        yield mode,value


def test_initial_real_receipt_after_original_J(unchanged: Any) -> None:
    mode,value=unchanged
    with pytest.raises(ValueError,match='reflection_transition_evidence_missing'):
        OLD.verify(mode,value,'step:601',35372)
    R.verify(mode,value,'step:601',35372)
    assert not mode.applied and value.frame==35370


@pytest.mark.parametrize('case',['closed','unseen','pending','origin','receipt','owner'])
def test_invalid_initial_reflection_rejected(unchanged: Any,case: str) -> None:
    mode,value=unchanged
    if case=='closed': mode.closed=True
    elif case=='unseen': mode.native.seen_calls.clear()
    elif case=='pending': mode.native.pending.append(object())
    elif case=='origin': mode.physical.origins['unsettled']={}
    elif case=='receipt': mode.initial_receipt['source_call_token']='foreign'
    else: mode.physical.native=object()
    with pytest.raises(ValueError):
        R.verify(mode,value,'step:601',35372)
