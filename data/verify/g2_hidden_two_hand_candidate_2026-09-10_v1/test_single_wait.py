"""一枠を二枠の証拠と誤認しない限定待機。原caller陽性の代用ではない。"""
from __future__ import annotations
from collections import deque
from types import SimpleNamespace as N
from typing import Any
import pytest
from prefix_connection import single_head_wait


def fixture() -> tuple[Any,...]:
    head = (5,4)
    queue,scope = deque([head]),('source','run',2,'1P')
    item = N(token='head',pair=head,queue=queue)
    state = N(history=(),current=None)
    binding = N(owner=N(state=state),scope=scope,next_token='head')
    view = N(refs=(head,),queue=queue,tokens=('head',),scope=scope)
    control = N(provider=N(handoff_proofs={}))
    return control,binding,item,view


def test_initial_and_prefix_tail_wait_only() -> None:
    control,binding,item,view = fixture()
    assert single_head_wait(control,binding,item,view)
    binding.owner.state.history = ('placement',)
    assert not single_head_wait(control,binding,item,view)
    binding.hidden_prefix_consumed = True
    assert single_head_wait(control,binding,item,view)
    assert len(view.queue)==1 and binding.next_token=='head'


@pytest.mark.parametrize('case',('token','scope','holder','reference','queue_length'))
def test_inconsistent_single_rejected(case: str) -> None:
    control,binding,item,view = fixture()
    if case=='token': view.tokens=('other',)
    if case=='scope': view.scope=('other','run',2,'1P')
    if case=='holder': control.provider.handoff_proofs['1P']=object()
    if case=='reference': item.pair=tuple([5,4])
    if case=='queue_length': view.queue.append((2,2))
    with pytest.raises(ValueError): single_head_wait(control,binding,item,view)
