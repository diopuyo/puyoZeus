"""後続票のscope/状態/観測と時計の連続性。元J陽性は全接続で別検収。"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'g2_hidden_two_hand_candidate_2026-09-10_v1'))
import tail_witness as T


def fixture() -> tuple[Any,...]:
    state = object()
    binding = N(owner=N(state=state))
    head,queue = (2,2),[]
    item = N(pair=head,token='BB')
    provider = N(adjacent=lambda old,view:old[0]+2==view.frame and view.clock==view.frame/60)
    view = N(scope=('source','run',2,'1P'),queue=queue,frame=34852,clock=34852/60)
    return N(provider=provider),binding,item,view


def test_two_fresh_votes() -> None:
    control,binding,item,view = fixture()
    first = T.vote(control,binding,item,'final','raw',{},view)
    view.frame,view.clock = 34854,34854/60
    second = T.vote(control,binding,item,'final','raw',{},view)
    assert first.count==1 and second.count==2 and second.first==(34852,34852/60)


@pytest.mark.parametrize('case',('state','item','scope','queue','raw','final','gap','duplicate'))
def test_changed_evidence_restarts(case: str) -> None:
    control,binding,item,view = fixture()
    T.vote(control,binding,item,'final','raw',{},view)
    view.frame,view.clock = 34854,34854/60
    final,raw = 'final','raw'
    if case=='state': binding.owner.state=object()
    if case=='item': item=N(pair=item.pair,token='BB')
    if case=='scope': view.scope=('source','other',2,'1P')
    if case=='queue': view.queue=[]
    if case=='raw': raw='other'
    if case=='final': final='other'
    if case=='gap': view.frame,view.clock=34856,34856/60
    if case=='duplicate': view.frame,view.clock=34852,34852/60
    result = T.vote(control,binding,item,final,raw,{},view)
    assert result.count==1 and result.first==(view.frame,view.clock)
