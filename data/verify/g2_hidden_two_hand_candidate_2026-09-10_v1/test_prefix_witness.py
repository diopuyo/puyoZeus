"""候補の二票は同Binding/同holder/同支持かつ隣接時刻だけで累積する。"""
from __future__ import annotations
from types import SimpleNamespace
import pytest
import prefix_witness as W
from test_path_support import inputs,CORE,P


def test_adjacent_votes_reset_on_gap_or_holder() -> None:
    before,raw = inputs()
    support = P.infer(CORE,before,(5,4),(2,2),raw)[0]
    control = SimpleNamespace(provider=SimpleNamespace(adjacent=lambda old,v:v.frame==old[0]+2))
    binding,held = SimpleNamespace(),object()
    view = lambda f:SimpleNamespace(frame=f,clock=f/60)
    assert W.vote(control,binding,held,support,view(35052)).count==1
    assert W.vote(control,binding,held,support,view(35054)).count==2
    assert W.vote(control,binding,held,support,view(35058)).count==1
    assert W.vote(control,binding,object(),support,view(35060)).count==1
    W.clear(binding)
    assert binding.hidden_prefix_votes is None


def test_support_cannot_request_authority() -> None:
    before,raw = inputs()
    support = P.infer(CORE,before,(5,4),(2,2),raw)[0]
    with pytest.raises(TypeError):
        P.Support(support.prefix,support.final,support.raw,support.pairs,current_permission=True)
