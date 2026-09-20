"""初回限定資格と原current有り委譲。実票/SM遷移はfull試験で検査する。"""
from __future__ import annotations
from types import SimpleNamespace as S
from typing import Any
import pytest
import initial_hand_connection as A

TOKEN, SCOPE, GRID, START = 'observed:GY', ('source','run',1,'1P'), (0,)*78, 34796


def fixture() -> Any:
    pair = (4,3)
    item = S(token=TOKEN, scope=SCOPE, baseline=GRID, started=START, queue=[pair], pair=pair)
    binding = S(owner=S(state=S(current=None,history=())),candidate=item,next_token=TOKEN,
        next_started=(START,START/60),phase='historical_owned',scope=SCOPE,grid=GRID,
        current=GRID,consumed_tokens=set())
    sm, signals = S(context=S(state=S(value='stable'))), S(is_match_active=True,chain_event=None)
    return sm,signals,binding


def test_initial_hand_and_inputs_unchanged() -> None:
    sm,signals,binding = fixture()
    before = repr(vars(binding))
    assert A.should_fall(lambda *args: False,sm,signals,binding)
    assert repr(vars(binding)) == before and not binding.owner.state.history


@pytest.mark.parametrize('field,value', [('token','other'),('scope',()),('baseline',()),
    ('started',START+2),('queue',[]),('pair',(3,4))])
def test_wrong_candidate_rejected(field: str, value: Any) -> None:
    sm,signals,binding = fixture()
    setattr(binding.candidate,field,value)
    assert not A.should_fall(lambda *args: False,sm,signals,binding)


@pytest.mark.parametrize('field,value', [('next_token',None),('next_started',None),
    ('phase','foreign'),('current',()),('candidate',None),('consumed_tokens',{TOKEN})])
def test_wrong_binding_rejected(field: str, value: Any) -> None:
    sm,signals,binding = fixture()
    setattr(binding,field,value)
    assert not A.should_fall(lambda *args: False,sm,signals,binding)


@pytest.mark.parametrize('field,value', [('current',object()),('history',('completed',))])
def test_not_initial_rejected(field: str, value: Any) -> None:
    sm,signals,binding = fixture()
    setattr(binding.owner.state,field,value)
    assert not A.should_fall(lambda *args: False,sm,signals,binding)


@pytest.mark.parametrize('state', ['tsumo_fall','ojama_fall','chain','inactive'])
def test_nonstable_proposal_not_reclassified(state: str) -> None:
    sm,signals,binding = fixture()
    sm.context.state.value = state
    assert not A.should_fall(lambda *args: False,sm,signals,binding)


@pytest.mark.parametrize('field,value', [('is_match_active',False),('chain_event',object())])
def test_inactive_or_chain_rejected(field: str, value: Any) -> None:
    sm,signals,binding = fixture()
    setattr(signals,field,value)
    assert not A.should_fall(lambda *args: False,sm,signals,binding)


def test_original_true_delegation_preserved() -> None:
    assert A.should_fall(lambda *args: True,None,None,None)
