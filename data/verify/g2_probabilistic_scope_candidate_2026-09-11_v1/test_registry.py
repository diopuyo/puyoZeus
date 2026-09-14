"""同一factory/scopeの二重所有と旧state再採用を拒否する。実J認証試験ではない。"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pytest
import belief as B
from registry import Registry,Binding
from test_belief import chain_prior,SCOPE


def fixture() -> tuple:
    factory=object()
    registry=Registry(factory)
    value,_=chain_prior()
    binding=registry.bind(factory,value,'baseline-call')
    observed=B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(B.Board.from_dict({'grid':value.worlds[0].grid})).final_board
    operation=lambda v:B.settle(v,SCOPE,12,'chain-source',1,observed)
    return factory,registry,value,binding,operation


def test_duplicate_binding_and_forged_anchor_rejected() -> None:
    factory,r,value,binding,_=fixture()
    with pytest.raises(ValueError,match='duplicate_scope'): r.bind(factory,value,'second-owner')
    with pytest.raises(ValueError,match='stale_binding'): r.current(Binding(binding.scope,binding.initial_call_token))
    assert r.current(binding) is value


def test_direct_math_cannot_replace_registered_state() -> None:
    factory,r,value,binding,operation=fixture()
    fork,_=operation(value)
    assert r.current(binding) is value
    with pytest.raises(ValueError,match='stale_state'):
        r.transition(factory,binding,fork,'same-call',operation)


def test_concurrent_single_winner() -> None:
    factory,r,value,binding,operation=fixture()
    def action(_: int) -> str:
        try:
            r.transition(factory,binding,value,'source-call',operation)
            return 'accepted'
        except ValueError as error:
            assert 'stale_state' in str(error)
            return 'stale'
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(action,range(8)))
    assert results.count('accepted')==1 and results.count('stale')==7
    assert not r.current(binding).accounting_permission


def test_failure_retains_state_and_call_can_be_retried() -> None:
    factory,r,value,binding,operation=fixture()
    with pytest.raises(ValueError,match='zero_support'):
        r.transition(factory,binding,value,'source-call',lambda v:B.settle(v,SCOPE,12,'chain-source',1,B.Board()))
    assert r.current(binding) is value
    r.transition(factory,binding,value,'source-call',operation)


def test_retired_reference_and_scope_reuse_rejected() -> None:
    factory,r,value,binding,_=fixture()
    assert r.retire(factory,binding) is value
    with pytest.raises(ValueError,match='stale_binding'): r.current(binding)
    with pytest.raises(ValueError,match='duplicate_scope'): r.bind(factory,value,'old-scope')
    new=replace(value,scope=(*SCOPE[:2],4,*SCOPE[3:]))
    fresh=r.bind(factory,new,'new-scope')
    assert r.current(fresh) is new
