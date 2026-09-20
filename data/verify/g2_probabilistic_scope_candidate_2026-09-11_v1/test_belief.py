"""実PBの保持と原連鎖シミュレータによる条件付け。live接続の試験ではない。"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any
import pytest
import belief as B

SCOPE=('saved-source','saved-run',3,123,456,3,'1P')
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_belief_carry_repro',ROOT.parent/'g2_hidden_probability_carry_2026-09-11_v1/reproduce.py')
R=importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def actual() -> tuple[Any,Any,Any]:
    before,_,call=R.inputs()
    board=B.Board.from_dict({'grid':before['confirmed']['grid']})
    pb=B.ProbabilisticBoard.from_board(board)
    for r in range(B.BOARD_ROWS):
        for c in range(B.BOARD_COLS): pb.set_distribution(r,c,dict(before['probability']['cells'][r][c]))
    return B.establish(SCOPE,35164,35184,board,pb),call,pb


def test_actual_prior_and_mismatched_pair() -> None:
    value,call,pb=actual()
    assert len(value.worlds)==343 and value.tokens==()
    assert R.cells(B.marginals(value))==R.cells(pb)
    observed=B.Board.from_dict({'grid':call['inputs']['current']['grid']})
    with pytest.raises(ValueError,match='pair_mismatch'):
        B.land(value,SCOPE,35172,'actual-v12-input-not-authorized',tuple(call['inputs']['previous_next_pair']),
            ((1,4,4),(2,4,3)),observed)
    assert value.tokens==() and len(value.worlds)==343
    assert not value.integer_current_permission and not value.accounting_permission


def test_counterfactual_matching_pair_keeps_all_hidden_worlds() -> None:
    value,call,pb=actual()
    observed=B.Board.from_dict({'grid':call['inputs']['current']['grid']})
    # 正常対照だけ：実NEXTは(4,5)であり、この(4,3)を実tokenへ流用しない。
    following,mass=B.land(value,SCOPE,35172,'artificial-matching-pair',(4,3),((1,4,4),(2,4,3)),observed)
    assert len(following.worlds)==343 and mass==pytest.approx(1.0)
    for c in range(B.BOARD_COLS):
        assert B.marginals(following).cell(0,c).probs==pytest.approx(pb.cell(0,c).probs)
    assert not following.source_producer_connected and not following.quality_gate_clear


def chain_prior() -> tuple[Any,Any]:
    board=B.Board()
    for r in range(1,9): board.set(r,0,2 if r%2 else 3)
    for r in range(9,13): board.set(r,0,1)
    board.set(0,0,10)
    pb=B.ProbabilisticBoard.from_board(board)
    pb.set_distribution(0,0,{4:0.3,5:0.7})
    return B.establish(SCOPE,10,30,board,pb),board


def test_chain_reveals_hidden_color_without_point_estimate() -> None:
    value,_=chain_prior()
    expected=value.worlds[0].grid
    assert expected[0][0]==4
    sim=B.ChainSimulator(exclude_hidden_row_from_pop=True)
    result=sim.simulate(B.Board.from_dict({'grid':expected}))
    assert result.chain_count==1
    following,mass=B.settle(value,SCOPE,12,'artificial-chain',1,result.final_board)
    assert len(value.worlds)==2 and len(following.worlds)==1
    assert mass==pytest.approx(0.3) and following.worlds[0].weight==1.0
    assert following.worlds[0].grid==B.grid(result.final_board)
    assert not following.integer_current_permission


def test_chain_contradiction_does_not_become_empty_success() -> None:
    value,_=chain_prior()
    with pytest.raises(ValueError,match='zero_support'):
        B.settle(value,SCOPE,12,'wrong-chain-observation',1,B.Board())
    assert len(value.worlds)==2 and value.tokens==()


@pytest.mark.parametrize('case',('scope','expired','same_frame','duplicate','bad_mass','nan'))
def test_finite_and_identity_rejections(case: str) -> None:
    value,_=chain_prior()
    scope,frame=SCOPE,12
    if case=='scope': scope=(*SCOPE[:2],4,*SCOPE[3:])
    elif case=='expired': frame=31
    elif case=='same_frame': frame=10
    elif case=='duplicate': value=replace(value,tokens=('token',))
    elif case=='bad_mass': value=replace(value,worlds=(replace(value.worlds[0],weight=0.2),))
    elif case=='nan': value=replace(value,worlds=(replace(value.worlds[0],weight=float('nan')),))
    with pytest.raises(ValueError): B.settle(value,scope,frame,'token',1,B.Board())


def test_support_limit_refuses_instead_of_pruning() -> None:
    board=B.Board()
    pb=B.ProbabilisticBoard.from_board(board)
    for c in range(B.BOARD_COLS): pb.set_distribution(0,c,{k:1/7 for k in B.PROB_COLORS})
    with pytest.raises(ValueError,match='support_limit'): B.establish(SCOPE,0,20,board,pb)
