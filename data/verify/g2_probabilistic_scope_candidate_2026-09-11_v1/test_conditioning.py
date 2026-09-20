"""不可能候補だけを重力で除き、除外質量を隠さない対照。"""
import pytest
import belief as B
from conditioning import establish_conditioned
from test_belief import SCOPE,actual,chain_prior


def test_partial_impossible_mass_is_explicit() -> None:
    _,board=chain_prior()
    pb=B.ProbabilisticBoard.from_board(board)
    pb.set_distribution(0,0,{4:0.3,5:0.7})
    pb.set_distribution(0,1,{0:0.5,3:0.5})
    state,mass,removed=establish_conditioned(SCOPE,10,30,board,pb)
    assert mass==pytest.approx(0.5) and removed==2 and len(state.worlds)==2
    assert B.marginals(state).cell(0,0).probs==pytest.approx({4:0.3,5:0.7})
    assert B.marginals(state).cell(0,1).probs=={0:1.0}
    assert pb.cell(0,1).probs=={0:0.5,3:0.5} and not state.physical_certified


def test_zero_support_rejected() -> None:
    board=B.Board()
    board.set(5,0,1)
    with pytest.raises(ValueError,match='gravity_has_zero_support'):
        establish_conditioned(SCOPE,0,20,board,B.ProbabilisticBoard.from_board(board))


def test_real_343_no_world_removed() -> None:
    previous,_,pb=actual()
    board=B.Board.from_dict({'grid':previous.worlds[0].grid})
    result,mass,removed=establish_conditioned(SCOPE,35164,35184,board,pb)
    assert len(result.worlds)==343 and mass==pytest.approx(1.0) and removed==0
    for old,new in zip(previous.worlds,result.worlds):
        assert old.grid==new.grid and old.weight==pytest.approx(new.weight)


def test_overflow_does_not_use_gravity_to_hide_resource_cap() -> None:
    board=B.Board()
    pb=B.ProbabilisticBoard.from_board(board)
    for c in range(6): pb.set_distribution(0,c,{k:1/7 for k in B.PROB_COLORS})
    with pytest.raises(ValueError,match='support_limit'):
        establish_conditioned(SCOPE,0,20,board,pb)
