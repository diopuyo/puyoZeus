"""独立レビューの反例閉鎖。実factoryへの接続や認定は含まない。"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import belief as B
from owner import Owner
from test_belief import SCOPE, actual, chain_prior


def test_floating_initial_board_rejected() -> None:
    board=B.Board()
    board.set(5,0,1)
    with pytest.raises(ValueError,match='world_gravity'):
        B.establish(SCOPE,0,20,board,B.ProbabilisticBoard.from_board(board))


@pytest.mark.parametrize('dtype',(float,bool))
def test_noninteger_board_rejected(dtype: type) -> None:
    board=B.Board()
    board._grid=np.zeros((13,6),dtype=dtype)
    with pytest.raises(ValueError,match='board_array_type'): B.grid(board)


def test_real_support_remains_gravity_consistent() -> None:
    value,_,_=actual()
    assert len(value.worlds)==343 and all(B.supported(w.grid) for w in value.worlds)


def test_hidden_placement_not_independent_evidence() -> None:
    value,_=chain_prior()
    with pytest.raises(ValueError,match='hidden_placement_requires_independent_support'):
        B.land(value,SCOPE,12,'hidden',(4,5),((0,0,4),(0,1,5)),B.Board())


def test_owner_failure_preserves_identity_and_concurrent_once() -> None:
    value,_=chain_prior()
    owner=Owner(value)
    with pytest.raises(ValueError,match='zero_support'):
        owner.settle(value,SCOPE,12,'bad',1,B.Board())
    assert owner.state is value
    observed=B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(
        B.Board.from_dict({'grid':value.worlds[0].grid})).final_board
    def attempt(_: int) -> str:
        try:
            owner.settle(value,SCOPE,12,'one',1,observed)
            return 'accepted'
        except ValueError as error:
            assert 'owner_stale_reference' in str(error)
            return 'stale'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt,range(2)))==['accepted','stale']
    assert owner.state.tokens==('one',) and value.tokens==()


def test_scope_precedes_enumeration() -> None:
    with pytest.raises(ValueError,match='scope_values'):
        B.establish(('',*SCOPE[1:]),0,20,None,None)


def test_frozen_physics_path() -> None:
    import frozen_physics as F
    import inspect
    assert '.runtime_snapshots' in inspect.getfile(B.ChainSimulator)
    assert F.SOURCE.is_file()
