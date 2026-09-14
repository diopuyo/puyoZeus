"""独立cellへ戻すと壊れる相関を、モデルへ渡す盤面batchで検査する。"""
from dataclasses import replace
import numpy as np
import pytest
import belief as B
import joint_evaluation as J
from test_belief import SCOPE,chain_prior


def states() -> tuple:
    value,_=chain_prior()
    grids=[]
    for index,world in enumerate(value.worlds):
        board=B.Board.from_dict({'grid':world.grid})
        for r in range(1,13): board.set(r,1,2 if r%2 else 3)
        board.set(0,1,5 if index==0 else 4)
        grids.append(B.World(B.grid(board),world.weight))
    p1=replace(value,worlds=tuple(grids))
    p2=B.establish((*SCOPE[:-1],'2P'),10,30,B.Board(),B.ProbabilisticBoard.from_board(B.Board()))
    return (p1,p2),(B.Board.from_dict({'grid':p1.worlds[0].grid}),B.Board())


def test_joint_correlation_reaches_callback() -> None:
    values,observed=states()
    calls=[]
    def scorer(batch: np.ndarray) -> np.ndarray:
        assert batch.shape==(256,2,13,6) and not batch.flags.writeable
        assert np.all(batch[:,0,0,0]!=batch[:,0,0,1])
        calls.append(batch.copy())
        return np.full(len(batch),0.7)
    result=J.evaluate(values,12,('STABLE','STABLE'),observed,scorer)
    assert len(calls)==1 and result.probability_p1==pytest.approx(0.7)
    assert result.provisional and not result.quality_gate_clear and not result.accounting_permission


@pytest.mark.parametrize('case',('nonstable','stale_visible','scope','expired','bad_score','nan_score'))
def test_bad_context_or_output(case: str) -> None:
    values,observed=states()
    frame,phase=12,('STABLE','STABLE')
    scorer=lambda batch:np.zeros(len(batch))
    if case=='nonstable': phase=('STABLE','CHAIN')
    elif case=='stale_visible': observed=(B.Board(),observed[1])
    elif case=='scope': values=(values[0],replace(values[1],scope=('other',*values[1].scope[1:])))
    elif case=='expired': frame=31
    elif case=='bad_score': scorer=lambda batch:np.full(len(batch),2.0)
    elif case=='nan_score': scorer=lambda batch:np.full(len(batch),float('nan'))
    with pytest.raises(ValueError): J.evaluate(values,frame,phase,observed,scorer)
