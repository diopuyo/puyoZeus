"""retRのIだけが撃った盤の射影へ替わることを実連鎖計算で検査する。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
from src.phase_j._live_ex_frozen.src.dl1k_engine import ExEngine
from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import ConnectedLedger
from tests.test_live_fire_board import boards


def fixture() -> SimpleNamespace:
    native = pytest.importorskip('puyo_core')
    _, fired = boards()
    receiver = np.zeros((13, 6), dtype=np.int8)
    receiver[12, :4] = 1
    state = SimpleNamespace(t=1.01, game_idx=1, game_start_t=0., first_move_t=lambda: 0.,
        sides=[SimpleNamespace(board=fired), SimpleNamespace(board=receiver)], sl=ConnectedLedger(),
        sl_snap=dict(margin_origin_sec=0., generated_p1=0, generated_p2=0))
    chain = dict(started=1., finished=False, best=dict(scores=[100]), observed=[dict(count=1)],
                 carry_at_fire=0, generated_at_fire=0)
    return SimpleNamespace(native=native, state=state, chains=[chain, None],
        _resolving=lambda side: side == 0, observe=lambda *args: None)


@pytest.mark.parametrize('enabled', [False, True])
def test_engine_step_gates_tracker_and_I(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """前の FireBoardTracker の扱い (PUYO_RETR_FORECAST=0)。依頼 A の識別子で選ぶ扱いは tests/test_live_forecast.py。"""
    monkeypatch.setenv('PUYO_RETR_FORECAST', '0')
    monkeypatch.setenv('PUYO_EX_RETR', str(int(enabled)))
    engine = fixture()
    _, fired = boards()
    raw = [dict(raw_cnn_board=fired.tolist(), formula=dict(valid=False)), {}]
    ExEngine.step(engine, .9, 1, raw)
    raw[0]['formula'] = dict(valid=True, product=100)
    ExEngine.step(engine, 1.01, 1, raw)
    assert hasattr(engine, 'retr_fire') is enabled
    if not enabled:
        return
    value = dict(pending=[0, 1], rate=70, board_prediction_active=[True, False],
                 reconciliation=[dict(times=[1.5]), None], boards=[np.zeros((13, 6)).tolist(), None])
    before = deepcopy((value, engine.chains, engine.state.sl_snap))
    extra, why = retr.compute_retr(engine, value)
    assert engine.retr_diag[1]['I'] == 6
    assert engine.retr_diag[1]['I_pred'] == 1
    assert engine.retr_fire.found[0].steps == (100, 320)
    assert extra['retR_r_p2'] == 6 and why[1] == 'board_chain'
    assert (value, engine.chains, engine.state.sl_snap) == before
