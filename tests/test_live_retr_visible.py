"""見えている手の返し列と配信の保持を確認する。"""
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
from src.phase_j.live_ex_hold import LiveExHold
from tests.test_live_24h_hold import source, event


def setup_case(monkeypatch: pytest.MonkeyPatch, busy: bool = False) -> tuple:
    board = np.zeros((13, 6), dtype=np.int64)
    engine = SimpleNamespace(native=None, state=SimpleNamespace(t=0., game_idx=1,
        sides=[SimpleNamespace(board=board.copy()) for _ in retr.SIDES]),
        chains=[None, None], _resolving=lambda s: busy,
        retr_ama=SimpleNamespace(profile=lambda *args: [70, 140, 210, 700]+[700]*12),
        retr_pre={}, retr_seed={})
    value = dict(pending=[100, 200], rate=70, board_prediction_active=[True, True],
                 boards=[board.copy(), board.copy()])
    monkeypatch.setattr(retr, 'chain_end', lambda *args: 10.)
    monkeypatch.setattr(retr, 'chain_score', lambda *args: (0, 0))
    monkeypatch.setattr(retr, 'seed_pairs', lambda *args: ([[1, 2]]*16, 'ok'))
    monkeypatch.setenv(retr.H_FORECAST_ENV, '0')
    monkeypatch.setenv(retr.DEADLINE_EXTEND_ENV, '0')
    return engine, value


@pytest.mark.parametrize('busy,v', [(False, 3), (True, 2)])
@pytest.mark.parametrize('h', [0, 1, 2, 3, 4, 16])
def test_visible_definition(busy: bool, v: int, h: int) -> None:
    cm = [70, 140, 210, 700]+[700]*12
    C = cm[h-1]//70 if h else 0
    cv = cm[min(h, v)-1]//70 if h else 0
    result = retr.visible_values(C, 100, h, busy, np.zeros((13, 6)), cm, 70)
    assert result == (cv, 100-cv, C-cv, cv/C if C else 1., (100-cv)/72, (100-C)/72)


@pytest.mark.parametrize('I,expected', [(0, 0.), (5, 0.), (6, 10.), (1000, 10.)])
def test_room_zero(I: int, expected: float) -> None:
    result = retr.visible_values(5., I, 1, False, np.ones((13, 6)), None, 70)
    assert result[-2:] == (expected, expected)


def test_off_bits_and_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(retr.VISIBLE_ENV, raising=False)
    engine, value = setup_case(monkeypatch)
    baseline, why = retr.compute_retr(engine, value)
    monkeypatch.setenv(retr.VISIBLE_ENV, '0')
    off, off_why = retr.compute_retr(engine, value)
    assert pickle.dumps((off, off_why)) == pickle.dumps((baseline, why))
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    on, on_why = retr.compute_retr(engine, value)
    assert pickle.dumps(({k: on[k] for k in off}, on_why)) == pickle.dumps((off, off_why))
    for side, I in zip(retr.NAMES, (100, 200)):
        assert on[f'retR_cvis_{side}'] == 1.
        assert on[f'retR_rvis_{side}'] == I-1
    swapped = retr.swap(on)
    assert swapped['retR_rvis_p1'] == on['retR_rvis_p2']
    assert retr.swap(swapped) == on


def test_busy_carry_and_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    engine, value = setup_case(monkeypatch)
    initial, _ = retr.compute_retr(engine, value)
    engine._resolving = lambda s: True
    value['board_prediction_active'] = [False, False]
    carried, why = retr.compute_retr(engine, value)
    assert carried == initial and why == ['carry', 'carry']
    engine.state.game_idx = 2
    empty, _ = retr.compute_retr(engine, value)
    assert all(math.isnan(empty[c]) for c in retr.COLUMNS+retr.VISIBLE_COLUMNS)
    value['board_prediction_active'] = [True, True]
    monkeypatch.setattr(retr, 'chain_end', lambda e, v, s, n: 0. if s == 0 else 10.)
    active, _ = retr.compute_retr(engine, value)
    assert active['retR_cvis_p1'] == 2.
    assert active['retR_cvis_p2'] == 0.


def test_hold_columns() -> None:
    hold, src = LiveExHold(), source()
    src['extra'].update(dict.fromkeys(retr.VISIBLE_COLUMNS, 7.))
    adopted = hold.adopt(event(.2), src, 'ok')
    assert all(adopted[c] == 7. for c in retr.VISIBLE_COLUMNS)
    reset = hold.adopt(event(.4, reset={'p1': True, 'p2': False}), None, 'stale')
    assert math.isnan(reset['retR_cvis_p1']) and reset['retR_cvis_p2'] == 7.


@pytest.mark.parametrize('reason', ['no_attack', 'no_seed', 'no_board'])
def test_nan_matches_c(monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    engine, value = setup_case(monkeypatch)
    if reason == 'no_attack':
        value['pending'] = [0, 0]
    elif reason == 'no_seed':
        monkeypatch.setattr(retr, 'seed_pairs', lambda *args: (None, 'no_seed'))
    else:
        for side in engine.state.sides:
            side.board = None
    extra, _ = retr.compute_retr(engine, value)
    assert all(math.isnan(extra[c]) for c in retr.VISIBLE_COLUMNS)
    assert all(math.isnan(extra[f'retR_c_{s}']) for s in retr.NAMES)


def test_room_cap_and_board_chain() -> None:
    board = np.zeros((13, 6))
    board[2, 2] = 1  # 高さ11、空き6個。セル数でなく既存の高さを使う。
    values = retr.visible_values(10., 100, 1, False, board, None, 70)
    assert values == (10., 90., 0., 1., 10., 10.)


def test_prediction_keeps_visible_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json
    from src.phase_j.live_24h import Live24h
    from tests.test_live_24h_model import Worker, extra, model, result
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    calls = []
    values = extra(**dict.fromkeys(retr.VISIBLE_COLUMNS, 7.))
    worker = Worker(values)
    owner = Live24h(tmp_path, model(calls), {}, lambda _: worker, use_ama=True,
                    temperature_enabled=False, counter_fire_shift_enabled=False,
                    midgame_board_layer_enabled=False)
    assert owner.predict(result()) == .8
    owner.close()
    rows = [json.loads(line) for line in (tmp_path/'live_24h.jsonl').read_text().splitlines()]
    prediction = next(row for row in rows if row['kind'] == 'prediction')
    assert all(prediction['learning'][c] == 7. for c in retr.VISIBLE_COLUMNS)


def test_row_predictor_uses_saved_columns_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """保存された列だけを取り、同一辞書の旗ON/OFFで予測がビット一致する。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from src.b3_predict import RowPredictor
    columns = ('retR_c_p1', 'retR_r_p1')
    X = np.array([[0., 100.], [1., 99.], [10., 90.], [100., 0.]], dtype=np.float32)
    estimator = HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=1,
                                               random_state=0).fit(X, [0, 0, 1, 1])
    evaluator = SimpleNamespace(columns=columns, model=estimator)
    saved = pickle.loads(pickle.dumps(SimpleNamespace(static=evaluator, exchange=evaluator)))
    predictor = RowPredictor(saved)
    row = dict(mode_exchange=1., retR_c_p1=10., retR_r_p1=90.,
               **dict.fromkeys(retr.VISIBLE_COLUMNS, 7.))
    monkeypatch.setenv(retr.VISIBLE_ENV, '0')
    off = predictor.predict(row)
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    on = predictor.predict(row)
    assert math.isfinite(on)
    assert pickle.dumps(on) == pickle.dumps(off)
    assert pickle.dumps(on) == pickle.dumps(predictor.predict(
        {k: row[k] for k in ('mode_exchange', *columns)}))
    # モデルへ実際に渡る値も保存列の順序・float32に限られる。
    assert np.array_equal(np.asarray(predictor.getters[1](row), dtype=np.float32), X[2])
