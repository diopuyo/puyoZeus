"""高速経路の特徴値・推論値・キャッシュ寿命を検証する。"""
from __future__ import annotations

import struct
import pickle

import numpy as np
import pandas as pd
import pytest

from src import b3_features, b3_static, indicators_v2 as iv
from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.chain import ChainSimulator

SEED = 603


def boards() -> list[Board]:
    """空・通常・穴・隠し段・窒息・UNKNOWN・全列満杯を含む。"""
    rng = np.random.default_rng(SEED)
    grids = [np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8),
             np.ones((BOARD_ROWS, BOARD_COLS), dtype=np.int8)]
    for high in (3, 7, 13):
        for unknown in (False, True):
            g = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
            for col in range(BOARD_COLS):
                height = int(rng.integers(1, high + 1))
                g[-height:, col] = rng.choice((1, 2, 3, 4, 5, 9), height)
            if unknown:
                g[0, 2] = 10
                g[BOARD_ROWS - 2, 1] = 0
            grids.append(g)
    # s1 g6 f30764: 消去後に元から存在する穴も詰まることが最小発火数を変える。
    hole = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    hole[1:, 3] = (2, 2, 2, 2, 9, 2, 4, 3, 3, 3, 0, 3)
    hole[9:, :2] = ((1, 1), (5, 4), (3, 3), (3, 3))
    grids.append(hole)
    return [Board.from_list(g.tolist()) for g in grids]


@pytest.mark.parametrize('ghost', (False, True))
def test_min_ignite_exact(ghost: bool) -> None:
    sim = ChainSimulator(exclude_hidden_row_from_pop=ghost)
    for board in boards():
        original = board._grid.copy()
        expected = iv.min_puyos_to_ignite(board, sim)
        actual = b3_static.min_ignite(board, sim)
        assert actual == expected
        np.testing.assert_array_equal(board._grid, original)


def test_static_all_columns_exact() -> None:
    for board in boards():
        actual = b3_static.evaluate(board, b3_features._IND_FUNCS)
        for name, func in b3_features._IND_FUNCS.items():
            expected = float(func(board).score)
            assert struct.pack('d', actual[name]) == struct.pack('d', expected), name


def test_min_ignite_without_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b3_static.core, 'NATIVE_AVAILABLE', False)
    sim = ChainSimulator()
    board = boards()[0]
    assert b3_static.min_ignite(board, sim) == iv.min_puyos_to_ignite(board, sim)


def test_min_ignite_hole_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object) -> None:
        raise AssertionError('穴のある盤面をRustへ渡してはいけない')

    monkeypatch.setattr(b3_static.core, 'chain_metrics_after_drops', unexpected)
    sim = ChainSimulator(exclude_hidden_row_from_pop=True, fast_groups=True)
    assert b3_static.min_ignite(boards()[-1], sim).raw == 2.0


def test_static_cache_hit_boundary_and_limit() -> None:
    from src.b3_state import GameState

    b3_features.clear_static_caches()
    raw = boards()[0]._grid.astype(np.int8).tobytes()
    first = b3_features.static_indicators(raw)
    assert b3_features.static_indicators(raw) is first
    info = b3_features.static_indicators.cache_info()
    assert info.hits == 1 and info.misses == 1
    assert info.maxsize == b3_features.STATIC_CACHE_SIZE
    b3_features.board_is_dead(raw)
    # reset_gameは初期化時のlean読み取りを必要としない。
    state = object.__new__(GameState)
    state.t = 0.0
    state.reset_game(2)
    assert b3_features.static_indicators.cache_info().currsize == 0
    assert b3_features.board_is_dead.cache_info().currsize == 0


@pytest.mark.parametrize('columns', [('b', 'a', 'c'), ('a',)])
def test_predict_row_bits_and_column_order(columns: tuple[str, ...]) -> None:
    from src.b3_train_model import HgbConfig, fit_two_state, to_matrix

    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=['a', 'b', 'c'])
    frame.loc[::3, 'b'] = np.nan
    frame['mode_exchange'] = np.arange(len(frame)) % 2
    frame['won_p1'] = (np.arange(len(frame)) // 2) % 2
    frame['weight'] = 1.0
    model = fit_two_state(frame, columns, HgbConfig('test', 4, 0.1, 3, 2), SEED)
    for row in frame.to_dict('records')[:12]:
        reference = model.predict(pd.DataFrame([row]))[0]
        assert struct.pack('d', model.predict_row(row)) == struct.pack('d', reference)
    restored = pickle.loads(pickle.dumps(model))
    assert '_row_predictor' not in restored.__dict__
    assert struct.pack('d', restored.predict_row(row)) == struct.pack('d', reference)
    row = frame.iloc[0].to_dict()
    values = np.asarray([row[c] for c in columns], dtype=np.float32).reshape(1, -1)
    assert values.tobytes() == to_matrix(pd.DataFrame([row]), columns).tobytes()
    with pytest.raises(ValueError, match='欠測'):
        model.predict_row({**row, 'mode_exchange': float('nan')})
    with pytest.raises(KeyError):
        model.predict_row({'mode_exchange': 0})
    assert np.isnan(model.predict_row({**row, 'mode_exchange': 2}))


def test_predict_row_missing_evaluator() -> None:
    from src.b3_train_model import TwoStateModel

    with pytest.raises(ValueError, match='評価器が無い'):
        TwoStateModel(None, None).predict_row({'mode_exchange': 0})
