"""同一行比較の凍結盤面と、観測済み・未来得点の分離を検査する。"""
from __future__ import annotations

import numpy as np
import pytest

from scripts import compare_e15b_models_20260928 as compare
from scripts.e15_training_rows_20260928 import Chain, Exchange
from src.exchange_event_features import SCORE_COLUMNS, score_features

SIDES, HEIGHT, WIDTH, COUNT_COLUMNS = 2, 13, 6, 6
TRIGGER, NEXT_TRIGGER, FIRST_SCORE, LATER_SCORE = 10., 12., 100., 500.


@pytest.mark.parametrize("source", [0, 1])
def test_frozen_observed_and_endpoint_separate_future_scores(
        monkeypatch: pytest.MonkeyPatch, source: int) -> None:
    """同じ開始盤面を使い、後発連鎖は参考条件だけへ入る。"""
    grid = np.zeros((SIDES, HEIGHT, WIDTH), np.int8)
    grid[0, -1, 0] = 1
    raw = dict(grids=grid, **{key: np.ones(SIDES, np.int8) for key in compare.v2.QUEUE})
    chains = [Chain(0, TRIGGER, TRIGGER, 0, TRIGGER, FIRST_SCORE),
              Chain(1, NEXT_TRIGGER, NEXT_TRIGGER, 1, NEXT_TRIGGER, LATER_SCORE)]
    exchange = Exchange(1, TRIGGER, np.arange(SIDES), chains)
    captured = []
    def observe(observation: object, firing: tuple, before: np.ndarray,
                after: np.ndarray) -> np.ndarray:
        captured.append((observation, firing, after.copy()))
        return np.zeros((SIDES, COUNT_COLUMNS))
    monkeypatch.setattr(compare, "side_features", observe)
    observed = compare.replacement(exchange, raw, 0., 1, source)
    endpoint = compare.replacement(exchange, raw, 0., len(chains), source)
    np.testing.assert_array_equal(captured[0][2], [FIRST_SCORE, 0])
    np.testing.assert_array_equal(captured[1][2], [FIRST_SCORE, LATER_SCORE])
    for observation, firing, _ in captured:
        np.testing.assert_array_equal(observation.grids, grid)
        assert observation.elapsed_sec == TRIGGER and not observation.live
        assert firing == (True, False)
    for vector, totals in ((observed, [FIRST_SCORE, 0]), (endpoint, [FIRST_SCORE, LATER_SCORE])):
        np.testing.assert_array_equal(vector[:len(SCORE_COLUMNS)],
            score_features(np.zeros(SIDES), np.array(totals), TRIGGER, source))
    assert not np.array_equal(observed, endpoint)
