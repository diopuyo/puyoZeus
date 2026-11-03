"""更新行の因果的採用、得点の発火条件、対称化を検証する。"""
from __future__ import annotations

import numpy as np
import pytest

from scripts import e15_training_rows_20260928 as rows
from src.exchange_event_features import D_COLUMNS, ARRIVAL_COLUMNS, SCORE_COLUMNS


def raw_game() -> dict:
    """片側の連鎖中に相手が置き、その後に相手も発火する。"""
    grid = np.zeros((7, 13, 6), np.int8)
    grid[:, -1, :3] = 1
    grid[3, -1, 3] = 2
    grid[5:, -1, :3] = 9
    value = dict(grids=grid, t_sec=np.array([0., 0., 1., 2., 3., 4., 5.]),
        side=np.array(["1P", "2P", "1P", "2P", "2P", "1P", "2P"]),
        chain_trigger_sec=np.array([np.nan, np.nan, .5, np.nan, 2.5, np.nan, np.nan]),
        chain_mechanism=np.array(["", "", "formula", "", "formula", "", ""]),
        score=np.array([0, 0, 10, 0, 20, 100, 500]), game_idx=np.ones(7, int))
    value.update({key: np.ones(7, np.int8) for key in rows.v2.QUEUE})
    return value


def test_latest_stable_and_live_chain() -> None:
    raw = raw_game()
    generated = list(rows.samples(raw, np.arange(7)))
    assert [r[1] for r in generated] == [2., 4., 5.]
    first, last = generated[0], generated[-1]
    np.testing.assert_array_equal(first[2], [0, 3])
    assert len(first[3]) == 1 and first[3][0].side == 0
    assert last[3] == []


def test_no_future_endpoint_needed_for_membership() -> None:
    raw = raw_game()
    shortened = {key: value[:4] for key, value in raw.items()}
    first = list(rows.samples(shortened, np.arange(4)))
    assert [r[1] for r in first] == [2.]


def test_later_chain_score_not_available_early(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = raw_game()
    iterator = rows.samples(raw, np.arange(7))
    exchange, stamp, ids, active = next(iterator)
    exchange.base = np.zeros(len(D_COLUMNS))
    exchange.arrival = np.zeros(len(ARRIVAL_COLUMNS))
    captured = []
    def observe(obs: object, firing: tuple, before: np.ndarray | None = None,
                after: np.ndarray | None = None) -> np.ndarray:
        captured.append(None if after is None else after.copy())
        return np.zeros((2, 6))
    monkeypatch.setattr(rows, "side_features", observe)
    rows.features(exchange, stamp, ids, active, raw, 0.)
    np.testing.assert_array_equal(captured[0], [100, 0])
    assert captured[1] is None
    assert all(c.observed <= stamp for c in exchange.chains)


def test_counter_absolute_columns_swap_without_sign() -> None:
    values = np.array([[1, 2, 3, 4, 5, -6], [7, 8, 9, 10, 11, 12]])
    left, right = rows.orient(values, 0), rows.orient(values, 1)
    np.testing.assert_array_equal(left[:5], right[5:10])
    np.testing.assert_array_equal(left[10:15], -right[10:15])
    np.testing.assert_array_equal(left[-2:], right[-2:][::-1])
