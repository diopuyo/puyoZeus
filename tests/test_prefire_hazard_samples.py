"""hazard 学習サンプルのラベル規則 (scripts/prefire_hazard_samples_20260930.py) の単体テスト (段1)。"""
from __future__ import annotations

import numpy as np

from scripts import prefire_hazard_samples_20260930 as samples

RATE = 70   # 経過0秒のおじゃま1個あたり得点


def data(triggers: list[float], scores: list[int], games: list[int] | None = None) -> dict:
    """1側だけの行列 (t = 0,1,2,...)。"""
    n = len(triggers)
    return dict(t_sec=np.arange(n, dtype=float), chain_trigger_sec=np.asarray(triggers, dtype=float),
                score=np.asarray(scores), game_idx=np.asarray(games or [0] * n))


def test_label_positive_when_big_fire_within_horizon() -> None:
    d = data([np.nan, np.nan, 1.5, np.nan, np.nan], [0, 0, 70 * 10, 700, 700])
    label, missing = samples.label_row(d, np.arange(5), 0, best_send=10.0, start=0.0)
    assert (label, missing) == (1.0, 0)


def test_small_fire_does_not_count_but_later_big_one_does() -> None:
    d = data([np.nan, 0.5, np.nan, 2.5, np.nan], [0, 70, 70, 70 + 70 * 8, 0])
    label, _ = samples.label_row(d, np.arange(5), 0, best_send=10.0, start=0.0)
    assert label == 1.0   # 1個 (整地) は数えず、次の8個 (≥ 0.5×10) で1


def test_fire_beyond_horizon_is_negative() -> None:
    n = samples.HORIZON_PLACEMENTS + 2
    triggers = [np.nan] * n
    triggers[-1] = n - 1.5
    scores = [0] * (n - 1) + [70 * 20]
    label, _ = samples.label_row(data(triggers, scores), np.arange(n), 0, best_send=10.0, start=0.0)
    assert label == 0.0


def test_game_boundary_stops_lookahead() -> None:
    d = data([np.nan, np.nan, 1.5], [0, 0, 700], games=[0, 1, 1])
    label, _ = samples.label_row(d, np.arange(3), 0, best_send=10.0, start=0.0)
    assert label == 0.0


def test_unknown_score_is_missing_not_guessed() -> None:
    d = data([np.nan, 0.5], [-1, 700])
    label, missing = samples.label_row(d, np.arange(2), 0, best_send=10.0, start=0.0)
    assert np.isnan(label) and missing == 1
