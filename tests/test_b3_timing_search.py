"""B3 1手時間テーブル・理想ツモ探索ラッパの単体テスト。"""
from __future__ import annotations

import numpy as np
import pytest

from src import indicators_v2 as iv
from src.b3_search import HMAX_TOTAL_HANDS, ideal_fire_search, points_to_ojama
from src.b3_timing import (SEC_PER_HAND_DEFAULT, TimingTable, collect_hand_samples, column_heights,
                           fit_timing_table, height_bucket)
from src.board import Board


def _board_with_chain_seed() -> np.ndarray:
    g = np.zeros((13, 6), dtype=np.int8)
    g[11:13, 0] = 1
    g[12, 1] = 1
    g[11, 1] = 2
    g[12, 2] = 2
    g[10:13, 3] = 3
    g[12, 4] = 4
    return g


def test_column_heights():
    g = np.zeros((13, 6), dtype=np.int8)
    g[12, 0] = 1
    g[8:13, 2] = 2
    assert column_heights(g).tolist() == [1, 0, 5, 0, 0, 0]


def test_hand_samples_only_clean_two_cell_placements():
    g0 = np.zeros((13, 6), dtype=np.int8)
    g1 = g0.copy(); g1[11:13, 0] = 1                 # +2 セル
    g2 = g1.copy(); g2[8:13, 3] = 2                  # +5 (着弾等) → 除外
    grids = np.stack([g0, g1, g2])
    s = collect_hand_samples(grids, np.array(["1P"] * 3), np.zeros(3, int), np.array([0.0, 0.7, 1.5]))
    assert s.shape == (1, 3) and s[0, 0] == pytest.approx(0.7) and s[0, 2] == 0


def test_table_fit_and_lookup_fallback_and_fold_injection():
    rng = np.random.default_rng(0)
    hi = np.column_stack((rng.normal(0.5, 0.02, 200), np.full(200, 11.0), np.full(200, 9.0)))
    lo = np.column_stack((rng.normal(0.8, 0.02, 200), np.full(200, 2.0), np.full(200, 1.0)))
    t_all = fit_timing_table(np.vstack([hi, lo]))
    med_hi, fast_hi, key = t_all.hand_sec(11.0, 9.0)
    med_lo, _, _ = t_all.hand_sec(2.0, 1.0)
    assert med_hi < med_lo and fast_hi < med_hi and key.startswith("m")
    # 高さ別 (高く積むほど速い)。未fit は既定値と key=default
    assert TimingTable().hand_sec(5.0)[2] == "default" and TimingTable().hand_sec(5.0)[0] == SEC_PER_HAND_DEFAULT
    # fold 内推定の差し込み: 学習サンプルだけで fit した表は全データ表と別の値になる
    t_fold = fit_timing_table(lo)
    assert t_fold.hand_sec(11.0, 9.0)[0] != med_hi
    assert TimingTable.from_json(t_all.to_json()).hand_sec(2.0, 1.0)[0] == med_lo


def test_height_bucket_edges():
    assert [height_bucket(v, (6, 9, 11)) for v in (0, 5, 6, 8, 9, 10, 11, 13)] == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_known_first_pair_matches_existing_placement(seed: int) -> None:
    """既存指標は保持し、B3の既知1手だけ配置・得点と照合する。"""
    from src.b3_search import _SIM
    rng = np.random.default_rng(seed)
    g = np.zeros((13, 6), dtype=np.int8)
    for col in range(6):
        h = int(rng.integers(1, 6))
        g[-h:, col] = rng.integers(1, 5, size=h)
    board = Board.from_list(g.tolist())
    ref = iv._near_future_known_expand([(0, board)], (1, 2), _SIM, use_exact_score=True)
    res = ideal_fire_search(g, (1, 2, 3, 4), (1, 2, 3, 4), hmax=1)
    assert res.scores[0] == ref[0][0]


def test_search_monotone_and_truncation_flag():
    g = _board_with_chain_seed()
    res = ideal_fire_search(g, (1, 2, 3, 4), (1, 2, 3, 4), HMAX_TOTAL_HANDS)
    assert all(b >= a for a, b in zip(res.scores, res.scores[1:]))
    assert res.at(HMAX_TOTAL_HANDS + 3)[2] is True      # 要求Hが探索上限超なら打切りを記録
    assert res.at(1)[2] is False


def test_points_to_ojama_margin_origin():
    assert points_to_ojama(700, 0.0) == 10
    assert points_to_ojama(700, 400.0) > 10             # マージンタイム後はレートが下がる
