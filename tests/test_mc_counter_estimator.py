"""scripts/mc_counter_estimator.py (#24 K拡張MCロールアウト) のテスト。

決定論シード再現・空盤面/満杯(窒息)盤面の境界・既知ネクスト有無・
段別テーブルによる手数予算積分を検証する。
"""
from __future__ import annotations

import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_RED, Board
from scripts.mc_counter_estimator import (
    MC_COUNTER_MAX_HANDS_HARD_CAP,
    PLACEMENT_SPEED_BY_ROW_SEC,
    PLACEMENT_SPEED_FALLBACK_SEC,
    _clamp_row_index,
    _mc_counter_seed,
    _placement_row_index,
    estimate_counter_distribution,
)


def _seed_board_ready_to_fire() -> Board:
    """既知ペア (赤,青) を1手置くだけで2連鎖 (360点、お邪魔換算>0) が完成する
    盤面 (既知ネクスト有無テスト用)。

    tests/test_compute_exchange_delta_winprob.py の
    _seed_board_with_small_counter と同じ構図 (盤面構築のみ、ロジックは
    再実装しない)。1連結4個のみ(40点)だとOJAMA_RATE_STANDARD=70未満で
    お邪魔換算が0になってしまうため、2連鎖が確実に起きる構図にしている。
    """
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    g[12][0] = COLOR_RED
    g[12][1] = COLOR_RED
    g[11][0] = COLOR_RED
    g[12][2] = COLOR_BLUE
    g[12][3] = COLOR_BLUE
    g[11][2] = COLOR_BLUE
    return Board.from_list(g)


class TestPlacementSpeedTable:
    """段別最速設置時間テーブル (2026-08-03実測較正) の参照・防御ロジック。"""

    def test_table_covers_all_13_rows(self) -> None:
        assert set(PLACEMENT_SPEED_BY_ROW_SEC.keys()) == set(range(BOARD_ROWS))

    def test_top_row_is_faster_than_bottom_row(self) -> None:
        """user伝授の物理 (盤面が埋まっているほど落下距離が短く速い) が
        テーブルに反映されていることの健全性チェック (row0=最上段側が速い)。
        """
        assert PLACEMENT_SPEED_BY_ROW_SEC[0] < PLACEMENT_SPEED_BY_ROW_SEC[9]

    def test_clamp_row_index_handles_out_of_range(self) -> None:
        assert _clamp_row_index(-5) == 0
        assert _clamp_row_index(999) == 12
        assert _clamp_row_index(7) == 7

    def test_fallback_is_table_max(self) -> None:
        assert PLACEMENT_SPEED_FALLBACK_SEC == max(PLACEMENT_SPEED_BY_ROW_SEC.values())

    def test_placement_row_index_detects_topmost_new_cell(self) -> None:
        before = Board()._grid.copy()
        after = before.copy()
        after[11][0] = COLOR_RED
        after[10][0] = COLOR_RED
        assert _placement_row_index(before, after) == 10

    def test_placement_row_index_no_new_cell_defaults_to_bottom(self) -> None:
        before = Board()._grid.copy()
        after = before.copy()  # 新規セル無し (満杯で置けなかった防御的ケース)
        assert _placement_row_index(before, after) == 12


class TestSeedDeterminism:
    """シードは盤面+時間予算のみに依存する決定論設計 (stateless)。"""

    def test_same_board_and_budget_same_seed(self) -> None:
        board = _seed_board_ready_to_fire()
        assert _mc_counter_seed(board, 3.0) == _mc_counter_seed(board, 3.0)

    def test_different_budget_changes_seed(self) -> None:
        board = _seed_board_ready_to_fire()
        assert _mc_counter_seed(board, 3.0) != _mc_counter_seed(board, 5.0)

    def test_distribution_reproducible_across_calls(self) -> None:
        board = _seed_board_ready_to_fire()
        d1 = estimate_counter_distribution(board, time_budget_sec=3.0, n_rollouts=20)
        d2 = estimate_counter_distribution(board, time_budget_sec=3.0, n_rollouts=20)
        assert d1.mean == pytest.approx(d2.mean)
        assert d1.p25 == pytest.approx(d2.p25)
        assert d1.p75 == pytest.approx(d2.p75)
        assert d1.mean_hands_used == pytest.approx(d2.mean_hands_used)


class TestBoundaryBoards:
    """空盤面・窒息(満杯)盤面の境界ケース。"""

    def test_dead_board_returns_zero_distribution(self) -> None:
        g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        g[1][2] = COLOR_RED  # 窒息判定セル (DEATH_ROW=1, DEATH_COL=2)
        dead_board = Board.from_list(g)
        assert dead_board.is_dead()
        dist = estimate_counter_distribution(dead_board, time_budget_sec=5.0, n_rollouts=20)
        assert dist.n_rollouts == 0
        assert dist.mean == 0.0
        assert dist.p25 == 0.0
        assert dist.p75 == 0.0

    def test_empty_board_with_zero_budget_never_places(self) -> None:
        """時間予算0秒 (最速でも0.134秒かかる) では1手も打てない。"""
        empty = Board()
        dist = estimate_counter_distribution(empty, time_budget_sec=0.0, n_rollouts=20)
        assert dist.mean == 0.0
        assert dist.mean_hands_used == 0.0

    def test_n_rollouts_zero_returns_empty(self) -> None:
        empty = Board()
        dist = estimate_counter_distribution(empty, time_budget_sec=5.0, n_rollouts=0)
        assert dist.n_rollouts == 0

    def test_hands_used_never_exceeds_hard_cap(self) -> None:
        empty = Board()
        # 極端に大きい時間予算でも安全弁 (MC_COUNTER_MAX_HANDS_HARD_CAP) を超えない。
        dist = estimate_counter_distribution(empty, time_budget_sec=1000.0, n_rollouts=10)
        assert dist.mean_hands_used <= MC_COUNTER_MAX_HANDS_HARD_CAP


class TestKnownPairsUsage:
    """既知ネクスト (next_pair/dnext_pair相当) の有無で挙動が変わることの確認。"""

    def test_known_pair_guarantees_immediate_fire_regardless_of_random_tail(self) -> None:
        """既知1手目が確実に発火する組み合わせなら、乱数タイル(以降の手)に
        関わらず全ロールアウトで到達値>0 になる (=既知ツモが実際に先頭手に
        強制適用されていることの間接確認)。
        """
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(
            board, time_budget_sec=2.0, known_pairs=((COLOR_RED, COLOR_BLUE),), n_rollouts=30,
        )
        assert dist.p25 > 0.0  # 25パーセンタイルでも既に発火済み (全数成功の証拠)

    def test_invalid_known_pair_falls_back_to_random(self) -> None:
        """無効な既知ペア ((-1,-1)) は _near_future_is_valid_pair で弾かれ、
        ランダムサンプルにフォールバックする (クラッシュしないことの確認)。
        """
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(
            board, time_budget_sec=2.0, known_pairs=((-1, -1),), n_rollouts=10,
        )
        assert dist.n_rollouts == 10
        assert dist.mean >= 0.0

    def test_empty_known_pairs_does_not_crash(self) -> None:
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(board, time_budget_sec=2.0, n_rollouts=10)
        assert dist.n_rollouts == 10


class TestDistributionShape:
    """分位点の基本的な整合性 (定義上の単調性)。"""

    def test_p75_at_least_p25(self) -> None:
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(board, time_budget_sec=4.0, n_rollouts=50)
        assert dist.p75 >= dist.p25

    def test_prob_at_least_zero_threshold_is_one_when_any_score_reached(self) -> None:
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(
            board, time_budget_sec=2.0, known_pairs=((COLOR_RED, COLOR_RED),),
            thresholds_ojama=(0.0,), n_rollouts=10,
        )
        assert dist.prob_at_least[0.0] == pytest.approx(1.0)

    def test_prob_at_least_large_threshold_is_low_for_small_budget(self) -> None:
        board = Board()
        dist = estimate_counter_distribution(
            board, time_budget_sec=1.0, thresholds_ojama=(400.0,), n_rollouts=30,
        )
        assert dist.prob_at_least[400.0] < 0.1


class TestBackwardCompatWithRealizableCounterOjama:
    """_realizable_counter_ojama への接続 (フラグ既定Falseで無挙動変化)。"""

    def test_enable_mc_counter_default_off_matches_fix_h(self) -> None:
        from scripts.compute_exchange_delta_winprob import _realizable_counter_ojama

        board = _seed_board_ready_to_fire()
        fix_h_only = _realizable_counter_ojama(board, attacker_chain_count=6.0)
        explicit_off = _realizable_counter_ojama(
            board, attacker_chain_count=6.0, enable_mc_counter=False)
        assert explicit_off == pytest.approx(fix_h_only)

    def test_enable_mc_counter_true_never_decreases_value(self) -> None:
        from scripts.compute_exchange_delta_winprob import _realizable_counter_ojama

        board = _seed_board_ready_to_fire()
        fix_h_only = _realizable_counter_ojama(board, attacker_chain_count=6.0)
        with_mc = _realizable_counter_ojama(
            board, attacker_chain_count=6.0, enable_mc_counter=True, mc_channel="p75")
        assert with_mc >= fix_h_only

    def test_mc_channel_p25_le_p75(self) -> None:
        from scripts.compute_exchange_delta_winprob import _realizable_counter_ojama

        board = _seed_board_ready_to_fire()
        with_p25 = _realizable_counter_ojama(
            board, attacker_chain_count=6.0, enable_mc_counter=True, mc_channel="p25")
        with_p75 = _realizable_counter_ojama(
            board, attacker_chain_count=6.0, enable_mc_counter=True, mc_channel="p75")
        assert with_p25 <= with_p75
