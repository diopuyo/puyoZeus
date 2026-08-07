"""scripts/mc_counter_estimator.py (#24 K拡張MCロールアウト、v2) のテスト。

決定論シード再現・空盤面/満杯(窒息)盤面の境界・既知ネクスト有無・
段別テーブルによる手数予算積分・v2「積んで、期限に発火」ポリシーを検証する。

⚠️ n_rollouts/time_budget_secについて: v2は毎手 current_max_chain
(既存指標III-1、内部で最大30回simulate) を候補ごとに評価するため、v1比で
大幅に重い (scripts/mc_counter_estimator.py モジュールdocstring参照)。
テスト実行時間を抑えるため、本ファイルは n_rollouts=2-5 / 時間予算を
小さめ (1-3秒、手数2-4程度) に設定する (本番既定 n_rollouts=200 の精度は
scripts/_bench_mc_counter_v2_2026-08-04.py の実測ベンチで別途確認する)。
"""
from __future__ import annotations

import pytest

import src.indicators_v2 as iv
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_RED, Board
from scripts.mc_counter_estimator import (
    MC_COUNTER_MAX_HANDS_HARD_CAP,
    PLACEMENT_SPEED_BY_ROW_SEC,
    PLACEMENT_SPEED_FALLBACK_SEC,
    _clamp_row_index,
    _deadline_trigger_value,
    _mc_counter_seed,
    _placement_row_index,
    _select_build_placement,
    estimate_counter_distribution,
)
from src.chain import ChainSimulator


def _seed_board_ready_to_fire() -> Board:
    """既知ペア (赤,青) 相当の材料で2連鎖 (360点、お邪魔換算>0) を組める
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
        d1 = estimate_counter_distribution(board, time_budget_sec=1.5, n_rollouts=3)
        d2 = estimate_counter_distribution(board, time_budget_sec=1.5, n_rollouts=3)
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
        dist = estimate_counter_distribution(dead_board, time_budget_sec=5.0, n_rollouts=3)
        assert dist.n_rollouts == 0
        assert dist.mean == 0.0
        assert dist.p25 == 0.0
        assert dist.p75 == 0.0

    def test_empty_board_with_zero_budget_never_places(self) -> None:
        """時間予算0秒 (最速でも0.134秒かかる) では1手も打てない
        (=組む機会が無い、期限トリガーのみが現在盤面[空]に適用される)。
        """
        empty = Board()
        dist = estimate_counter_distribution(empty, time_budget_sec=0.0, n_rollouts=3)
        assert dist.mean == 0.0
        assert dist.mean_hands_used == 0.0

    def test_n_rollouts_zero_returns_empty(self) -> None:
        empty = Board()
        dist = estimate_counter_distribution(empty, time_budget_sec=5.0, n_rollouts=0)
        assert dist.n_rollouts == 0

    def test_hands_used_never_exceeds_hard_cap(self) -> None:
        board = _seed_board_ready_to_fire()
        # 極端に大きい時間予算でも安全弁 (MC_COUNTER_MAX_HANDS_HARD_CAP) を超えない
        # (n_rollouts=2に絞り、v2の重さ [毎手current_max_chain評価] でも高速に保つ)。
        dist = estimate_counter_distribution(board, time_budget_sec=1000.0, n_rollouts=2)
        assert dist.mean_hands_used <= MC_COUNTER_MAX_HANDS_HARD_CAP


class TestBuildPhasePolicy:
    """v2「組むフェーズ」: 消去を起こさない配置を優先し、消去が避けられない
    場合のみ後退する (_select_build_placement の直接テスト、低コスト)。
    """

    def test_build_only_placement_does_not_ignite(self) -> None:
        """組む候補がある盤面では、選ばれた配置は消去を起こさない
        (=盤面の色ぷよ総数が2個増えるだけ、既存構造の赤/青は消えない)。
        """
        board = _seed_board_ready_to_fire()
        sim = ChainSimulator()
        # 既存の赤/青とは無関係な (黄,黄) を遠い列に置けば消去を避けられるはず。
        placed = _select_build_placement(board, (COLOR_RED, COLOR_RED), sim)
        assert placed is not None
        # 既存の赤3個+青3個(計6個)がそのまま残っている(=どこかで消えていない)
        # ことを、色ぷよ総数が2個ちょうど増えていることで確認する。
        before_count = int((board._grid != 0).sum())
        after_count = int((placed._grid != 0).sum())
        assert after_count == before_count + 2

    def test_forced_fire_fallback_when_all_placements_ignite(self) -> None:
        """22配置全てが消去を伴う極端な盤面では、後退フォールバックで
        (Noneでない) 何らかの盤面を返す (クラッシュしないことの確認)。
        """
        # 列0,1に赤を3段積み、どこに(赤,赤)を置いても4連結完成に近い状況を作る
        # (厳密に「全22配置が発火」でなくても、フォールバック分岐が例外を出さず
        # 動くことの確認が主目的)。
        g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        for col in range(BOARD_COLS):
            g[12][col] = COLOR_RED
            g[11][col] = COLOR_RED
        board = Board.from_list(g)
        sim = ChainSimulator()
        placed = _select_build_placement(board, (COLOR_RED, COLOR_RED), sim)
        assert placed is not None  # 例外を出さず何らかの盤面を返す


class TestDeadlineTrigger:
    """v2「発火フェーズ」: 期限到達時の最良トリガー1手 (既知/任意1色)。"""

    def test_known_pair_trigger_uses_real_colors(self) -> None:
        board = _seed_board_ready_to_fire()
        sim = ChainSimulator()
        value = _deadline_trigger_value(
            board, known_pairs=((COLOR_RED, COLOR_BLUE),), known_used=0,
            sim=sim, elapsed_sec=0.0,
        )
        assert value > 0.0  # 既存の赤/青材料を使った2連鎖トリガーで発火する

    def test_no_known_pair_falls_back_to_immediate_fire_power(self) -> None:
        """既知ツモを使い切っている場合、既存指標 immediate_fire_power
        (任意1色の最良トリガー、既存機構) にそのまま委譲する (値が一致する
        ことで委譲の配線を確認、板は _seed_board_ready_to_fire — 単色1個
        drop では2連結ずつしか完成せずOJAMA_RATE_STANDARD=70未満で0点に
        なる点は既存挙動そのものであり本テストの検証対象ではない)。
        """
        board = _seed_board_ready_to_fire()
        sim = ChainSimulator()
        value = _deadline_trigger_value(
            board, known_pairs=(), known_used=0, sim=sim, elapsed_sec=0.0,
        )
        expected = iv.immediate_fire_power(board, elapsed_sec=0.0, simulator=sim).raw
        assert value == pytest.approx(expected)

    def test_dead_board_trigger_is_zero(self) -> None:
        g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        g[1][2] = COLOR_RED
        dead_board = Board.from_list(g)
        sim = ChainSimulator()
        value = _deadline_trigger_value(
            dead_board, known_pairs=(), known_used=0, sim=sim, elapsed_sec=0.0,
        )
        assert value == 0.0


class TestKnownPairsUsage:
    """既知ネクスト (next_pair/dnext_pair相当) の有無で挙動が変わることの確認。"""

    def test_known_pair_eventually_contributes_to_deadline_trigger(self) -> None:
        """既知1手目に確実に組める材料 (赤,青) を渡すと、時間予算が0で
        組む機会が無くても、既存の赤/青材料そのものが期限トリガーの対象になり
        到達値>0 になる (=既知ツモが有効に扱われている間接確認、v2では
        「即時発火」でなく「期限に撃つ」ため v1 と検証の仕方を変えている)。
        """
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(
            board, time_budget_sec=0.0, known_pairs=((COLOR_RED, COLOR_BLUE),), n_rollouts=3,
        )
        assert dist.mean > 0.0

    def test_invalid_known_pair_falls_back_to_random(self) -> None:
        """無効な既知ペア ((-1,-1)) は _near_future_is_valid_pair で弾かれ、
        ランダムサンプルにフォールバックする (クラッシュしないことの確認)。
        """
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(
            board, time_budget_sec=1.0, known_pairs=((-1, -1),), n_rollouts=3,
        )
        assert dist.n_rollouts == 3
        assert dist.mean >= 0.0

    def test_empty_known_pairs_does_not_crash(self) -> None:
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(board, time_budget_sec=1.0, n_rollouts=3)
        assert dist.n_rollouts == 3


class TestDistributionShape:
    """分位点の基本的な整合性 (定義上の単調性)。"""

    def test_p75_at_least_p25(self) -> None:
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(board, time_budget_sec=1.5, n_rollouts=5)
        assert dist.p75 >= dist.p25

    def test_prob_at_least_zero_threshold_is_one_when_material_present(self) -> None:
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(
            board, time_budget_sec=0.0, known_pairs=((COLOR_RED, COLOR_BLUE),),
            thresholds_ojama=(0.0,), n_rollouts=3,
        )
        assert dist.prob_at_least[0.0] == pytest.approx(1.0)

    def test_prob_at_least_large_threshold_is_low_for_empty_board(self) -> None:
        board = Board()
        dist = estimate_counter_distribution(
            board, time_budget_sec=1.0, thresholds_ojama=(400.0,), n_rollouts=3,
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

    def test_mc_channel_p25_le_p75(self) -> None:
        from scripts.compute_exchange_delta_winprob import _realizable_counter_ojama

        board = _seed_board_ready_to_fire()
        with_p25 = _realizable_counter_ojama(
            board, attacker_chain_count=1.0, enable_mc_counter=True, mc_channel="p25")
        with_p75 = _realizable_counter_ojama(
            board, attacker_chain_count=1.0, enable_mc_counter=True, mc_channel="p75")
        assert with_p25 <= with_p75
