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

import math
import random

import numpy as np
import pytest

import src.indicators_v2 as iv
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_GREEN, COLOR_RED, Board
from scripts.mc_counter_estimator import (
    BEAM_ROLLOUT_AVG_STEP_TIME_SEC,
    EXACT_SHALLOW_MAX_DEPTH,
    EXACT_SHALLOW_PRUNE_HEIGHT,
    EXACT_SHALLOW_SEED_DEPTH,
    MC_COUNTER_MAX_HANDS_HARD_CAP,
    PLACEMENT_SPEED_BY_ROW_SEC,
    PLACEMENT_SPEED_FALLBACK_SEC,
    _board_is_gravity_consistent,
    _canonical_pair,
    _clamp_row_index,
    _deadline_trigger_value,
    _draw_beam_tsumo_sequence,
    _lookup_or_compute_exact_shallow_seed,
    _mc_counter_seed,
    _ojama_threshold_to_score_threshold,
    _placement_row_index,
    _resolve_auto_rollout_mode,
    _rollout_once_beam,
    _rollout_once_exact_shallow,
    _select_best_placement,
    _select_build_placement,
    _time_budget_to_beam_depth,
    estimate_counter_distribution,
)
from src.chain import ChainSimulator
from src.indicators_v2 import _score_to_ojama_count
from src.puyo_core_bridge import NATIVE_AVAILABLE, exact_shallow_search


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


def _seed_gravity_violation_board() -> Board:
    """列0に浮きぷよ (row10=赤の下、row11=空、row12=赤という重力違反) を
    仕込んだ人工盤面 (native 安全弁テスト専用)。

    認識由来の浮きぷよ欠陥
    (`project_gravity_violation_regen_lead_2026-07-30`、実測0.28%)を模した
    もの。列2-3には通常材料 (青2連結) も積んでおき、ロールアウトが実際に
    手を打てる (組む/発火フェーズが両方動く) 構図にする。
    """
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    g[10][0] = COLOR_RED  # 浮きぷよ (下に row11 の空きを挟む)
    g[12][0] = COLOR_RED
    g[12][2] = COLOR_BLUE
    g[12][3] = COLOR_BLUE
    g[11][2] = COLOR_BLUE
    return Board.from_list(g)


class TestGravityViolationSafetyValve:
    """重力違反盤面 (認識由来の浮きぷよ) に対する native 安全弁のテスト
    (モジュール docstring「v3.1 重力違反盤面の安全弁」参照)。
    """

    def test_detects_floating_puyo_column(self) -> None:
        board = _seed_gravity_violation_board()
        assert not _board_is_gravity_consistent(board)

    def test_normal_board_is_gravity_consistent(self) -> None:
        board = _seed_board_ready_to_fire()
        assert _board_is_gravity_consistent(board)

    def test_empty_board_is_gravity_consistent(self) -> None:
        assert _board_is_gravity_consistent(Board())

    def test_native_default_matches_python_on_violation(self) -> None:
        """重力違反盤面では use_native=True (既定) でも安全弁が働き、
        呼び出し全体が純Python経路に固定される。use_native=False (明示的
        純Python) と完全一致することで、native/Python混在による不整合が
        起きないことを確認する (「完全一致」要件)。
        """
        board = _seed_gravity_violation_board()
        native_default = estimate_counter_distribution(
            board, time_budget_sec=1.5, n_rollouts=3,
        )
        python_explicit = estimate_counter_distribution(
            board, time_budget_sec=1.5, n_rollouts=3, use_native=False,
        )
        assert native_default.mean == pytest.approx(python_explicit.mean)
        assert native_default.p25 == pytest.approx(python_explicit.p25)
        assert native_default.p75 == pytest.approx(python_explicit.p75)
        assert native_default.mean_hands_used == pytest.approx(
            python_explicit.mean_hands_used,
        )


@pytest.mark.skipif(
    not NATIVE_AVAILABLE, reason="puyo_core ネイティブ拡張が未ビルド (maturin develop 要)",
)
class TestNativePythonSelectionParity:
    """v3.2 (2026-08-13、選択ロジックの境界コスト削減) の回帰確認。

    重力違反盤面限定の安全弁テスト (`TestGravityViolationSafetyValve`) とは
    別に、通常 (重力一貫) 盤面で `use_native=True` (境界コスト削減後の
    native経路) が `use_native=False` (純Python経路) と完全一致することを
    直接確認する (`_select_best_placement`/`_select_build_placement` の
    リファクタ自体の正しさの検証、往復回数を減らしても選択結果が変わらない
    ことの保証)。
    """

    def test_select_best_placement_native_matches_python(self) -> None:
        board = _seed_board_ready_to_fire()
        sim = ChainSimulator()
        native = _select_best_placement(board, (COLOR_RED, COLOR_BLUE), sim, use_native=True)
        python = _select_best_placement(board, (COLOR_RED, COLOR_BLUE), sim, use_native=False)
        assert native is not None
        assert python is not None
        assert native[0] == pytest.approx(python[0])
        assert np.array_equal(native[1]._grid, python[1]._grid)
        assert np.array_equal(native[2]._grid, python[2]._grid)

    def test_select_build_placement_native_matches_python(self) -> None:
        board = _seed_board_ready_to_fire()
        sim = ChainSimulator()
        native = _select_build_placement(board, (COLOR_RED, COLOR_RED), sim, use_native=True)
        python = _select_build_placement(board, (COLOR_RED, COLOR_RED), sim, use_native=False)
        assert native is not None
        assert python is not None
        assert np.array_equal(native._grid, python._grid)

    def test_estimate_counter_distribution_native_matches_python_on_normal_board(self) -> None:
        board = _seed_board_ready_to_fire()
        native = estimate_counter_distribution(
            board, time_budget_sec=1.5, n_rollouts=5,
            known_pairs=((COLOR_RED, COLOR_BLUE),),
        )
        python = estimate_counter_distribution(
            board, time_budget_sec=1.5, n_rollouts=5,
            known_pairs=((COLOR_RED, COLOR_BLUE),), use_native=False,
        )
        assert native.mean == pytest.approx(python.mean)
        assert native.p25 == pytest.approx(python.p25)
        assert native.p75 == pytest.approx(python.p75)
        assert native.mean_hands_used == pytest.approx(python.mean_hands_used)


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


class TestKnownPrefixDedup:
    """v3.3 (2026-08-21、既知ツモ重複計算排除) の回帰確認。

    既知ツモ区間 (乱数不使用) を1回だけ計算して全ロールアウトで共有する
    `enable_prefix_dedup=True` (既定) が、毎ロールアウトでフルに
    再計算する `enable_prefix_dedup=False` (旧挙動相当) と完全一致することを
    直接確認する (最適化自体が値を変えていないことの保証)。
    """

    def test_dedup_matches_full_recompute_with_known_pairs(self) -> None:
        board = _seed_board_ready_to_fire()
        dedup = estimate_counter_distribution(
            board, time_budget_sec=2.0, n_rollouts=5,
            known_pairs=((COLOR_RED, COLOR_BLUE), (COLOR_RED, COLOR_RED)),
        )
        full_recompute = estimate_counter_distribution(
            board, time_budget_sec=2.0, n_rollouts=5,
            known_pairs=((COLOR_RED, COLOR_BLUE), (COLOR_RED, COLOR_RED)),
            enable_prefix_dedup=False,
        )
        assert dedup.mean == pytest.approx(full_recompute.mean)
        assert dedup.p25 == pytest.approx(full_recompute.p25)
        assert dedup.p75 == pytest.approx(full_recompute.p75)
        assert dedup.mean_hands_used == pytest.approx(full_recompute.mean_hands_used)

    def test_dedup_matches_full_recompute_without_known_pairs(self) -> None:
        """既知ツモが空 (=既知区間が0手) でも安全に動作し、値が変わらないこと。"""
        board = _seed_board_ready_to_fire()
        dedup = estimate_counter_distribution(board, time_budget_sec=1.5, n_rollouts=5)
        full_recompute = estimate_counter_distribution(
            board, time_budget_sec=1.5, n_rollouts=5, enable_prefix_dedup=False,
        )
        assert dedup.mean == pytest.approx(full_recompute.mean)
        assert dedup.mean_hands_used == pytest.approx(full_recompute.mean_hands_used)

    def test_dedup_matches_full_recompute_with_zero_time_budget(self) -> None:
        """既知手の1手目で即座に時間予算超過になるケース (early_stop分岐)。"""
        board = _seed_board_ready_to_fire()
        dedup = estimate_counter_distribution(
            board, time_budget_sec=0.0, n_rollouts=4,
            known_pairs=((COLOR_RED, COLOR_BLUE),),
        )
        full_recompute = estimate_counter_distribution(
            board, time_budget_sec=0.0, n_rollouts=4,
            known_pairs=((COLOR_RED, COLOR_BLUE),), enable_prefix_dedup=False,
        )
        assert dedup.mean == pytest.approx(full_recompute.mean)

    def test_dedup_matches_full_recompute_with_invalid_known_pair(self) -> None:
        """既知ペアが無効 (-1,-1) で即座に乱数区間へ移る境界ケース。"""
        board = _seed_board_ready_to_fire()
        dedup = estimate_counter_distribution(
            board, time_budget_sec=1.0, n_rollouts=4, known_pairs=((-1, -1),),
        )
        full_recompute = estimate_counter_distribution(
            board, time_budget_sec=1.0, n_rollouts=4, known_pairs=((-1, -1),),
            enable_prefix_dedup=False,
        )
        assert dedup.mean == pytest.approx(full_recompute.mean)
        assert dedup.mean_hands_used == pytest.approx(full_recompute.mean_hands_used)


@pytest.mark.skipif(
    not NATIVE_AVAILABLE, reason="puyo_core ネイティブ拡張が未ビルド (maturin develop 要)",
)
class TestBeamRolloutWiring:
    """v4 (2026-08-21、ビームロールアウト方式、user決定 project_counter_
    beam_rollout_design_2026-08-21) の配線確認。既定は greedy のまま
    (backwards compat)、rollout_mode="beam" を明示した場合のみ有効になる。
    """

    def test_default_rollout_mode_is_greedy_unchanged(self) -> None:
        """rollout_mode を省略した場合、既存 (greedy) と完全一致すること。"""
        board = _seed_board_ready_to_fire()
        implicit = estimate_counter_distribution(board, time_budget_sec=1.5, n_rollouts=5)
        explicit = estimate_counter_distribution(
            board, time_budget_sec=1.5, n_rollouts=5, rollout_mode="greedy",
        )
        assert implicit.mean == pytest.approx(explicit.mean)
        assert implicit.mean_hands_used == pytest.approx(explicit.mean_hands_used)

    def test_beam_mode_without_beam_width_raises(self) -> None:
        board = _seed_board_ready_to_fire()
        with pytest.raises(ValueError, match="beam_width"):
            estimate_counter_distribution(
                board, time_budget_sec=1.5, n_rollouts=2, rollout_mode="beam",
            )

    def test_invalid_rollout_mode_raises(self) -> None:
        board = _seed_board_ready_to_fire()
        with pytest.raises(ValueError, match="rollout_mode"):
            estimate_counter_distribution(
                board, time_budget_sec=1.5, n_rollouts=2, rollout_mode="not_a_mode",
            )

    def test_beam_mode_is_deterministic(self) -> None:
        board = _seed_board_ready_to_fire()
        d1 = estimate_counter_distribution(
            board, time_budget_sec=2.0, n_rollouts=3, rollout_mode="beam", beam_width=5,
        )
        d2 = estimate_counter_distribution(
            board, time_budget_sec=2.0, n_rollouts=3, rollout_mode="beam", beam_width=5,
        )
        assert d1.mean == pytest.approx(d2.mean)
        assert d1.mean_hands_used == pytest.approx(d2.mean_hands_used)

    def test_beam_mode_dead_board_returns_zero(self) -> None:
        g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        g[1][2] = COLOR_RED
        dead_board = Board.from_list(g)
        dist = estimate_counter_distribution(
            dead_board, time_budget_sec=2.0, n_rollouts=2, rollout_mode="beam", beam_width=5,
        )
        assert dist.n_rollouts == 0

    def test_beam_mode_zero_budget_never_places(self) -> None:
        board = _seed_board_ready_to_fire()
        dist = estimate_counter_distribution(
            board, time_budget_sec=0.0, n_rollouts=2, rollout_mode="beam", beam_width=5,
        )
        assert dist.mean == 0.0
        assert dist.mean_hands_used == 0.0

    def test_time_budget_to_beam_depth_matches_avg_step_time(self) -> None:
        depth = _time_budget_to_beam_depth(BEAM_ROLLOUT_AVG_STEP_TIME_SEC * 5.0)
        assert depth == 5

    def test_time_budget_to_beam_depth_never_exceeds_hard_cap(self) -> None:
        depth = _time_budget_to_beam_depth(1000.0)
        assert depth == MC_COUNTER_MAX_HANDS_HARD_CAP

    def test_draw_beam_tsumo_sequence_uses_known_pairs_first(self) -> None:
        rng = random.Random(0)
        seq = _draw_beam_tsumo_sequence(
            depth=3, known_pairs=((COLOR_RED, COLOR_BLUE),), colors=(1, 2, 3, 4), rng=rng,
        )
        assert len(seq) == 3
        assert seq[0] == (COLOR_RED, COLOR_BLUE)

    def test_rollout_once_beam_uses_known_material_for_higher_score(self) -> None:
        """既知ツモ (赤,青) で2連鎖が組める盤面では、素点0でないこと
        (`_rollout_once_beam` がビームサーチ経路に正しく配線されている
        ことの直接確認、native puyo_core への配線が切れていれば0になる)。
        """
        board = _seed_board_ready_to_fire()
        rng = random.Random(1)
        outcome = _rollout_once_beam(
            board, time_budget_sec=2.0, colors=(1, 2, 3, 4),
            known_pairs=((COLOR_RED, COLOR_BLUE),), rng=rng, elapsed_sec=0.0, beam_width=5,
        )
        assert outcome.achieved_ojama > 0.0
        assert outcome.hands_used > 0


def _seed_two_red_stack_board() -> Board:
    """col0 に赤2個 (高さ2) だけ置いた盤面 (exact_shallow の陽性対照用)。

    (赤,緑) を繰り返し引くロールアウトでは、1手だけでは4連結を完成できず
    (3連結止まり、非発火)、2手目でようやく4連結が完成して発火する
    (深さ1では過小評価=0、深さ2以降で40点、深さ3で100点になることを
    対話実行で確認済み、`TestExactShallowPositiveControl` 参照)。
    """
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    g[12][0] = COLOR_RED
    g[11][0] = COLOR_RED
    return Board.from_list(g)


class TestExactShallowCorrectness:
    """v5 exact_shallow (ama方式の浅い完全探索) の正しさ (message2 受け入れ
    条件1「小さい盤面で手計算と一致するユニットテスト」)。深さ2なら
    22+22²=484通りの全列挙なので決定的・手計算で検証可能。
    """

    def test_two_same_color_pairs_ignite_4_connect_on_empty_board(self) -> None:
        """空盤面に (赤,赤) を2手続けて同じ列に積むと4連結40点で発火する
        (手計算: 4個ちょうどの連結、連結ボーナス0、4*10=40)。
        """
        r1 = exact_shallow_search(Board(), [(COLOR_RED, COLOR_RED)], exclude_hidden_row_from_pop=True)
        r2 = exact_shallow_search(
            Board(), [(COLOR_RED, COLOR_RED), (COLOR_RED, COLOR_RED)],
            exclude_hidden_row_from_pop=True,
        )
        assert r1.best_score == 0, "1手だけでは赤2個で4連結未完成、発火しない"
        assert r2.best_score == 40, "2手で赤4個ちょうど連結、40点"

    def test_frontier_size_matches_placement_count(self) -> None:
        """深さ1の完全探索フロンティア数は `enumerate_placements` の22通り
        と一致する (空盤面・2色ペアでは誰も発火しないため全件残るはず)。
        """
        r = exact_shallow_search(Board(), [(COLOR_RED, COLOR_BLUE)], exclude_hidden_row_from_pop=True)
        assert len(r.final_frontier) == 22

    def test_max_height_prunes_vertical_placements_at_low_threshold(self) -> None:
        """max_height=2 (=高さ2以上を枝刈り) を空盤面+2色ペアに適用すると、
        縦置き12通り (どの列でも高さ2になる) が枝刈りされ、横置き10通り
        (各列高さ1) のみ残るはず (手計算で検証可能)。
        """
        r = exact_shallow_search(
            Board(), [(COLOR_RED, COLOR_BLUE)], exclude_hidden_row_from_pop=True, max_height=2,
        )
        assert len(r.final_frontier) == 10

    def test_max_height_none_keeps_all_22(self) -> None:
        r = exact_shallow_search(
            Board(), [(COLOR_RED, COLOR_BLUE)], exclude_hidden_row_from_pop=True, max_height=None,
        )
        assert len(r.final_frontier) == 22

    def test_prune_height_derived_from_death_row_physical_constant(self) -> None:
        """`EXACT_SHALLOW_PRUNE_HEIGHT` (=12) が `BOARD_ROWS - DEATH_ROW`
        から導出されていること (ama の `11` を直接持ち込んでいない確認、
        モジュール docstring「v5」参照)。"""
        assert EXACT_SHALLOW_PRUNE_HEIGHT == BOARD_ROWS - 1  # DEATH_ROW=1


class TestExactShallowPositiveControl:
    """message2 受け入れ条件4: 深さを1に落とすと過小評価が悪化すること。"""

    def test_depth_one_underestimates_vs_deeper_search(self) -> None:
        board = _seed_two_red_stack_board()
        r1 = exact_shallow_search(
            board, [(COLOR_RED, COLOR_GREEN)], exclude_hidden_row_from_pop=True,
        )
        r2 = exact_shallow_search(
            board, [(COLOR_RED, COLOR_GREEN)] * 2, exclude_hidden_row_from_pop=True,
        )
        r3 = exact_shallow_search(
            board, [(COLOR_RED, COLOR_GREEN)] * 3, exclude_hidden_row_from_pop=True,
        )
        assert r1.best_score == 0
        assert r2.best_score == 40
        assert r3.best_score == 100
        assert r1.best_score < r2.best_score < r3.best_score

    def test_rollout_once_exact_shallow_depth_override_reproduces_underestimate(self) -> None:
        """`_rollout_once_exact_shallow` の `_max_depth_override` (テスト専用
        フック) を1に落とすと、既定 (`EXACT_SHALLOW_MAX_DEPTH`) より
        反撃値が下がること (実際のロールアウト関数経由での確認)。
        """
        board = _seed_two_red_stack_board()
        # 既定深さ (EXACT_SHALLOW_MAX_DEPTH=3) までランダムを使わず全て
        # 既知ツモにする (乱数分岐を排除し、決定的に比較するため)。
        known_pairs = ((COLOR_RED, COLOR_GREEN),) * EXACT_SHALLOW_MAX_DEPTH
        rng_depth1 = random.Random(0)
        outcome_depth1 = _rollout_once_exact_shallow(
            board, time_budget_sec=2.0, colors=(1, 2, 3, 4), known_pairs=known_pairs,
            rng=rng_depth1, elapsed_sec=0.0, _max_depth_override=1,
        )
        rng_default = random.Random(0)
        outcome_default = _rollout_once_exact_shallow(
            board, time_budget_sec=2.0, colors=(1, 2, 3, 4), known_pairs=known_pairs,
            rng=rng_default, elapsed_sec=0.0,
        )
        assert outcome_depth1.achieved_ojama < outcome_default.achieved_ojama


class TestAutoRolloutDispatch:
    """v5 `rollout_mode="auto"` の物理量に基づく振り分け
    (message2 受け入れ条件3)。"""

    def test_short_budget_resolves_to_exact_shallow(self) -> None:
        # depth<=EXACT_SHALLOW_MAX_DEPTH になる短い予算
        budget = BEAM_ROLLOUT_AVG_STEP_TIME_SEC * EXACT_SHALLOW_MAX_DEPTH
        assert _resolve_auto_rollout_mode(budget) == "exact_shallow"

    def test_long_budget_resolves_to_beam(self) -> None:
        budget = BEAM_ROLLOUT_AVG_STEP_TIME_SEC * (EXACT_SHALLOW_MAX_DEPTH + 5)
        assert _resolve_auto_rollout_mode(budget) == "beam"

    def test_auto_mode_matches_exact_shallow_for_short_budget(self) -> None:
        board = _seed_board_ready_to_fire()
        budget = BEAM_ROLLOUT_AVG_STEP_TIME_SEC * 2.0
        known_pairs = ((COLOR_RED, COLOR_BLUE),)
        auto_dist = estimate_counter_distribution(
            board, budget, known_pairs=known_pairs, n_rollouts=4, rollout_mode="auto",
            beam_width=10,
        )
        exact_dist = estimate_counter_distribution(
            board, budget, known_pairs=known_pairs, n_rollouts=4, rollout_mode="exact_shallow",
        )
        assert auto_dist.mean == pytest.approx(exact_dist.mean)
        assert auto_dist.mean_hands_used == pytest.approx(exact_dist.mean_hands_used)

    def test_auto_mode_matches_beam_for_long_budget(self) -> None:
        board = _seed_board_ready_to_fire()
        budget = BEAM_ROLLOUT_AVG_STEP_TIME_SEC * (EXACT_SHALLOW_MAX_DEPTH + 5)
        known_pairs = ((COLOR_RED, COLOR_BLUE),)
        auto_dist = estimate_counter_distribution(
            board, budget, known_pairs=known_pairs, n_rollouts=3, rollout_mode="auto",
            beam_width=10,
        )
        beam_dist = estimate_counter_distribution(
            board, budget, known_pairs=known_pairs, n_rollouts=3, rollout_mode="beam",
            beam_width=10,
        )
        assert auto_dist.mean == pytest.approx(beam_dist.mean)

    def test_auto_mode_without_beam_width_raises(self) -> None:
        """'auto' は beam に振り分けられる可能性があるため beam_width を
        常に必須とする (振り分け結果に関わらず事前に要求、安全側)。
        """
        board = _seed_board_ready_to_fire()
        with pytest.raises(ValueError, match="beam_width"):
            estimate_counter_distribution(
                board, time_budget_sec=1.5, n_rollouts=2, rollout_mode="auto",
            )


class TestEarlyExitAtThreshold:
    """v5 user指示②「答えを変えない打ち切り」。確率チャネル
    (`prob_at_least`) は打ち切り有無に関わらず完全一致すること
    (message3 受け入れ条件1②)。分布 (mean/p25/p75) は一致を要求しない
    (仕様通り、打ち切り時点の下限値になる)。
    """

    def test_score_threshold_inversion_is_exact(self) -> None:
        """`_ojama_threshold_to_score_threshold` が `_score_to_ojama_count`
        の厳密な逆変換になっているか (境界値で直接確認)。
        """
        elapsed_sec = 0.0
        for ojama_threshold in (1.0, 5.0, 12.0, 12.5, 100.0):
            score_threshold = _ojama_threshold_to_score_threshold(ojama_threshold, elapsed_sec)
            # score_threshold-1 はまだ閾値未満、score_threshold は閾値以上のはず
            below = float(_score_to_ojama_count(float(score_threshold - 1), elapsed_sec))
            at = float(_score_to_ojama_count(float(score_threshold), elapsed_sec))
            assert below < math.ceil(ojama_threshold)
            assert at >= math.ceil(ojama_threshold)

    @pytest.mark.parametrize("rollout_mode,extra_kwargs", [
        ("beam", {"beam_width": 20}),
        ("exact_shallow", {}),
    ])
    def test_prob_at_least_matches_regardless_of_early_exit(
        self, rollout_mode: str, extra_kwargs: dict,
    ) -> None:
        board = _seed_two_red_stack_board()
        known_pairs = ((COLOR_RED, COLOR_GREEN), (COLOR_RED, COLOR_GREEN))
        thresholds = (0.0, 1.0, 5.0)
        common = dict(
            board=board, time_budget_sec=3.0, known_pairs=known_pairs, n_rollouts=15,
            thresholds_ojama=thresholds, rollout_mode=rollout_mode, **extra_kwargs,
        )
        with_exit = estimate_counter_distribution(**common, early_exit_at_threshold=True)
        without_exit = estimate_counter_distribution(**common, early_exit_at_threshold=False)
        for th in thresholds:
            assert with_exit.prob_at_least[th] == pytest.approx(without_exit.prob_at_least[th]), (
                f"threshold={th}: with_exit={with_exit.prob_at_least[th]} "
                f"without_exit={without_exit.prob_at_least[th]}"
            )

    def test_early_exit_disabled_when_thresholds_empty(self) -> None:
        """thresholds_ojama が空だと打ち切り基準が無いため無効化され、
        early_exit_at_threshold=True でも通常と完全一致すること。
        """
        board = _seed_two_red_stack_board()
        known_pairs = ((COLOR_RED, COLOR_GREEN),)
        with_flag = estimate_counter_distribution(
            board, time_budget_sec=2.0, known_pairs=known_pairs, n_rollouts=5,
            rollout_mode="exact_shallow", early_exit_at_threshold=True,
        )
        without_flag = estimate_counter_distribution(
            board, time_budget_sec=2.0, known_pairs=known_pairs, n_rollouts=5,
            rollout_mode="exact_shallow", early_exit_at_threshold=False,
        )
        assert with_flag.mean == pytest.approx(without_flag.mean)


class TestSeedFrontierRegression:
    """v5 user指示① 初期集団の質を上げる (message3 受け入れ条件1①)。

    幅を十分広く (探索空間全体を上回る) 取れば、exact_shallow の完全探索
    結果をビームサーチの初期集団として使っても・使わなくても最終結果は
    完全一致するはず (打ち切りが起きない極限では「初期集団の質」自体が
    結果に影響しないことの regression確認、①の効果自体は幅を絞った場面で
    測る、`scripts/_bench_counter_beam_rollout_2026-08-21.py` 参照)。
    """

    def test_seeded_matches_unseeded_at_exhaustive_width(self) -> None:
        board = _seed_two_red_stack_board()
        pairs = [(COLOR_RED, COLOR_GREEN)] * 4
        exhaustive_width = 22 ** 4  # 深さ4を打ち切らない幅

        from src.puyo_core_bridge import beam_search, beam_search_continue, exact_shallow_search as ess

        unseeded = beam_search(
            board, pairs, exhaustive_width, exclude_hidden_row_from_pop=True, use_exact_score=True,
        )

        seed = ess(board, pairs[:EXACT_SHALLOW_SEED_DEPTH], exclude_hidden_row_from_pop=True)
        seeded = beam_search_continue(
            seed.final_frontier, seed.best_score, pairs[EXACT_SHALLOW_SEED_DEPTH:],
            exhaustive_width, exclude_hidden_row_from_pop=True, use_exact_score=True,
        )
        assert seeded.best_score == unseeded.best_score

    def test_rollout_once_beam_matches_plain_beam_search_when_width_ge_22(self) -> None:
        """**実測で判明した重要な事実**: `EXACT_SHALLOW_SEED_DEPTH=2` の下では
        22配置が常に seed_depth=1 の全候補数なので、`beam_width>=22`
        (実務上ほぼ常にこの範囲) では seed の有無で最終結果が**完全に
        一致する** (`_truncate_frontier_by_running_best` による絞り込みが
        `beam_search` 自身の depth1 での非絞り込みと同じ結果になるため)。
        つまり ① の「質が上がる」効果は beam_width<22 の範囲でのみ生じる
        (`scripts/_bench_counter_beam_speedups_2026-08-21.py` の実測で
        beam_width=30/100 では改善0/15件・beam_width=10では改善5/15件
        だが速度は逆に1.5〜3倍遅くなることを確認済み — ①は「コストが
        増えない」という前提が崩れており、費用対効果は薄いことをuserに
        報告する)。本テストはその regression 確認 (production 経路
        `_rollout_once_beam` を直接使う)。
        """
        from src.puyo_core_bridge import beam_search

        board = _seed_two_red_stack_board()
        known_pairs = ((COLOR_RED, COLOR_GREEN),) * 6
        beam_width = 30  # >= 22 (22配置の総数)
        rng = random.Random(0)
        outcome = _rollout_once_beam(
            board, time_budget_sec=BEAM_ROLLOUT_AVG_STEP_TIME_SEC * 6, colors=(1, 2, 3, 4),
            known_pairs=known_pairs, rng=rng, elapsed_sec=0.0, beam_width=beam_width,
        )
        pairs = [(COLOR_RED, COLOR_GREEN)] * 6
        reference = beam_search(
            board, pairs, beam_width, exclude_hidden_row_from_pop=False, use_exact_score=True,
        )
        reference_ojama = float(_score_to_ojama_count(float(reference.best_score), 0.0))
        assert outcome.achieved_ojama == pytest.approx(reference_ojama)

    def test_truncate_frontier_keeps_top_n_by_running_best(self) -> None:
        from scripts.mc_counter_estimator import _truncate_frontier_by_running_best
        from src.puyo_core_bridge import FrontierEntry

        entries = [FrontierEntry(board=Board(), running_best=v) for v in (5, 1, 9, 3, 7)]
        top3 = _truncate_frontier_by_running_best(entries, 3)
        assert [e.running_best for e in top3] == [9, 7, 5]


class TestDepth1To2FrontierSharing:
    """v5 user指示④ 深さ1〜2限定の部分木共有 (`_lookup_or_compute_
    exact_shallow_seed`)。キャッシュの有無で結果が変わらないこと (答えを
    変えない共有であることの確認)。
    """

    def test_cache_hit_matches_fresh_computation(self) -> None:
        board = _seed_two_red_stack_board()
        seed_pairs = (COLOR_RED, COLOR_GREEN), (COLOR_RED, COLOR_GREEN)
        cache: dict = {}
        first = _lookup_or_compute_exact_shallow_seed(board, seed_pairs, None, cache)
        second = _lookup_or_compute_exact_shallow_seed(board, seed_pairs, None, cache)
        assert first.best_score == second.best_score
        assert len(cache) == 1, "同じ接頭辞は1件にまとまるはず"

    def test_canonical_pair_order_invariance(self) -> None:
        """(赤,緑) と (緑,赤) は到達可能盤面集合が同一なので、キャッシュ
        キーとして同一視されるはず (`_canonical_pair` の数学的根拠通り)。
        """
        assert _canonical_pair((COLOR_RED, COLOR_GREEN)) == _canonical_pair((COLOR_GREEN, COLOR_RED))

    def test_beam_rollout_seed_cache_reduces_native_calls(self, monkeypatch) -> None:
        """`seed_cache` を渡すと既知ツモ (乱数なし、常に同一接頭辞) の
        exact_shallow_search 呼び出しが1回に減ること (呼び出し回数を
        直接計装して確認)。
        """
        import scripts.mc_counter_estimator as mc_mod

        board = _seed_two_red_stack_board()
        call_count = {"n": 0}
        original = mc_mod._native_exact_shallow_search

        def counting_wrapper(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(mc_mod, "_native_exact_shallow_search", counting_wrapper)

        cache: dict = {}
        for seed in range(5):
            rng = random.Random(seed)
            _rollout_once_beam(
                board, time_budget_sec=BEAM_ROLLOUT_AVG_STEP_TIME_SEC * 6, colors=(1, 2, 3, 4),
                known_pairs=((COLOR_RED, COLOR_GREEN), (COLOR_RED, COLOR_GREEN)), rng=rng,
                elapsed_sec=0.0, beam_width=10, seed_cache=cache,
            )
        # 既知ツモ2手は乱数不使用で全ロールアウト共通のため、キャッシュ
        # ヒットにより exact_shallow_search 呼び出しは1回だけになるはず。
        assert call_count["n"] == 1


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
