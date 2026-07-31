"""src/indicators_v2.py の counter_reach_probability / _fast (#24 Step2) 回帰テスト。

打ち合い計測器 Step2「有効性判定MC」の核となる新規関数の検証。
既存 expected_fire_power (XVI) は一切変更していないため、そのテストへの
影響はない (本ファイルは新規追加のみ)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import Board, COLOR_RED, COLOR_BLUE, DEATH_COL, DEATH_ROW
from src.chain import ChainSimulator
from src.indicators_v2 import (
    COUNTER_REACH_EFFECTIVE_THRESHOLD_PROB,
    CounterReachResult,
    counter_reach_probability,
    counter_reach_probability_fast,
)


def _dead_board() -> Board:
    """窒息判定セル (DEATH_ROW, DEATH_COL) にぷよを積んだ窒息盤面。"""
    board = Board()
    board.set(DEATH_ROW, DEATH_COL, COLOR_RED)
    assert board.is_dead(), "テスト構築ミス: 窒息盤面になっていない"
    return board


def _empty_ish_board() -> Board:
    """ほぼ空盤面 (少量のぷよのみ)。"""
    board = Board()
    board.set(12, 0, COLOR_RED)
    board.set(12, 1, COLOR_BLUE)
    return board


# ============================
# 基本契約 (0〜1範囲, 窒息, 閾値<=0)
# ============================


@pytest.mark.parametrize("fn", [counter_reach_probability, counter_reach_probability_fast])
def test_dead_board_returns_zero_for_all_k(fn) -> None:
    """窒息盤面 (応手不能) は全K で確率0.0。"""
    result = fn(_dead_board(), threshold_ojama=6.0)
    assert isinstance(result, CounterReachResult)
    for k, p in result.probabilities.items():
        assert p == 0.0, f"K={k} で0でない: {p}"
        assert result.n_evaluated[k] == 0


@pytest.mark.parametrize("fn", [counter_reach_probability, counter_reach_probability_fast])
def test_threshold_zero_or_below_is_always_reachable(fn) -> None:
    """閾値<=0 は (窒息でない限り) 常に到達可能=確率1.0。"""
    result = fn(_empty_ish_board(), threshold_ojama=0.0)
    for k, p in result.probabilities.items():
        assert p == pytest.approx(1.0), f"K={k} で1.0でない: {p}"


@pytest.mark.parametrize("fn", [counter_reach_probability, counter_reach_probability_fast])
def test_probabilities_are_in_unit_range(fn) -> None:
    """全てのK水準で確率は0〜1に収まる。"""
    board = Board()
    for row in range(9, 13):
        board.set(row, 0, COLOR_RED)
        board.set(row, 5, COLOR_BLUE)
    result = fn(board, threshold_ojama=12.0)
    for p in result.probabilities.values():
        assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("fn", [counter_reach_probability, counter_reach_probability_fast])
def test_probability_monotonic_non_increasing_in_threshold(fn) -> None:
    """閾値を上げるほど到達確率は単調非増加になる (同一rng_seed固定で再現性確保)。"""
    board = Board()
    for row in range(6, 13):
        board.set(row, 0, COLOR_RED)
        board.set(row, 1, COLOR_BLUE)
    thresholds = [0.0, 6.0, 12.0, 24.0, 48.0, 96.0]
    prev = {k: 1.1 for k in (1, 2, 3, 4)}
    for th in thresholds:
        result = fn(board, threshold_ojama=th, rng_seed=12345)
        for k, p in result.probabilities.items():
            assert p <= prev[k] + 1e-9, f"K={k} th={th} で単調性が崩れた: prev={prev[k]} cur={p}"
            prev[k] = p


@pytest.mark.parametrize("fn", [counter_reach_probability, counter_reach_probability_fast])
def test_n_evaluated_matches_expected_counts(fn) -> None:
    """n_evaluated は K=1: len(colors)^2, K=2: len(colors)^4, K=3,4: mc_n_samples になる。

    active_colors を明示 (4色) しないと `_empty_ish_board` は出現色が
    NEAR_FUTURE_MIN_OBSERVED_COLORS(4) 未満のため5色フォールバックが
    発動する (_near_future_active_colors 仕様)。本テストは組み合わせ数の
    契約を見たいので active_colors を明示して固定する。
    """
    board = _empty_ish_board()
    mc_n = 8
    result = fn(
        board, threshold_ojama=6.0, mc_n_samples=mc_n, active_colors=(1, 2, 3, 4),
    )
    assert result.n_evaluated[1] == 16  # 4色^2
    assert result.n_evaluated[2] == 256  # 4色^4
    assert result.n_evaluated[3] == mc_n
    assert result.n_evaluated[4] == mc_n


@pytest.mark.parametrize("fn", [counter_reach_probability, counter_reach_probability_fast])
def test_stateless_same_board_same_seed_reproducible(fn) -> None:
    """rng_seed省略時 (盤面から自動導出) でも同一盤面には常に同一結果 (stateless原則)。"""
    board = _empty_ish_board()
    r1 = fn(board, threshold_ojama=6.0)
    r2 = fn(board, threshold_ojama=6.0)
    assert r1.probabilities == r2.probabilities
    # 呼び出し前後で board 自体が破壊されていないことも確認
    assert board.count_puyos() == 2


@pytest.mark.parametrize("fn", [counter_reach_probability, counter_reach_probability_fast])
def test_k_levels_subset_only_computes_requested(fn) -> None:
    """k_levels を絞ると、その水準のみ結果に含まれる。"""
    board = _empty_ish_board()
    result = fn(board, threshold_ojama=6.0, k_levels=(1, 3))
    assert set(result.probabilities.keys()) == {1, 3}
    assert set(result.n_evaluated.keys()) == {1, 3}


# ============================
# active_colors (4色前提) の伝播確認
# ============================


def test_active_colors_restricts_pair_generation() -> None:
    """active_colors の色数^2 が K=1 の評価件数になる (1色なら1通り、4色なら16通り)。"""
    board = _empty_ish_board()
    result_1color = counter_reach_probability(
        board, threshold_ojama=6.0, active_colors=(COLOR_RED,),
    )
    result_4color = counter_reach_probability(
        board, threshold_ojama=6.0, active_colors=(1, 2, 3, 4),
    )
    assert result_1color.n_evaluated[1] == 1
    assert result_4color.n_evaluated[1] == 16


# ============================
# precise / fast モードの整合性 (近似の妥当性)
# ============================


def test_fast_mode_never_exceeds_precise_mode_probability_no_pending_match() -> None:
    """4連結が既に成立していない (=有効なSTABLE想定) 盤面では、fast の到達確率は

    precise 以下になる (連結ボーナス0近似は常に過小評価方向、厳密な不等号)。

    ⚠️ 手作り盤面に既存の4連結未解消グループを含めると (実際にはあり得ない
    非STABLE入力)、新規ツモがそこへ連結し巨大グループ (5+連結) を作る
    ケースで乖離が大きくなることを実装中に確認した (scratchpad調査、
    exact=600 vs approx=300 等、6連結の連結ボーナス+3分がまるごと近似
    から抜けるため)。本テストは「4連結が0個 (=STABLE)」の妥当な入力に
    絞ることで、その極端ケースを避けて構造的性質 (方向性のみ) を確認する。
    """
    boards = []
    b1 = Board()
    b1.set(12, 0, COLOR_RED)
    b1.set(12, 3, COLOR_BLUE)
    b1.set(11, 3, COLOR_RED)
    boards.append(b1)

    b2 = Board()
    b2.set(12, 0, COLOR_RED)
    b2.set(11, 0, COLOR_RED)
    b2.set(12, 5, COLOR_BLUE)
    b2.set(11, 5, COLOR_BLUE)
    boards.append(b2)

    sim = ChainSimulator()
    for board in boards:
        assert sim.simulate(board).chain_count == 0, "テスト構築ミス: 既に4連結が成立している"
        precise = counter_reach_probability(board, threshold_ojama=6.0, rng_seed=7)
        fast = counter_reach_probability_fast(board, threshold_ojama=6.0, rng_seed=7)
        for k in (1, 2, 3, 4):
            diff = precise.probabilities[k] - fast.probabilities[k]
            assert diff >= -1e-9, f"K={k} で fast が precise を上回った (近似の想定外れ): {diff}"


def test_fast_mode_approximation_direction_on_real_stable_boards() -> None:
    """実データ (STABLE確定盤面) 複数件で fast<=precise の方向性を確認する。

    ⚠️ 正直な注記 (実装中に scratchpad で確認した事実): 5+連結が生まれ
    やすい盤面では乖離が大きくなりうる (連結ボーナス分がまるごと近似から
    抜けるため、例: 6連結で厳密600点 vs 近似300点)。本テストは方向性の
    保証のみを行い、乖離の大きさそのものは断定しない (npz データが無い
    環境では skip)。
    """
    npz_path = Path("data/indicators_v2/boards_lean_fixed/c62.npz")
    if not npz_path.exists():
        pytest.skip(f"{npz_path} が存在しない (npz キャッシュ未生成)")
    data = np.load(str(npz_path), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(3)
    n = min(10, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)

    checked = 0
    max_gap = 0.0
    for i in idxs:
        board = Board.from_list(grids[i].tolist())
        if board.is_dead() or board.count_puyos() < 10:
            continue
        precise = counter_reach_probability(board, threshold_ojama=6.0, rng_seed=7)
        fast = counter_reach_probability_fast(board, threshold_ojama=6.0, rng_seed=7)
        for k in (1, 2, 3, 4):
            diff = precise.probabilities[k] - fast.probabilities[k]
            assert diff >= -1e-9, f"K={k} で fast が precise を上回った (近似の想定外れ): {diff}"
            max_gap = max(max_gap, diff)
        checked += 1
    if checked == 0:
        pytest.skip("評価対象盤面が0件だった")
    # 参考情報として最大乖離を表示する (アサーションではなく記録目的)。
    print(f"\n[INFO] 実盤面{checked}件でのprecise-fast最大乖離: {max_gap:.3f}")


# ============================
# 計算コスト実測 (簡易、CIでの目安値)
# ============================


def test_precise_and_fast_complete_within_generous_timeout() -> None:
    """1イベントあたりの計算が異常に暴走しないことの粗いタイムアウト確認

    (詳細な実測値は scratchpad ベンチスクリプトで別途計測、正式な性能
    要件ではなく「固まらない」ことの確認)。
    """
    import time

    board = Board()
    for row in range(4, 13):
        board.set(row, 0, COLOR_RED)
        board.set(row, 3, COLOR_BLUE)

    t0 = time.perf_counter()
    counter_reach_probability(board, threshold_ojama=12.0)
    t1 = time.perf_counter()
    counter_reach_probability_fast(board, threshold_ojama=12.0)
    t2 = time.perf_counter()

    assert (t1 - t0) < 5.0, f"precise mode が想定外に遅い: {(t1-t0):.2f}s"
    assert (t2 - t1) < 5.0, f"fast mode が想定外に遅い: {(t2-t1):.2f}s"
