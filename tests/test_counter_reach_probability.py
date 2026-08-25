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


def test_multi_threshold_matches_individual_calls_exactly() -> None:
    """thresholds_ojama をまとめて渡した結果は、各閾値を個別に単独呼び

    出しした結果と全K・全確率で完全一致する (S1b 2026-08-21、近似ゼロの
    半減。浮動小数の完全一致を要求する — 近似許容ではない)。
    """
    board = Board()
    for row in range(5, 13):
        board.set(row, 0, COLOR_RED)
        board.set(row, 2, COLOR_BLUE)
    th_opp_fire = 6.0
    th_forecast = 18.0

    individual_a = counter_reach_probability(board, th_opp_fire, rng_seed=42)
    individual_b = counter_reach_probability(board, th_forecast, rng_seed=42)
    combined = counter_reach_probability(
        board, th_opp_fire, rng_seed=42, thresholds_ojama=(th_forecast,),
    )

    assert isinstance(combined, dict)
    assert set(combined.keys()) == {th_opp_fire, th_forecast}
    for k in (1, 2, 3, 4):
        assert combined[th_opp_fire].probabilities[k] == individual_a.probabilities[k]
        assert combined[th_opp_fire].n_evaluated[k] == individual_a.n_evaluated[k]
        assert combined[th_forecast].probabilities[k] == individual_b.probabilities[k]
        assert combined[th_forecast].n_evaluated[k] == individual_b.n_evaluated[k]


def test_multi_threshold_none_default_unchanged_return_type() -> None:
    """thresholds_ojama 省略時 (既定 None) は従来どおり CounterReachResult

    単体を返し、値も従来と同一 (完全後方互換、既存呼び出しは無変更)。
    """
    board = _empty_ish_board()
    result = counter_reach_probability(board, threshold_ojama=6.0, rng_seed=1)
    assert isinstance(result, CounterReachResult)
    assert not isinstance(result, dict)


def test_multi_threshold_dead_board_returns_dict_of_empty_results() -> None:
    """窒息盤面 + 複数閾値でも dict 形式・全閾値ぶん0.0で返る。"""
    combined = counter_reach_probability(
        _dead_board(), threshold_ojama=6.0, thresholds_ojama=(12.0, 24.0),
    )
    assert set(combined.keys()) == {6.0, 12.0, 24.0}
    for result in combined.values():
        for k, p in result.probabilities.items():
            assert p == 0.0


def test_multi_threshold_duplicate_values_collapse_to_one_entry() -> None:
    """threshold_ojama と thresholds_ojama に同じ値が重複しても1エントリに

    統合される (dict.fromkeys の重複除去)。
    """
    board = _empty_ish_board()
    combined = counter_reach_probability(
        board, threshold_ojama=6.0, rng_seed=3, thresholds_ojama=(6.0, 12.0),
    )
    assert set(combined.keys()) == {6.0, 12.0}


def test_multi_threshold_speed_roughly_halves_vs_two_individual_calls() -> None:
    """「2閾値を個別に2回呼ぶ」と「1回でまとめる」の壁時間を比較し、

    まとめ呼びがおおむね半減していることを実測する (cProfile 禁止、
    perf_counter のみ使用、2026-08-21 S1b 受け入れ条件)。

    40件フル実測は `scripts/_bench_counter_reach_multi_threshold_2026-08-21.py`
    に分離した (project既存の _bench_* 系ベンチスクリプトの慣習に合わせ、
    pytest 本体を170秒級で重くしないため)。本テストは pytest 常設用に
    件数を5件へ縮小した縮小版で、同じ構造的性質 (シミュレーション自体は
    閾値に無依存) が壊れていないことだけを軽量に見張る。
    """
    import time

    boards: list[Board] = []
    rng = np.random.default_rng(0)
    for i in range(5):
        board = Board()
        n_puyo = 6 + int(rng.integers(0, 6))
        cols = rng.integers(0, 6, size=n_puyo)
        for j, col in enumerate(cols):
            color = 1 + int((i + j) % 4)
            row = 12 - int(j % 3)
            board.set(row, int(col), color)
        boards.append(board)
    th_a, th_b = 6.0, 18.0

    t0 = time.perf_counter()
    for board in boards:
        counter_reach_probability(board, th_a, rng_seed=1)
        counter_reach_probability(board, th_b, rng_seed=1)
    t_individual = time.perf_counter() - t0

    t0 = time.perf_counter()
    for board in boards:
        counter_reach_probability(board, th_a, rng_seed=1, thresholds_ojama=(th_b,))
    t_combined = time.perf_counter() - t0

    print(
        f"\n[INFO] 5件(縮小版) 個別2回呼び={t_individual:.3f}s "
        f"まとめ1回呼び={t_combined:.3f}s "
        f"比率={t_combined / t_individual:.3f}",
    )
    # 完全な0.5にはノイズで届かないため、余裕を持った上限0.75で「半減方向」
    # のみ確認する (過学習/シーン逆算的な厳しい閾値は設けない)。
    assert t_combined < t_individual * 0.75, (
        f"まとめ呼びが個別2回呼びの75%未満に短縮されていない: "
        f"individual={t_individual:.3f}s combined={t_combined:.3f}s"
    )


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
