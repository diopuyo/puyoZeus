"""src/ojama_score_inferrer.py のテスト。

得点ベースで予告お邪魔ぷよ数を推定するロジックの動作を検証する。
"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.chain import ChainResult, ChainSimulator, ChainStep, PuyoGroup
from src.ojama_score_inferrer import (
    OjamaPrediction,
    OjamaScoreInferrer,
    SIDE_1P,
    SIDE_2P,
)
from src.scoring import (
    ALL_CLEAR_BONUS,
    MARGIN_TIME_START_SEC,
    OJAMA_RATE_STANDARD,
    calculate_chain_score,
    score_to_ojama,
)


# ============================
# テスト用ヘルパ
# ============================


def _empty_grid() -> list[list[int]]:
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def _empty_board() -> Board:
    return Board.from_list(_empty_grid())


def _fake_step_4connect_1color(chain_idx: int) -> ChainStep:
    """4 連結 1 色のみの ChainStep を擬似生成 (conn=0, color=0)。"""
    cells = frozenset({(0, c) for c in range(4)})
    group = PuyoGroup(
        color=COLOR_RED,
        cells=cells,
        size=4,
        ojama_adjacent=frozenset(),
    )
    return ChainStep(
        chain_index=chain_idx,
        erased_groups=[group],
        erased_ojama=0,
        erased_count=4,
        board_before=_empty_board(),
        board_after=_empty_board(),
    )


def _fake_chain_result(chain_count: int, *, all_clear: bool = False) -> ChainResult:
    """指定連鎖数 (各ステップ 4 連結 1 色) の ChainResult を擬似生成。

    Args:
        chain_count: 連鎖段数 (>=1)。
        all_clear: 連鎖後盤面を全消し状態 (空盤面) にするか。
            デフォルトは False で、final_board に 1 puyo を残して
            is_all_clear=False とする。
    """
    steps = [_fake_step_4connect_1color(i + 1) for i in range(chain_count)]
    if all_clear:
        final = _empty_board()
    else:
        # 残ぷよ 1 個を入れて全消しにならないようにする
        grid = _empty_grid()
        grid[12][5] = COLOR_BLUE
        final = Board.from_list(grid)
    total_erased = 4 * chain_count
    return ChainResult(
        steps=steps,
        chain_count=chain_count,
        total_erased=total_erased,
        total_ojama=0,
        final_board=final,
        participating_cells=total_erased,
    )


# ============================
# テスト 1: 4 連鎖 → 公式表通りの予告お邪魔個数
# ============================


def test_four_chain_official_pending_count() -> None:
    """
    4 連鎖、各ステップ 4 連結 1 色の素点:
        step1: CP=0,  mult=max(1,0)=1,  score=40
        step2: CP=8,  mult=8,           score=320
        step3: CP=16, mult=16,          score=640
        step4: CP=32, mult=32,          score=1280
        total = 2280
    通常レート 70 で 2280 / 70 = 32 個、余り 40
    """
    inferrer = OjamaScoreInferrer()
    cr = _fake_chain_result(4)
    pred, lo1, lo2 = inferrer.infer_from_chain_event(
        cr,
        fired_by=SIDE_1P,
        match_elapsed_sec=10.0,  # マージンタイム前
        prev_leftover_1p=0,
        prev_leftover_2p=0,
    )
    assert pred.chain_length == 4
    assert pred.base_score == 2280
    assert pred.total_score == 2280
    assert pred.pending == 32
    assert pred.side == SIDE_2P  # 1P 発火 → 2P 受け
    assert pred.fired_by_side == SIDE_1P
    assert pred.effective_rate == OJAMA_RATE_STANDARD
    # 送り手 (1P) のみ leftover が更新され、2P はそのまま
    assert lo1 == 40
    assert lo2 == 0


# ============================
# テスト 2: 全消し付与で +2100 個数増加
# ============================


def test_all_clear_bonus_increases_pending() -> None:
    """
    1P が全消しを持ち越した状態で 4 連鎖発火:
        effective_score = 2280 + 2100 = 4380
        4380 / 70 = 62 個、余り 40
    全消し持越しなしの場合 (32 個) より明確に多い。
    """
    inferrer = OjamaScoreInferrer()
    cr = _fake_chain_result(4)

    # 全消し持越し「なし」基準
    pred_no_ac, _, _ = inferrer.infer_from_chain_event(
        cr,
        fired_by=SIDE_1P,
        match_elapsed_sec=10.0,
        prev_leftover_1p=0,
        prev_leftover_2p=0,
        all_clear_pending_1p=False,
    )

    # 全消し持越し「あり」
    pred_ac, lo1, _ = inferrer.infer_from_chain_event(
        cr,
        fired_by=SIDE_1P,
        match_elapsed_sec=10.0,
        prev_leftover_1p=0,
        prev_leftover_2p=0,
        all_clear_pending_1p=True,
    )

    assert pred_ac.all_clear_bonus_applied == ALL_CLEAR_BONUS
    assert pred_ac.total_score == 2280 + ALL_CLEAR_BONUS
    assert pred_ac.pending == 62
    assert pred_ac.pending > pred_no_ac.pending
    assert lo1 == 40


# ============================
# テスト 3: マージンタイム発動で同じ連鎖でも個数が異なる
# ============================


def test_margin_time_changes_pending_for_same_chain() -> None:
    """
    96 秒以前と以後で同じ 4 連鎖の予告個数が変わる。
    マージン後はレートが下がる (70 → 52 → ...) ので個数が増える。
    """
    inferrer = OjamaScoreInferrer()
    cr = _fake_chain_result(4)

    pred_before, _, _ = inferrer.infer_from_chain_event(
        cr,
        fired_by=SIDE_1P,
        match_elapsed_sec=MARGIN_TIME_START_SEC - 1.0,
        prev_leftover_1p=0,
        prev_leftover_2p=0,
    )
    pred_after, _, _ = inferrer.infer_from_chain_event(
        cr,
        fired_by=SIDE_1P,
        match_elapsed_sec=MARGIN_TIME_START_SEC + 1.0,
        prev_leftover_1p=0,
        prev_leftover_2p=0,
    )

    # マージン前は標準 70、後は 52 (= 70 * 0.75 切り捨て)
    assert pred_before.effective_rate == OJAMA_RATE_STANDARD
    assert pred_after.effective_rate == 52
    # 同じ score=2280 でも、レートが下がる方が個数が多い
    assert pred_after.pending > pred_before.pending
    # 2280 / 52 = 43 個 (余り 44)
    assert pred_after.pending == 43


# ============================
# テスト 4: 連続発火で leftover が正しく繰越
# ============================


def test_leftover_carries_across_consecutive_fires() -> None:
    """
    1P が 1 連鎖 (40 点) を 2 回連続で発火:
        1 回目: 40 + 0 = 40, ojama=0, leftover=40
        2 回目: 40 + 40 = 80, ojama=1, leftover=10
    """
    inferrer = OjamaScoreInferrer()
    cr1 = _fake_chain_result(1)  # score=40

    pred1, lo1_a, lo2_a = inferrer.infer_from_chain_event(
        cr1,
        fired_by=SIDE_1P,
        match_elapsed_sec=10.0,
        prev_leftover_1p=0,
        prev_leftover_2p=0,
    )
    assert pred1.pending == 0
    assert lo1_a == 40
    assert lo2_a == 0

    pred2, lo1_b, lo2_b = inferrer.infer_from_chain_event(
        cr1,
        fired_by=SIDE_1P,
        match_elapsed_sec=11.0,
        prev_leftover_1p=lo1_a,
        prev_leftover_2p=lo2_a,
    )
    # (40 + 40) / 70 = 1, 余り 10
    assert pred2.pending == 1
    assert lo1_b == 10
    assert lo2_b == 0


def test_leftover_carries_via_timeline() -> None:
    """同じシナリオを infer_timeline で実行しても結果が一致する。"""
    inferrer = OjamaScoreInferrer()
    cr1 = _fake_chain_result(1)
    events = [
        (10.0, SIDE_1P, cr1, False),
        (11.0, SIDE_1P, cr1, False),
    ]
    preds = inferrer.infer_timeline(events, match_start_sec=0.0)
    assert len(preds) == 2
    assert preds[0].pending == 0
    assert preds[1].pending == 1
    # 内部状態としても 1P leftover が 10 で残る
    assert inferrer.leftover_1p == 10
    assert inferrer.leftover_2p == 0


# ============================
# テスト 5: 1P/2P 双方向の side 割り振り
# ============================


def test_side_assignment_1p_fires_2p_receives() -> None:
    """1P 発火 → 受け側は 2P。送り手 1P の leftover のみ更新。"""
    inferrer = OjamaScoreInferrer()
    cr = _fake_chain_result(4)
    pred, lo1, lo2 = inferrer.infer_from_chain_event(
        cr,
        fired_by=SIDE_1P,
        match_elapsed_sec=10.0,
        prev_leftover_1p=0,
        prev_leftover_2p=0,
    )
    assert pred.fired_by_side == SIDE_1P
    assert pred.side == SIDE_2P
    assert lo1 == 40
    assert lo2 == 0


def test_side_assignment_2p_fires_1p_receives() -> None:
    """2P 発火 → 受け側は 1P。送り手 2P の leftover のみ更新。"""
    inferrer = OjamaScoreInferrer()
    cr = _fake_chain_result(4)
    pred, lo1, lo2 = inferrer.infer_from_chain_event(
        cr,
        fired_by=SIDE_2P,
        match_elapsed_sec=10.0,
        prev_leftover_1p=0,
        prev_leftover_2p=0,
    )
    assert pred.fired_by_side == SIDE_2P
    assert pred.side == SIDE_1P
    assert lo1 == 0
    assert lo2 == 40


def test_timeline_alternating_sides() -> None:
    """1P/2P 交互発火で leftover も独立に管理される。"""
    inferrer = OjamaScoreInferrer()
    cr = _fake_chain_result(1)  # score=40
    events = [
        (10.0, SIDE_1P, cr, False),
        (11.0, SIDE_2P, cr, False),
        (12.0, SIDE_1P, cr, False),
        (13.0, SIDE_2P, cr, False),
    ]
    preds = inferrer.infer_timeline(events, match_start_sec=0.0)
    assert [p.fired_by_side for p in preds] == [SIDE_1P, SIDE_2P, SIDE_1P, SIDE_2P]
    assert [p.side for p in preds] == [SIDE_2P, SIDE_1P, SIDE_2P, SIDE_1P]
    # 各側 2 回ずつ 40 点 → 80 点 → 1 個 + leftover 10
    # 1 回目はどちらも leftover 40、2 回目で 1 個発生
    pendings_1p_to_2p = [p.pending for p in preds if p.fired_by_side == SIDE_1P]
    pendings_2p_to_1p = [p.pending for p in preds if p.fired_by_side == SIDE_2P]
    assert pendings_1p_to_2p == [0, 1]
    assert pendings_2p_to_1p == [0, 1]
    # 最終 leftover はそれぞれ 10
    assert inferrer.leftover_1p == 10
    assert inferrer.leftover_2p == 10


# ============================
# テスト: 異常系
# ============================


def test_invalid_side_raises() -> None:
    """fired_by に '1P'/'2P' 以外を渡すと ValueError。"""
    inferrer = OjamaScoreInferrer()
    cr = _fake_chain_result(1)
    with pytest.raises(ValueError):
        inferrer.infer_from_chain_event(
            cr,
            fired_by="P1",  # noqa: typo
            match_elapsed_sec=0.0,
            prev_leftover_1p=0,
            prev_leftover_2p=0,
        )


# ============================
# テスト: 全消し持越しフラグの遷移 (タイムライン)
# ============================


def test_timeline_all_clear_pending_propagates() -> None:
    """
    1P が全消し連鎖を発火 → 次の 1P 連鎖で 2100 ボーナスが乗る。
    2P 側の全消しフラグは別管理で影響を受けない。
    """
    inferrer = OjamaScoreInferrer()
    cr_ac = _fake_chain_result(1, all_clear=True)        # is_all_clear=True
    cr_normal = _fake_chain_result(1, all_clear=False)   # is_all_clear=False

    events = [
        (10.0, SIDE_1P, cr_ac, False),       # 1P 全消し連鎖
        (11.0, SIDE_2P, cr_normal, False),   # 2P が連鎖 (1P の AC 持越しは消えない)
        (12.0, SIDE_1P, cr_normal, False),   # 1P が次連鎖 → 2100 ボーナス加算
    ]
    preds = inferrer.infer_timeline(events, match_start_sec=0.0)

    # 1 回目: 1P 全消し連鎖 (素点 40)、ボーナスはまだ加算されない
    assert preds[0].fired_by_side == SIDE_1P
    assert preds[0].is_all_clear is True
    assert preds[0].all_clear_bonus_applied == 0
    assert preds[0].total_score == 40

    # 2 回目: 2P 連鎖、2P 側の AC フラグは初期値 False のまま
    assert preds[1].fired_by_side == SIDE_2P
    assert preds[1].all_clear_bonus_applied == 0

    # 3 回目: 1P 連鎖、ここで全消しボーナスが加算される
    assert preds[2].fired_by_side == SIDE_1P
    assert preds[2].all_clear_bonus_applied == ALL_CLEAR_BONUS
    assert preds[2].total_score == 40 + ALL_CLEAR_BONUS


# ============================
# テスト: 実 simulate との突き合わせ
# ============================


def test_real_simulate_4connect_1chain_consistency() -> None:
    """
    実 ChainSimulator で 1 連鎖 4 連結を作り、
    OjamaScoreInferrer の出力が score_to_ojama 直叩きと一致するか確認。
    """
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    board = Board.from_list(grid)
    sim_result = ChainSimulator().simulate(board)
    assert sim_result.chain_count == 1

    inferrer = OjamaScoreInferrer()
    pred, lo1, lo2 = inferrer.infer_from_chain_event(
        sim_result,
        fired_by=SIDE_1P,
        match_elapsed_sec=0.0,
        prev_leftover_1p=0,
        prev_leftover_2p=0,
    )

    expected_score = calculate_chain_score(sim_result).total_score
    expected_ojama = score_to_ojama(expected_score, prev_leftover=0, elapsed_sec=0.0)
    assert pred.base_score == expected_score
    assert pred.total_score == expected_score
    assert pred.pending == expected_ojama.ojama_count
    assert lo1 == expected_ojama.leftover_score
    assert lo2 == 0


# =============================================================================
# infer_from_score_delta テスト (得点 OCR 差分ベース推論、Phase A3)
# =============================================================================


def test_score_delta_basic_280_to_4_ojama() -> None:
    """score 差分 280, rate=70 → ojama 4 個、leftover 0。"""
    inferrer = OjamaScoreInferrer()
    pred, lo = inferrer.infer_from_score_delta(
        score_before=0,
        score_after=280,
        fired_by=SIDE_1P,
        match_elapsed_sec=0.0,
        prev_leftover_sender=0,
    )
    assert pred.pending == 4
    assert pred.side == SIDE_2P
    assert pred.fired_by_side == SIDE_1P
    assert pred.total_score == 280
    assert pred.base_score == 280
    assert lo == 0


def test_score_delta_with_leftover_carry() -> None:
    """score 差分 75 + 前回 leftover 5 = 80 → 1 個 + leftover 10。"""
    inferrer = OjamaScoreInferrer()
    pred, lo = inferrer.infer_from_score_delta(
        score_before=1000,
        score_after=1075,
        fired_by=SIDE_2P,
        match_elapsed_sec=10.0,
        prev_leftover_sender=5,
    )
    assert pred.pending == 1
    assert lo == 10
    assert pred.side == SIDE_1P


def test_score_delta_margin_time_increases_pending() -> None:
    """同じ score 差分でも 96s 以降 (マージンタイム) は ojama 個数が増加。"""
    inferrer = OjamaScoreInferrer()
    pred_before, _ = inferrer.infer_from_score_delta(
        score_before=0, score_after=420,
        fired_by=SIDE_1P, match_elapsed_sec=50.0, prev_leftover_sender=0,
    )
    pred_after, _ = inferrer.infer_from_score_delta(
        score_before=0, score_after=420,
        fired_by=SIDE_1P, match_elapsed_sec=200.0, prev_leftover_sender=0,
    )
    assert pred_before.pending == 6  # 420 / 70
    assert pred_after.pending > pred_before.pending
    assert pred_after.effective_rate < pred_before.effective_rate


def test_score_delta_negative_treated_as_zero() -> None:
    """OCR ノイズで score_after < score_before の場合は 0 個 + leftover 維持。"""
    inferrer = OjamaScoreInferrer()
    pred, lo = inferrer.infer_from_score_delta(
        score_before=2000, score_after=1500,
        fired_by=SIDE_1P, match_elapsed_sec=30.0, prev_leftover_sender=20,
    )
    assert pred.pending == 0
    assert pred.total_score == 0
    assert lo == 20  # delta=0 + prev_leftover=20 → 0 個 + leftover 20


def test_score_delta_invalid_side_raises() -> None:
    inferrer = OjamaScoreInferrer()
    with pytest.raises(ValueError):
        inferrer.infer_from_score_delta(
            score_before=0, score_after=70,
            fired_by="3P", match_elapsed_sec=0.0, prev_leftover_sender=0,
        )


def test_timeline_from_score_series_alternating() -> None:
    """1P/2P が交互に発火する series → 両側の予測が時系列で出る。"""
    inferrer = OjamaScoreInferrer()
    series = [
        (0.0, 0, 0),
        (10.0, 0, 0),       # 変化なし → イベントなし
        (20.0, 700, 0),     # 1P 発火 (10 個)
        (30.0, 700, 350),   # 2P 発火 (5 個)
        (40.0, 1400, 350),  # 1P 再発火 (10 個)
    ]
    preds = inferrer.infer_timeline_from_score_series(series, match_start_sec=0.0)
    assert len(preds) == 3
    assert preds[0].fired_by_side == SIDE_1P and preds[0].pending == 10
    assert preds[1].fired_by_side == SIDE_2P and preds[1].pending == 5
    assert preds[2].fired_by_side == SIDE_1P and preds[2].pending == 10


def test_timeline_min_chain_score_filters_noise() -> None:
    """min_chain_score 未満の小さい差分は連鎖イベントとみなさず無視。"""
    inferrer = OjamaScoreInferrer()
    series = [
        (0.0, 0, 0),
        (5.0, 30, 20),   # 30, 20 共に min_chain_score=40 未満 → 無視
        (10.0, 100, 20),  # 1P delta=70 ≥ 40 → 採用
    ]
    preds = inferrer.infer_timeline_from_score_series(
        series, match_start_sec=0.0, min_chain_score=40,
    )
    assert len(preds) == 1
    assert preds[0].fired_by_side == SIDE_1P
    assert preds[0].pending == 1  # 100 / 70


def test_timeline_leftover_carries_per_side() -> None:
    """同サイドで連続発火すると leftover が次回に繰越される。"""
    inferrer = OjamaScoreInferrer()
    series = [
        (0.0, 0, 0),
        (10.0, 75, 0),    # 1P: 75 → 1 個 + leftover 5
        (20.0, 150, 0),   # 1P: delta 75 + leftover 5 = 80 → 1 個 + leftover 10
    ]
    preds = inferrer.infer_timeline_from_score_series(series, match_start_sec=0.0)
    assert len(preds) == 2
    assert preds[0].pending == 1
    assert preds[1].pending == 1
    # 2 回目発火後の 1P leftover は 10
    assert inferrer.leftover_1p == 10


# =============================================================================
# infer_per_step_breakdown テスト (Phase R、画面 N×M 表示と対応)
# =============================================================================


def test_per_step_breakdown_basic() -> None:
    """1 連鎖 4 連結で各 step の N (erased*10), M (bonus) が取れる。"""
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    board = Board.from_list(grid)
    sim_result = ChainSimulator().simulate(board)
    inferrer = OjamaScoreInferrer()
    breakdown = inferrer.infer_per_step_breakdown(
        sim_result, fired_by=SIDE_1P,
        match_elapsed_sec=0.0, prev_leftover_sender=0,
    )
    assert len(breakdown) >= 1
    step1 = breakdown[0]
    assert step1["step_idx"] == 1
    assert step1["erased_count"] == 4
    assert step1["n_display"] == 40
    assert step1["m_display"] >= 1


def test_per_step_breakdown_cumulative_score() -> None:
    """各 step の cumulative_score が単調増加。"""
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    board = Board.from_list(grid)
    sim_result = ChainSimulator().simulate(board)
    inferrer = OjamaScoreInferrer()
    breakdown = inferrer.infer_per_step_breakdown(
        sim_result, fired_by=SIDE_1P, match_elapsed_sec=0.0,
    )
    prev = 0
    for step in breakdown:
        assert step["cumulative_score"] >= prev
        prev = step["cumulative_score"]


def test_per_step_breakdown_empty_chain() -> None:
    """連鎖 0 の盤面 → 空リスト。"""
    grid = _empty_grid()
    board = Board.from_list(grid)
    sim_result = ChainSimulator().simulate(board)
    inferrer = OjamaScoreInferrer()
    breakdown = inferrer.infer_per_step_breakdown(
        sim_result, fired_by=SIDE_1P, match_elapsed_sec=0.0,
    )
    assert breakdown == []


def test_per_step_breakdown_step_score_equals_n_times_m() -> None:
    """step_score = n_display × m_display を維持。"""
    grid = _empty_grid()
    for r in range(9, 13):
        grid[r][0] = COLOR_RED
    board = Board.from_list(grid)
    sim_result = ChainSimulator().simulate(board)
    inferrer = OjamaScoreInferrer()
    breakdown = inferrer.infer_per_step_breakdown(
        sim_result, fired_by=SIDE_1P, match_elapsed_sec=0.0,
    )
    for step in breakdown:
        assert step["step_score"] == step["n_display"] * step["m_display"]
