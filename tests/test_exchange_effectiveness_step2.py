"""#24 打ち合い計測器 Step2 (scripts/measure_exchange_effectiveness.py) の回帰テスト。

軽量なダミー/手作り盤面のみを使用し、実動画・実npzの重い処理は行わない
(tests/test_exchange_dynamics_opp_coverage.py と同じ方針)。
"""
from __future__ import annotations

import math

import pytest

from src.board import Board, COLOR_RED, COLOR_BLUE, DEATH_COL, DEATH_ROW
from src.indicators_v2 import CHAIN_ANIM_PER_STEP_SEC
from scripts.measure_exchange_dynamics import OppCoverageStatus
from scripts.measure_exchange_effectiveness import (
    MAX_SUPPORTED_K_HANDS,
    ExchangeAdvantageLabel,
    classify_exchange_advantage,
    estimate_available_hands,
    estimate_landing_delay_sec,
    is_effective_saisoku,
    judge_exchange_effectiveness,
)


# ============================
# 着弾遅延 -> 手数見積もり
# ============================


def test_landing_delay_delegates_to_indicators_v2_formula() -> None:
    """2026-08-01 Step0: 物差し一本化後は estimate_chain_anim_duration_sec

    (CHAIN_ANIM_PER_STEP_SEC=0.4秒/連鎖) と完全一致する
    (詳細な境界値テーブルは tests/test_landing_delay_unification.py 参照)。
    """
    for n in (1, 4, 8, 13):
        assert estimate_landing_delay_sec(n) == pytest.approx(CHAIN_ANIM_PER_STEP_SEC * n)


def test_landing_delay_non_negative_for_zero_or_negative_chain() -> None:
    """連鎖数0以下は0秒にクランプする (負の遅延は非現実的なため)。"""
    assert estimate_landing_delay_sec(0) >= 0.0
    assert estimate_landing_delay_sec(-5) >= 0.0


def test_available_hands_clamped_to_max_supported() -> None:
    """見積もり手数は MAX_SUPPORTED_K_HANDS (=4, counter_reach_probability の

    対応上限) にクランプされる。

    2026-08-01 Step0 訂正: 旧2アンカー点モデルでは1連鎖でも約13手相当に
    達し常に上限クランプだったが、物差し一本化 (CHAIN_ANIM_PER_STEP_SEC=
    0.4) + floor(遅延/1手時間)+1 修正後は連鎖数が小さいうちは
    クランプされず (例: 1連鎖=1手)、8連鎖・13連鎖では依然クランプされる
    (詳細は tests/test_landing_delay_unification.py の境界値テーブル)。
    """
    hands_1 = estimate_available_hands(1)
    hands_8 = estimate_available_hands(8)
    hands_13 = estimate_available_hands(13)
    assert hands_1 < MAX_SUPPORTED_K_HANDS
    assert hands_8 == MAX_SUPPORTED_K_HANDS
    assert hands_13 == MAX_SUPPORTED_K_HANDS


def test_available_hands_never_below_one() -> None:
    """2026-08-01 Step0: +1 修正 (受け側の着地1手分) により、

    連鎖数が0・負でも見積もり手数は必ず1以上になる
    (0手=応手する暇がない、という旧仕様は撤廃)。
    """
    assert estimate_available_hands(-3) >= 1
    assert estimate_available_hands(0) >= 1


# ============================
# 有効性判定 (<=50%)
# ============================


def test_is_effective_saisoku_boundary() -> None:
    """閾値ちょうど (0.5) は有効側 (<=)。"""
    from src.indicators_v2 import COUNTER_REACH_EFFECTIVE_THRESHOLD_PROB
    assert COUNTER_REACH_EFFECTIVE_THRESHOLD_PROB == pytest.approx(0.5)
    assert is_effective_saisoku(0.5) is True
    assert is_effective_saisoku(0.5000001) is False
    assert is_effective_saisoku(0.0) is True
    assert is_effective_saisoku(1.0) is False


def test_is_effective_saisoku_rejects_nan() -> None:
    """NaN (判定不能) を渡すと誤って「有効」扱いせず例外を送出する。"""
    with pytest.raises(ValueError):
        is_effective_saisoku(float("nan"))


# ============================
# 有利/やや不利/不利 3値判定
# ============================


def test_classify_advantage_when_both_low() -> None:
    """小さい返りにすら届きにくい (両方<=50%) なら有利。"""
    label = classify_exchange_advantage(prob_return_one_dan=0.2, prob_return_two_dan=0.1)
    assert label == ExchangeAdvantageLabel.ADVANTAGE


def test_classify_advantage_when_both_high() -> None:
    """大きい返りにまで届きやすい (両方>50%) なら不利。"""
    label = classify_exchange_advantage(prob_return_one_dan=0.9, prob_return_two_dan=0.7)
    assert label == ExchangeAdvantageLabel.DISADVANTAGE


def test_classify_advantage_when_mixed() -> None:
    """小さい返りには届くが大きい返りには届きにくい (中間) ならやや不利。"""
    label = classify_exchange_advantage(prob_return_one_dan=0.7, prob_return_two_dan=0.3)
    assert label == ExchangeAdvantageLabel.SLIGHT_DISADVANTAGE


def test_classify_advantage_boundary_consistency_with_effectiveness() -> None:
    """prob_return_one_dan<=0.5 の有利判定は is_effective_saisoku と整合する。"""
    for p1 in (0.0, 0.3, 0.5, 0.51, 0.9):
        label = classify_exchange_advantage(prob_return_one_dan=p1, prob_return_two_dan=0.0)
        effective = is_effective_saisoku(p1)
        if effective:
            assert label == ExchangeAdvantageLabel.ADVANTAGE
        else:
            assert label != ExchangeAdvantageLabel.ADVANTAGE


# ============================
# OppCoverageStatus 接続 (最重要: OPP_CHAININGは確率0扱い)
# ============================


def test_opp_chaining_forces_probability_zero_without_computation() -> None:
    """相手が OPP_CHAINING (連鎖中=応手不能) の場合、opp_board=None でも

    counter_reach_probability を呼ばずに確率0・有効・有利判定になる
    (2026-07-29の重要な発見の反映)。
    """
    judgement = judge_exchange_effectiveness(
        opp_board=None, coverage_status=OppCoverageStatus.OPP_CHAINING, chain_count=5,
    )
    assert judgement.reach_probability == 0.0
    assert judgement.is_effective is True
    assert judgement.advantage_label == ExchangeAdvantageLabel.ADVANTAGE
    assert judgement.coverage_status == OppCoverageStatus.OPP_CHAINING


@pytest.mark.parametrize("status", [
    OppCoverageStatus.UNOBSERVED, OppCoverageStatus.MATCH_END, OppCoverageStatus.UNKNOWN,
])
def test_unobservable_status_without_board_returns_nan_not_zero(status) -> None:
    """観測不能 (OPP_CHAINING以外) で盤面が無い場合は NaN (判定不能) を返し、

    0 (=有効) に丸めて誤認させない。
    """
    judgement = judge_exchange_effectiveness(
        opp_board=None, coverage_status=status, chain_count=5,
    )
    assert math.isnan(judgement.reach_probability)
    assert judgement.is_effective is None
    assert judgement.advantage_label is None
    assert judgement.coverage_status == status


def test_observed_status_with_dead_opponent_board_returns_nan() -> None:
    """OBSERVED でも相手盤面が窒息なら (応手不能の別形態) NaN 扱い。

    ⚠️ 設計判断: 窒息は「応手不能」という意味では OPP_CHAINING と同じ
    帰結だが、意味論上は別 (連鎖中でなく単に負けている) ため、本関数では
    「計算不能」として NaN を返すに留める (0固定にはしない、呼び出し側が
    別途窒息を検知して扱うべき、と割り切った設計)。
    """
    dead_board = Board()
    dead_board.set(DEATH_ROW, DEATH_COL, COLOR_RED)
    judgement = judge_exchange_effectiveness(
        opp_board=dead_board, coverage_status=OppCoverageStatus.OBSERVED, chain_count=5,
    )
    assert math.isnan(judgement.reach_probability)


def test_observed_status_with_valid_board_computes_probability() -> None:
    """OBSERVED かつ有効な盤面なら実際に確率計算が行われる (NaNにならない)。"""
    board = Board()
    board.set(12, 0, COLOR_RED)
    board.set(12, 1, COLOR_BLUE)
    judgement = judge_exchange_effectiveness(
        opp_board=board, coverage_status=OppCoverageStatus.OBSERVED, chain_count=3,
    )
    assert not math.isnan(judgement.reach_probability)
    assert isinstance(judgement.is_effective, bool)
    assert isinstance(judgement.advantage_label, ExchangeAdvantageLabel)


def test_fast_and_precise_mode_both_runnable() -> None:
    """mode="fast" / "precise" のどちらでも例外なく動作する (二層設計の疎通確認)。"""
    board = Board()
    board.set(12, 0, COLOR_RED)
    board.set(12, 1, COLOR_BLUE)
    for mode in ("precise", "fast"):
        judgement = judge_exchange_effectiveness(
            opp_board=board, coverage_status=OppCoverageStatus.OBSERVED,
            chain_count=3, mode=mode,
        )
        assert not math.isnan(judgement.reach_probability)
