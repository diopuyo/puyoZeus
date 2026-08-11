"""着弾遅延の物差し一本化 (2026-08-01 Step0) の回帰テスト。

背景: 着弾遅延(連鎖アニメ所要秒数)の推定に3系統の定数が併存していた
(TIME_PER_CHAIN_SEC=0.30 の過小評価版 / measure_exchange_effectiveness.py の
2アンカー点線形補間 n=2 の暫定値 / CHAIN_ANIM_PER_STEP_SEC=0.4 の23動画418
イベント実測ベース、検証件数最多)。src.indicators_v2.
estimate_chain_anim_duration_sec (CHAIN_ANIM_PER_STEP_SEC ベース) に一本化し、
scripts/measure_exchange_effectiveness.py と scripts/label_exchange_outcome.py
の双方がこの関数経由で計算するようになったことを境界値テーブルで確認する。
"""
from __future__ import annotations

import pytest

from src.indicators_v2 import (
    CHAIN_ANIM_DURATION_BIAS_SEC_2026_08_11,
    CHAIN_ANIM_DURATION_MAX_CHAIN_COUNT_2026_08_11,
    CHAIN_ANIM_PER_STEP_SEC,
    SEC_PER_HAND,
    estimate_chain_anim_duration_sec,
)
from scripts.measure_exchange_effectiveness import (
    MAX_SUPPORTED_K_HANDS,
    estimate_available_hands,
    estimate_landing_delay_sec,
    judge_exchange_effectiveness,
)
from scripts.measure_exchange_dynamics import OppCoverageStatus
from src.board import Board, COLOR_RED, COLOR_BLUE


# ============================
# estimate_chain_anim_duration_sec 境界値テーブル
# ============================

@pytest.mark.parametrize("chain_count,expected_sec", [
    (0, 0.0),
    (1, CHAIN_ANIM_PER_STEP_SEC * 1),
    (4, CHAIN_ANIM_PER_STEP_SEC * 4),
    (8, CHAIN_ANIM_PER_STEP_SEC * 8),
    (13, CHAIN_ANIM_PER_STEP_SEC * 13),
    (20, CHAIN_ANIM_PER_STEP_SEC * 20),
])
def test_estimate_chain_anim_duration_sec_table(chain_count: int, expected_sec: float) -> None:
    """CHAIN_ANIM_PER_STEP_SEC * chain_count と厳密一致する (負値は0クランプ済み対象外)。"""
    assert estimate_chain_anim_duration_sec(chain_count) == pytest.approx(expected_sec)


def test_estimate_chain_anim_duration_sec_negative_clamped_to_zero() -> None:
    """連鎖数が負の場合は0.0にクランプする。"""
    assert estimate_chain_anim_duration_sec(-5) == 0.0


# ============================
# 較正 Phase 1 (2026-08-11、chain_end_sec_gap 全域再測定) 回帰テスト
# ============================
# calibration="v2026_08_11" は data/verify/chain_end_sec_gap_2026-08-09.jsonl
# (11動画・7,324イベント、mechanism=formula・c109除く10動画) の固定バイアス
# 較正 (CHAIN_ANIM_DURATION_BIAS_SEC_2026_08_11=0.17秒、event単位 LOVO で
# 連鎖数依存の傾き変更より優れることを確認済み)。 詳細は
# estimate_chain_anim_duration_sec 直前のコメントブロック参照。

@pytest.mark.parametrize("chain_count", [0, 1, 4, 8, 13, 20])
def test_estimate_chain_anim_duration_sec_default_unchanged(
    chain_count: int,
) -> None:
    """calibration 省略時は従来 (calibration="legacy") と bit-identical。"""
    assert (
        estimate_chain_anim_duration_sec(chain_count)
        == estimate_chain_anim_duration_sec(chain_count, calibration="legacy")
    )


def test_estimate_chain_anim_duration_sec_v2026_08_11_monotonic() -> None:
    """calibration="v2026_08_11" は連鎖数が増えるほど単調非減少 (クランプ域含む)。"""
    values = [
        estimate_chain_anim_duration_sec(n, calibration="v2026_08_11")
        for n in range(0, 25)
    ]
    for prev, cur in zip(values, values[1:]):
        assert cur >= prev


def test_estimate_chain_anim_duration_sec_v2026_08_11_clamped_beyond_max() -> None:
    """実測連鎖数上限 (=15) を超える chain_count はクランプされ、上限値と一致する。"""
    at_max = estimate_chain_anim_duration_sec(
        CHAIN_ANIM_DURATION_MAX_CHAIN_COUNT_2026_08_11, calibration="v2026_08_11",
    )
    beyond_max = estimate_chain_anim_duration_sec(30, calibration="v2026_08_11")
    assert beyond_max == pytest.approx(at_max)


def test_estimate_chain_anim_duration_sec_v2026_08_11_formula() -> None:
    """クランプ域内では CHAIN_ANIM_PER_STEP_SEC*n + バイアス と厳密一致する。"""
    for n in (1, 4, 8, 15):
        expected = CHAIN_ANIM_PER_STEP_SEC * n + CHAIN_ANIM_DURATION_BIAS_SEC_2026_08_11
        assert (
            estimate_chain_anim_duration_sec(n, calibration="v2026_08_11")
            == pytest.approx(expected)
        )


def test_estimate_chain_anim_duration_sec_v2026_08_11_zero_or_negative() -> None:
    """連鎖数が0以下ならバイアスも足さず0.0を返す (legacy と同じクランプ規約)。"""
    assert estimate_chain_anim_duration_sec(0, calibration="v2026_08_11") == 0.0
    assert estimate_chain_anim_duration_sec(-3, calibration="v2026_08_11") == 0.0


def test_estimate_chain_anim_duration_sec_unknown_calibration_raises() -> None:
    """未知の calibration 指定は ValueError で明示的に失敗する。"""
    with pytest.raises(ValueError):
        estimate_chain_anim_duration_sec(1, calibration="not_a_real_calibration")


# ============================
# estimate_available_hands 境界値テーブル (floor(遅延/1手時間)+1、K<=4クランプ)
# ============================

@pytest.mark.parametrize("chain_count,expected_hands", [
    (0, 1),   # floor(0/0.733)+1 = 1 (受け側の着地1手分は必ず数える)
    (1, 1),   # floor(0.4/0.733)+1 = 0+1 = 1
    (4, 3),   # floor(1.6/0.733)+1 = 2+1 = 3
    (8, MAX_SUPPORTED_K_HANDS),   # floor(3.2/0.733)+1=5 -> クランプ4
    (13, MAX_SUPPORTED_K_HANDS),  # floor(5.2/0.733)+1=8 -> クランプ4
    (20, MAX_SUPPORTED_K_HANDS),  # floor(8.0/0.733)+1=11 -> クランプ4
])
def test_estimate_available_hands_table(chain_count: int, expected_hands: int) -> None:
    """floor(estimate_landing_delay_sec(n) / SEC_PER_HAND) + 1 を

    MAX_SUPPORTED_K_HANDS でクランプした値と一致する
    (user伝授: おじゃまは受け側のツモ着地時に降るため +1 が正しい)。
    """
    delay_sec = estimate_landing_delay_sec(chain_count)
    manual = min(MAX_SUPPORTED_K_HANDS, int(delay_sec // SEC_PER_HAND) + 1)
    assert estimate_available_hands(chain_count) == manual
    assert estimate_available_hands(chain_count) == expected_hands


def test_estimate_available_hands_never_zero_or_negative() -> None:
    """+1 修正により、どんな連鎖数でも見積もり手数は1以上になる。"""
    for n in (-10, -1, 0, 1, 2, 3):
        assert estimate_available_hands(n) >= 1


# ============================
# judge_exchange_effectiveness: k_hands<=0 分岐は到達不能 (assert で明示)
# ============================

def test_k_hands_never_reaches_zero_branch() -> None:
    """+1 修正後、judge_exchange_effectiveness 内の k_hands>=1 assert が

    どんな連鎖数 (0以下含む) でも例外を出さずに正常終了することを確認する
    (旧 k_hands<=0 分岐は到達不能になったため assert に置換済み、
    scripts/measure_exchange_effectiveness.py:_judge_exchange_effectiveness 参照)。
    """
    board = Board()
    board.set(12, 0, COLOR_RED)
    board.set(12, 1, COLOR_BLUE)
    for chain_count in (-5, 0, 1, 4, 8, 20):
        judgement = judge_exchange_effectiveness(
            opp_board=board, coverage_status=OppCoverageStatus.OBSERVED,
            chain_count=chain_count,
        )
        # assert が発火せず正常に確率が計算されること (NaNでない) を確認。
        assert judgement.reach_probability == judgement.reach_probability  # not NaN
