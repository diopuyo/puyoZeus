"""時間予算内で K を伸ばす応手判定のテスト (2026-08-09)。

user 指示「性能に余裕がある限り K を増やし、モンテカルロ法で優位な数が
保証できる形で実装」に対する担保。

検証の要点:
- **信頼区間の計算が正しいこと** (Wilson 区間。 p=0/1 で幅 0 にならない)
- **必要サンプル数が統計的に妥当なこと**
- **予算内で K が伸びること**、 予算が尽きたら正直に truncated を立てること
- **精度が足りない K を採用しないこと** (浅くても信頼できる方を返す)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.counter_reach_adaptive import (
    DEFAULT_TARGET_HALF_WIDTH,
    EXACT_K_MAX,
    Z_95,
    estimate_with_budget,
    required_samples,
    wilson_half_width,
)


def _board(fill_rows: int = 6, seed: int = 0) -> Board:
    rng = np.random.RandomState(seed)
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - fill_rows, BOARD_ROWS):
        for c in range(BOARD_COLS):
            g[r][c] = int(rng.choice([1, 2, 3, 4]))
    return Board.from_list(g)


class TestWilsonHalfWidth:
    """信頼区間の計算を検証する (統計的保証の土台)。"""

    def test_zero_samples_is_full_width(self) -> None:
        """サンプル 0 なら幅は最大 (何も分かっていない)。"""
        assert wilson_half_width(0.0, 0) == 1.0

    def test_p_zero_still_has_width(self) -> None:
        """p=0 でも幅が 0 にならないこと。

        正規近似 (sqrt(p(1-p)/n)) は p=0 で 0 になり、「サンプルが足りない
        のに精度十分」と誤判定する。 Wilson 区間はこれを避けられる。
        """
        assert wilson_half_width(0.0, 10) > 0.0
        assert wilson_half_width(1.0, 10) > 0.0

    def test_width_shrinks_with_samples(self) -> None:
        """サンプルが増えるほど幅が狭くなること。"""
        w10 = wilson_half_width(0.5, 10)
        w100 = wilson_half_width(0.5, 100)
        w1000 = wilson_half_width(0.5, 1000)
        assert w10 > w100 > w1000

    def test_width_near_normal_approx_for_large_n(self) -> None:
        """n が大きければ正規近似に近づくこと (実装の妥当性確認)。"""
        n, p = 10000, 0.5
        normal = Z_95 * np.sqrt(p * (1 - p) / n)
        assert abs(wilson_half_width(p, n) - normal) < 0.002


class TestRequiredSamples:
    """必要サンプル数が統計的に妥当か。"""

    def test_five_percent_needs_about_384(self) -> None:
        """±5% には約 384 サンプル (教科書的な値)。"""
        assert 380 <= required_samples(0.05) <= 390

    def test_tighter_target_needs_more(self) -> None:
        """目標を厳しくするほど必要数が増えること。"""
        assert required_samples(0.01) > required_samples(0.05)

    def test_rejects_nonpositive(self) -> None:
        """0 以下の目標は誤り。"""
        with pytest.raises(ValueError):
            required_samples(0.0)


class TestEstimateWithBudget:
    """時間予算内で K を伸ばす挙動を検証する。"""

    def test_returns_exact_for_small_k(self) -> None:
        """K<=2 は全列挙なので誤差ゼロと報告されること。"""
        r = estimate_with_budget(_board(), 12.0, budget_sec=5.0, k_hard_max=2)
        assert r.achieved_k <= EXACT_K_MAX
        assert r.exact is True
        assert r.half_width == 0.0

    def test_zero_budget_reports_truncated(self) -> None:
        """予算ゼロなら何も保証できず、 truncated を立てること。"""
        r = estimate_with_budget(_board(), 12.0, budget_sec=0.0)
        assert r.achieved_k == 0
        assert r.truncated_by_budget is True

    def test_larger_budget_reaches_deeper(self) -> None:
        """予算を増やすと到達 K が浅くならないこと。"""
        small = estimate_with_budget(_board(), 12.0, budget_sec=0.3)
        large = estimate_with_budget(_board(), 12.0, budget_sec=3.0)
        assert large.achieved_k >= small.achieved_k

    def test_probability_in_range(self) -> None:
        """確率が 0〜1 に収まること。"""
        r = estimate_with_budget(_board(), 12.0, budget_sec=2.0)
        assert 0.0 <= r.probability <= 1.0

    def test_dead_board_returns_zero(self) -> None:
        """窒息盤面は応手不能なので確率 0。"""
        g = [[1] * BOARD_COLS for _ in range(BOARD_ROWS)]
        r = estimate_with_budget(Board.from_list(g), 12.0, budget_sec=1.0)
        assert r.probability == 0.0

    def test_reports_elapsed_and_budget(self) -> None:
        """使った時間と与えられた予算を必ず返すこと (黙って超過しない)。"""
        r = estimate_with_budget(_board(), 12.0, budget_sec=1.0)
        assert r.requested_budget_sec == 1.0
        assert r.elapsed_sec >= 0.0

    def test_precision_target_is_respected(self) -> None:
        """MC の K を採用した場合、 半幅が目標以内であること。

        これが「優位な数が保証できる形」の中核。 目標に届かない K は
        採用しない設計なので、 採用された結果は必ず目標以内になる。
        """
        r = estimate_with_budget(
            _board(), 12.0, budget_sec=10.0,
            target_half_width=DEFAULT_TARGET_HALF_WIDTH,
        )
        if not r.exact and r.achieved_k > EXACT_K_MAX:
            assert r.half_width <= DEFAULT_TARGET_HALF_WIDTH
