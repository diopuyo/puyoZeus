"""
PhaseAwareScorer のテスト。

- phase 分類の境界条件
- start / mid / end フェーズで重みが切り替わること
- interpolate=True で過渡域の重みが滑らかに混ざること
- score() 出力が ScoreResult として整合すること
"""

from __future__ import annotations

import pytest

from src.indicators import (
    ALL_INDICATOR_NAMES,
    INDICATOR_DEATH_RISK,
    INDICATOR_MAIN_CHAIN,
    IndicatorResult,
    IndicatorSet,
)
from src.scorer import (
    DEFAULT_WEIGHTS,
    LEARNED_WEIGHTS_END,
    LEARNED_WEIGHTS_GLOBAL,
    LEARNED_WEIGHTS_MIDPOINT,
    LEARNED_WEIGHTS_START,
    LEARNED_WEIGHTS_V3_GLOBAL,
    OPTIMAL_PHASE_WEIGHTS,
    PHASE_END,
    PHASE_MID,
    PHASE_START,
    WEIGHT_MODE_LEARNED,
    WEIGHT_MODE_OPTIMAL,
    PhaseAwareScorer,
    ScoreResult,
    classify_phase,
)


# ============================
# テスト用ヘルパー
# ============================


def make_indicator_set(scores: dict[str, float]) -> IndicatorSet:
    """指定スコアで IndicatorSet を生成する (欠けている指標は 0)。"""
    results: dict[str, IndicatorResult] = {}
    for name in ALL_INDICATOR_NAMES:
        s = scores.get(name, 0.0)
        results[name] = IndicatorResult(name=name, score=s, raw_value=s)
    return IndicatorSet(results=results)


def ones_indicator() -> IndicatorSet:
    return make_indicator_set({n: 1.0 for n in ALL_INDICATOR_NAMES})


def zeros_indicator() -> IndicatorSet:
    return make_indicator_set({})


# ============================
# classify_phase
# ============================


class TestClassifyPhase:
    def test_start_phase_at_zero(self) -> None:
        assert classify_phase(0.0, 60.0) == PHASE_START

    def test_start_phase_within_30s(self) -> None:
        assert classify_phase(15.0, 60.0) == PHASE_START
        assert classify_phase(30.0, 60.0) == PHASE_START

    def test_mid_phase_just_after_start(self) -> None:
        assert classify_phase(30.5, 80.0) == PHASE_MID
        assert classify_phase(45.0, 80.0) == PHASE_MID

    def test_end_phase_within_15s_of_end(self) -> None:
        # duration=60, elapsed=46 → remaining=14 → end
        assert classify_phase(46.0, 60.0) == PHASE_END
        assert classify_phase(60.0, 60.0) == PHASE_END

    def test_invalid_duration_returns_mid(self) -> None:
        assert classify_phase(10.0, 0.0) == PHASE_MID
        assert classify_phase(10.0, -5.0) == PHASE_MID

    def test_short_match_under_30s_treated_as_start(self) -> None:
        # duration=20, elapsed=5 → start (start_boundary=30 優先)
        assert classify_phase(5.0, 20.0) == PHASE_START


# ============================
# PhaseAwareScorer 基本
# ============================


class TestPhaseAwareScorerBasic:
    def test_score_returns_scoreresult(self) -> None:
        scorer = PhaseAwareScorer(interpolate=False)
        ind1 = ones_indicator()
        ind2 = zeros_indicator()
        result = scorer.score(ind1, ind2, elapsed_sec=10.0,
                              match_duration_sec=60.0)
        assert isinstance(result, ScoreResult)
        # 1P が完全に勝っているのでスコアは正
        assert result.total_score > 0

    def test_no_interpolate_uses_phase_weights(self) -> None:
        scorer = PhaseAwareScorer(interpolate=False)
        # start phase
        w_start = scorer.resolve_weights(5.0, 60.0)
        # LEARNED_WEIGHTS_START の値が反映されているはず
        assert w_start[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_START[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_no_interpolate_end_phase_weights(self) -> None:
        scorer = PhaseAwareScorer(interpolate=False)
        # duration=60, elapsed=58 → remaining=2 → end
        w_end = scorer.resolve_weights(58.0, 60.0)
        assert w_end[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_END[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_no_interpolate_mid_phase_weights(self) -> None:
        scorer = PhaseAwareScorer(interpolate=False)
        # duration=120, elapsed=60 → mid (60 > 30 かつ remaining=60 > 15)
        w_mid = scorer.resolve_weights(60.0, 120.0)
        assert w_mid[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_MIDPOINT[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_current_phase_returns_string(self) -> None:
        scorer = PhaseAwareScorer()
        assert scorer.current_phase(0.0, 60.0) == PHASE_START
        assert scorer.current_phase(40.0, 80.0) == PHASE_MID
        assert scorer.current_phase(70.0, 80.0) == PHASE_END


# ============================
# 補間モード
# ============================


class TestPhaseAwareScorerInterpolate:
    def test_interpolate_inside_start_uses_start_weights(self) -> None:
        scorer = PhaseAwareScorer(interpolate=True)
        # 過渡域 (25-35s) の外側 → 完全に start
        w = scorer.resolve_weights(10.0, 120.0)
        assert w[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_START[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_interpolate_at_phase_boundary_blends(self) -> None:
        scorer = PhaseAwareScorer(interpolate=True)
        # elapsed=30 (PHASE_START_BOUNDARY_SEC) で blend half
        w_blend = scorer.resolve_weights(30.0, 120.0)
        expected_main = (
            LEARNED_WEIGHTS_START[INDICATOR_MAIN_CHAIN] * 0.5
            + LEARNED_WEIGHTS_MIDPOINT[INDICATOR_MAIN_CHAIN] * 0.5
        )
        assert w_blend[INDICATOR_MAIN_CHAIN] == pytest.approx(
            expected_main, abs=1e-6,
        )

    def test_interpolate_inside_mid_uses_mid_weights(self) -> None:
        scorer = PhaseAwareScorer(interpolate=True)
        # 過渡域の外側で中盤
        w = scorer.resolve_weights(60.0, 120.0)
        assert w[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_MIDPOINT[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_interpolate_at_end_boundary_blends(self) -> None:
        scorer = PhaseAwareScorer(interpolate=True)
        # duration=120, end_center = 120 - 15 = 105
        # elapsed=105 で blend half mid↔end
        w_blend = scorer.resolve_weights(105.0, 120.0)
        expected_main = (
            LEARNED_WEIGHTS_MIDPOINT[INDICATOR_MAIN_CHAIN] * 0.5
            + LEARNED_WEIGHTS_END[INDICATOR_MAIN_CHAIN] * 0.5
        )
        assert w_blend[INDICATOR_MAIN_CHAIN] == pytest.approx(
            expected_main, abs=1e-6,
        )

    def test_interpolate_end_uses_end_weights(self) -> None:
        scorer = PhaseAwareScorer(interpolate=True)
        # 終盤完全に終わり側
        w = scorer.resolve_weights(115.0, 120.0)
        assert w[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_END[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_interpolate_default_weights_filled_for_missing_keys(self) -> None:
        scorer = PhaseAwareScorer(interpolate=True)
        w = scorer.resolve_weights(10.0, 120.0)
        # DEFAULT_WEIGHTS の全キーが含まれていること
        for key in DEFAULT_WEIGHTS:
            assert key in w


# ============================
# weights_overrides
# ============================


class TestWeightsOverrides:
    def test_weights_overrides_applied(self) -> None:
        override = {PHASE_START: {INDICATOR_MAIN_CHAIN: 99.0}}
        scorer = PhaseAwareScorer(
            interpolate=False, weights_overrides=override,
        )
        w = scorer.resolve_weights(5.0, 60.0)
        assert w[INDICATOR_MAIN_CHAIN] == pytest.approx(99.0)

    def test_weights_overrides_only_affect_target_phase(self) -> None:
        override = {PHASE_START: {INDICATOR_MAIN_CHAIN: 99.0}}
        scorer = PhaseAwareScorer(
            interpolate=False, weights_overrides=override,
        )
        # mid phase は影響を受けない
        w_mid = scorer.resolve_weights(60.0, 120.0)
        assert w_mid[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_MIDPOINT[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )


# ============================
# サニティチェック
# ============================


class TestPhaseAwareScorerSanity:
    def test_score_bounded_by_range(self) -> None:
        scorer = PhaseAwareScorer(interpolate=False)
        ind1 = ones_indicator()
        ind2 = zeros_indicator()
        for elapsed in (0.0, 15.0, 30.0, 60.0, 90.0):
            r = scorer.score(ind1, ind2, elapsed, 100.0)
            assert -100.0 <= r.total_score <= 100.0

    def test_score_zero_when_indicators_equal(self) -> None:
        scorer = PhaseAwareScorer(interpolate=False)
        ind = ones_indicator()
        r = scorer.score(ind, ind, 30.0, 60.0)
        assert r.total_score == pytest.approx(0.0, abs=1e-6)

    def test_interpolate_property(self) -> None:
        s1 = PhaseAwareScorer(interpolate=True)
        s2 = PhaseAwareScorer(interpolate=False)
        assert s1.interpolate is True
        assert s2.interpolate is False

    def test_phase_boundaries_constant(self) -> None:
        # クラス定数の構造確認
        assert PhaseAwareScorer.PHASE_BOUNDARIES_SEC == (30.0, -15.0)


# ============================
# weight_mode = "optimal"
# ============================


class TestWeightModeOptimal:
    """OPTIMAL_PHASE_WEIGHTS 切替モードの検証。

    実証データ (1390 サンプル) で確定した最良の組合せ:
    start=DEFAULT, mid=LEARNED_V3_GLOBAL, end=LEARNED_GLOBAL を使用する。
    """

    def test_optimal_start_uses_default_weights(self) -> None:
        scorer = PhaseAwareScorer(
            interpolate=False, weight_mode=WEIGHT_MODE_OPTIMAL,
        )
        w = scorer.resolve_weights(5.0, 60.0)
        # start phase の主指標は DEFAULT_WEIGHTS と一致するはず
        assert w[INDICATOR_MAIN_CHAIN] == pytest.approx(
            DEFAULT_WEIGHTS[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_optimal_mid_uses_v3_global_weights(self) -> None:
        scorer = PhaseAwareScorer(
            interpolate=False, weight_mode=WEIGHT_MODE_OPTIMAL,
        )
        # duration=120, elapsed=60 → mid
        w = scorer.resolve_weights(60.0, 120.0)
        assert w[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_V3_GLOBAL[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_optimal_end_uses_learned_global_weights(self) -> None:
        scorer = PhaseAwareScorer(
            interpolate=False, weight_mode=WEIGHT_MODE_OPTIMAL,
        )
        # duration=60, elapsed=58 → end
        w = scorer.resolve_weights(58.0, 60.0)
        assert w[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_GLOBAL[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_optimal_registry_keys_complete(self) -> None:
        # OPTIMAL_PHASE_WEIGHTS は 3 phase 全てを定義しているはず
        assert set(OPTIMAL_PHASE_WEIGHTS.keys()) == {
            PHASE_START, PHASE_MID, PHASE_END,
        }

    def test_default_mode_is_learned(self) -> None:
        scorer = PhaseAwareScorer()
        assert scorer.weight_mode == WEIGHT_MODE_LEARNED
        # 既存挙動互換: start phase の重みは LEARNED_WEIGHTS_START
        w = scorer.resolve_weights(5.0, 60.0)
        assert w[INDICATOR_MAIN_CHAIN] == pytest.approx(
            LEARNED_WEIGHTS_START[INDICATOR_MAIN_CHAIN], abs=1e-6,
        )

    def test_unknown_weight_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="未知の weight_mode"):
            PhaseAwareScorer(weight_mode="unknown_mode")

    def test_optimal_score_runs_end_to_end(self) -> None:
        scorer = PhaseAwareScorer(
            interpolate=True, weight_mode=WEIGHT_MODE_OPTIMAL,
        )
        ind1 = ones_indicator()
        ind2 = zeros_indicator()
        # 全 phase で出力が正の値 (1P 完全勝ち)
        for elapsed in (5.0, 30.0, 60.0, 100.0, 115.0):
            r = scorer.score(ind1, ind2, elapsed, 120.0)
            assert isinstance(r, ScoreResult)
            assert -100.0 <= r.total_score <= 100.0

    def test_optimal_interpolate_blends_default_and_v3(self) -> None:
        scorer = PhaseAwareScorer(
            interpolate=True, weight_mode=WEIGHT_MODE_OPTIMAL,
        )
        # elapsed=30 (start↔mid 境界中央) で半々ブレンド
        w_blend = scorer.resolve_weights(30.0, 120.0)
        expected_main = (
            DEFAULT_WEIGHTS[INDICATOR_MAIN_CHAIN] * 0.5
            + LEARNED_WEIGHTS_V3_GLOBAL[INDICATOR_MAIN_CHAIN] * 0.5
        )
        assert w_blend[INDICATOR_MAIN_CHAIN] == pytest.approx(
            expected_main, abs=1e-6,
        )
