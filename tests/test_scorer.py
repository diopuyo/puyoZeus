"""
scorer.py のテスト

総合スコア算出・重み制御・有利判定を検証する。
"""

from __future__ import annotations

import pytest

from src.old.indicators import (
    ALL_INDICATOR_NAMES,
    INDICATOR_DEATH_RISK,
    INDICATOR_MAIN_CHAIN,
    IndicatorResult,
    IndicatorSet,
)
from src.old.scorer import (
    ADVANTAGE_EVEN,
    DEFAULT_WEIGHTS,
    EVEN_THRESHOLD,
    PLAYER_1P,
    PLAYER_2P,
    SCORE_RANGE_MAX,
    SCORE_RANGE_MIN,
    ScoreResult,
    Scorer,
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


def zeros() -> IndicatorSet:
    return make_indicator_set({})


def ones() -> IndicatorSet:
    return make_indicator_set({n: 1.0 for n in ALL_INDICATOR_NAMES})


# ============================
# ScoreResult
# ============================


class TestScoreResult:
    def test_basic_fields(self):
        r = ScoreResult(
            total_score=42.0,
            player1_raw=5.0,
            player2_raw=2.0,
            player1_breakdown={},
            player2_breakdown={},
        )
        assert r.total_score == 42.0
        assert r.player1_raw == 5.0

    def test_advantage_1p(self):
        r = ScoreResult(
            total_score=50.0, player1_raw=0, player2_raw=0,
            player1_breakdown={}, player2_breakdown={},
        )
        assert r.advantage_side() == PLAYER_1P

    def test_advantage_2p(self):
        r = ScoreResult(
            total_score=-30.0, player1_raw=0, player2_raw=0,
            player1_breakdown={}, player2_breakdown={},
        )
        assert r.advantage_side() == PLAYER_2P

    def test_advantage_even_at_threshold(self):
        r = ScoreResult(
            total_score=EVEN_THRESHOLD, player1_raw=0, player2_raw=0,
            player1_breakdown={}, player2_breakdown={},
        )
        assert r.advantage_side() == ADVANTAGE_EVEN

    def test_advantage_even_negative_threshold(self):
        r = ScoreResult(
            total_score=-EVEN_THRESHOLD + 0.1, player1_raw=0, player2_raw=0,
            player1_breakdown={}, player2_breakdown={},
        )
        assert r.advantage_side() == ADVANTAGE_EVEN

    def test_to_dict_contains_advantage(self):
        r = ScoreResult(
            total_score=60.0, player1_raw=7, player2_raw=3,
            player1_breakdown={"x": 1}, player2_breakdown={"x": 0},
            weights={"x": 1.0},
        )
        d = r.to_dict()
        assert d["total_score"] == 60.0
        assert d["advantage"] == PLAYER_1P
        assert d["weights"]["x"] == 1.0


# ============================
# Scorer - 基本挙動
# ============================


class TestScorerInit:
    def test_default_weights_used(self):
        s = Scorer()
        for name in ALL_INDICATOR_NAMES:
            assert name in s._weights

    def test_custom_weights_override(self):
        s = Scorer(weights={INDICATOR_MAIN_CHAIN: 999.0})
        assert s._weights[INDICATOR_MAIN_CHAIN] == 999.0
        # 他はデフォルトのまま
        assert s._weights[INDICATOR_DEATH_RISK] == DEFAULT_WEIGHTS[INDICATOR_DEATH_RISK]

    def test_unknown_weight_raises(self):
        with pytest.raises(ValueError, match="未知の指標名"):
            Scorer(weights={"unknown_indicator": 1.0})


class TestScorerEqualInputs:
    def test_equal_sets_yield_zero_score(self):
        s = Scorer()
        result = s.score(ones(), ones())
        assert result.total_score == 0.0
        assert result.advantage_side() == ADVANTAGE_EVEN

    def test_both_zero_yield_zero(self):
        s = Scorer()
        result = s.score(zeros(), zeros())
        assert result.total_score == 0.0


# ============================
# Scorer - 方向性
# ============================


class TestScorerDirection:
    def test_player1_all_ones_wins(self):
        s = Scorer()
        result = s.score(ones(), zeros())
        # 窒息リスクは負の重みなので、1P が1.0だと不利寄与
        # しかし他の指標がすべて正の重みで有利 → 全体では差が出る
        # 符号を確認: total_score の符号は weights の net に依存
        net = sum(DEFAULT_WEIGHTS.values())
        if net > 0:
            assert result.total_score > 0
        elif net < 0:
            assert result.total_score < 0

    def test_player2_all_ones_loses_for_1p(self):
        s = Scorer()
        r_1p = s.score(ones(), zeros())
        r_2p = s.score(zeros(), ones())
        # 対称性: 入れ替えると符号が反転する
        assert r_1p.total_score == pytest.approx(-r_2p.total_score)

    def test_main_chain_only_1p_advantage(self):
        """本線完成度のみ 1P が勝っている場合、スコアは正。"""
        s = Scorer()
        p1 = make_indicator_set({INDICATOR_MAIN_CHAIN: 1.0})
        p2 = make_indicator_set({})
        result = s.score(p1, p2)
        assert result.total_score > 0

    def test_death_risk_higher_is_worse(self):
        """窒息リスクが高い 1P はスコアが下がる (負の重み)。"""
        s = Scorer()
        p1 = make_indicator_set({INDICATOR_DEATH_RISK: 1.0})
        p2 = make_indicator_set({})
        result = s.score(p1, p2)
        # 1Pのほうが死亡リスク高い=不利 → score 負
        assert result.total_score < 0


# ============================
# Scorer - 範囲とクランプ
# ============================


class TestScorerRange:
    def test_score_always_in_range(self):
        s = Scorer()
        for p1, p2 in [(ones(), zeros()), (zeros(), ones()), (ones(), ones())]:
            r = s.score(p1, p2)
            assert SCORE_RANGE_MIN <= r.total_score <= SCORE_RANGE_MAX

    def test_clamped_at_extreme_weights(self):
        # 極端な重みでも範囲内に収まる
        s = Scorer(weights={INDICATOR_MAIN_CHAIN: 100.0})
        r = s.score(ones(), zeros())
        assert r.total_score <= SCORE_RANGE_MAX


# ============================
# Scorer - 内訳
# ============================


class TestScorerBreakdown:
    def test_breakdown_has_all_indicators(self):
        s = Scorer()
        r = s.score(ones(), zeros())
        assert set(r.player1_breakdown.keys()) == set(ALL_INDICATOR_NAMES)
        assert set(r.player2_breakdown.keys()) == set(ALL_INDICATOR_NAMES)

    def test_breakdown_sum_equals_raw(self):
        s = Scorer()
        r = s.score(ones(), zeros())
        assert sum(r.player1_breakdown.values()) == pytest.approx(r.player1_raw)
        assert sum(r.player2_breakdown.values()) == pytest.approx(r.player2_raw)

    def test_breakdown_value_is_score_times_weight(self):
        s = Scorer()
        p1 = make_indicator_set({INDICATOR_MAIN_CHAIN: 0.5})
        r = s.score(p1, zeros())
        expected = 0.5 * DEFAULT_WEIGHTS[INDICATOR_MAIN_CHAIN]
        assert r.player1_breakdown[INDICATOR_MAIN_CHAIN] == pytest.approx(expected)

    def test_weights_included_in_result(self):
        s = Scorer()
        r = s.score(zeros(), zeros())
        assert r.weights == DEFAULT_WEIGHTS


# ============================
# Scorer - エッジケース
# ============================


class TestScorerEdgeCases:
    def test_missing_indicator_treated_as_zero(self):
        s = Scorer()
        partial = IndicatorSet(results={})
        r = s.score(partial, partial)
        assert r.total_score == 0.0
        for name in ALL_INDICATOR_NAMES:
            assert r.player1_breakdown[name] == 0.0

    def test_zero_normalizer(self):
        # 全重みを 0 にすれば normalizer=0 → total_score=0
        zero_weights = {n: 0.0 for n in ALL_INDICATOR_NAMES}
        s = Scorer(weights=zero_weights)
        r = s.score(ones(), zeros())
        assert r.total_score == 0.0


# ============================
# 学習済み重みセット (LEARNED_*)
# ============================


class TestLearnedWeightSets:
    """scorer.py に追加された LEARNED_WEIGHTS_* の動作確認。"""

    def test_default_weight_set_matches_default_weights(self):
        """weight_set='DEFAULT' と weights 未指定が等価。"""
        from src.old.scorer import WEIGHT_SET_DEFAULT
        s_default = Scorer()
        s_named = Scorer(weight_set=WEIGHT_SET_DEFAULT)
        assert s_default._weights == s_named._weights

    def test_learned_global_weight_set_loads(self):
        """LEARNED_GLOBAL を指定して Scorer を作れる。"""
        from src.old.scorer import (
            LEARNED_WEIGHTS_GLOBAL,
            WEIGHT_SET_LEARNED_GLOBAL,
        )
        s = Scorer(weight_set=WEIGHT_SET_LEARNED_GLOBAL)
        for name, val in LEARNED_WEIGHTS_GLOBAL.items():
            assert s._weights[name] == val

    def test_learned_midpoint_weight_set_loads(self):
        from src.old.scorer import (
            LEARNED_WEIGHTS_MIDPOINT,
            WEIGHT_SET_LEARNED_MIDPOINT,
        )
        s = Scorer(weight_set=WEIGHT_SET_LEARNED_MIDPOINT)
        for name, val in LEARNED_WEIGHTS_MIDPOINT.items():
            assert s._weights[name] == val

    def test_unknown_weight_set_raises(self):
        with pytest.raises(ValueError):
            Scorer(weight_set="UNKNOWN_SET")

    def test_weights_argument_overrides_weight_set(self):
        """weights 引数を渡せば weight_set より優先される。"""
        from src.old.scorer import WEIGHT_SET_LEARNED_GLOBAL
        custom = {INDICATOR_MAIN_CHAIN: 99.0}
        s = Scorer(
            weights=custom, weight_set=WEIGHT_SET_LEARNED_GLOBAL,
        )
        assert s._weights[INDICATOR_MAIN_CHAIN] == 99.0

    def test_learned_global_score_runs(self):
        """LEARNED_GLOBAL でも score() がエラーなく動く。"""
        from src.old.scorer import WEIGHT_SET_LEARNED_GLOBAL
        s = Scorer(weight_set=WEIGHT_SET_LEARNED_GLOBAL)
        r = s.score(ones(), zeros())
        assert SCORE_RANGE_MIN <= r.total_score <= SCORE_RANGE_MAX

    def test_registry_contains_all_learned_sets(self):
        """WEIGHT_SET_REGISTRY に主要セットが登録されている。"""
        from src.old.scorer import WEIGHT_SET_REGISTRY
        assert "DEFAULT" in WEIGHT_SET_REGISTRY
        assert "LEARNED_GLOBAL" in WEIGHT_SET_REGISTRY
        assert "LEARNED_MIDPOINT" in WEIGHT_SET_REGISTRY
        assert "LEARNED_END" in WEIGHT_SET_REGISTRY
        assert "LEARNED_START" in WEIGHT_SET_REGISTRY
        assert "LEARNED_V3_GLOBAL" in WEIGHT_SET_REGISTRY
        assert "RECOMMENDED" in WEIGHT_SET_REGISTRY


class TestRecommendedWeights:
    """ablation 結果を反映した RECOMMENDED 重みセットの検証。"""

    def test_recommended_loads(self):
        """weight_set='RECOMMENDED' で Scorer を構築できる。"""
        from src.old.scorer import (
            LEARNED_WEIGHTS_RECOMMENDED,
            WEIGHT_SET_RECOMMENDED,
        )
        s = Scorer(weight_set=WEIGHT_SET_RECOMMENDED)
        for name, val in LEARNED_WEIGHTS_RECOMMENDED.items():
            assert s._weights[name] == val

    def test_recommended_redundant_zeroed(self):
        """ablation で冗長と判定された指標は重み 0 に固定。"""
        from src.old.scorer import LEARNED_WEIGHTS_RECOMMENDED
        zero_features = (
            "next_acceptance", "offset_power", "touching_density",
            "tail_height", "second_chain_potential", "key_flexibility",
            "shape_score", "chain_timing_pressure", "harassment_resistance",
        )
        for name in zero_features:
            assert LEARNED_WEIGHTS_RECOMMENDED[name] == 0.0

    def test_recommended_non_zero_count(self):
        """非ゼロ特徴は 7 個 (RECOMMENDED の最小構成)。"""
        from src.old.scorer import LEARNED_WEIGHTS_RECOMMENDED
        non_zero = sum(1 for v in LEARNED_WEIGHTS_RECOMMENDED.values() if v != 0.0)
        assert non_zero == 7

    def test_recommended_score_runs(self):
        """RECOMMENDED でも score() がエラーなく動く。"""
        from src.old.scorer import WEIGHT_SET_RECOMMENDED
        s = Scorer(weight_set=WEIGHT_SET_RECOMMENDED)
        r = s.score(ones(), zeros())
        assert SCORE_RANGE_MIN <= r.total_score <= SCORE_RANGE_MAX

