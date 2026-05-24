"""
scripts/final_evaluation.py のテスト。

- phase_to_elapsed の各時刻ラベル → 経過秒換算
- compute_diff_with_weights が指標差分を正しく重み付け
- predictor 群が ±1 を返すこと
- ensemble predictor が単純多数決として機能すること
- evaluate_predictor が phase / video / overall 集計を生成すること
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.final_evaluation import (  # noqa: E402
    FeatureRow,
    build_predictors,
    compute_diff_with_weights,
    evaluate_predictor,
    make_ensemble_predictor,
    make_phase_aware_predictor,
    make_static_predictor,
    phase_to_elapsed,
)
from src.indicators import (  # noqa: E402
    INDICATOR_DEATH_RISK,
    INDICATOR_MAIN_CHAIN,
)
from src.scorer import (  # noqa: E402
    DEFAULT_WEIGHTS,
    LEARNED_WEIGHTS_GLOBAL,
    WEIGHT_MODE_LEARNED,
    WEIGHT_MODE_OPTIMAL,
)


# ============================
# phase_to_elapsed
# ============================


class TestPhaseToElapsed:
    def test_start_plus_offsets(self) -> None:
        assert phase_to_elapsed("start_plus_0", 60.0) == 0.0
        assert phase_to_elapsed("start_plus_15", 60.0) == 15.0
        assert phase_to_elapsed("start_plus_30", 60.0) == 30.0

    def test_midpoint(self) -> None:
        assert phase_to_elapsed("midpoint", 60.0) == 30.0
        assert phase_to_elapsed("midpoint", 80.0) == 40.0

    def test_mid_offsets(self) -> None:
        assert phase_to_elapsed("mid_minus_15", 80.0) == 25.0
        assert phase_to_elapsed("mid_plus_30", 80.0) == 70.0

    def test_end_offsets(self) -> None:
        assert phase_to_elapsed("end_minus_15", 60.0) == 45.0
        assert phase_to_elapsed("end_minus_5", 60.0) == 55.0

    def test_unknown_phase_returns_midpoint(self) -> None:
        assert phase_to_elapsed("unknown_phase", 60.0) == 30.0

    def test_zero_duration_safe(self) -> None:
        # duration<=0 でも例外を投げず 0 系の値に丸める
        assert phase_to_elapsed("midpoint", 0.0) == 0.0
        assert phase_to_elapsed("end_minus_5", 0.0) == 0.0


# ============================
# 重み付け差分計算
# ============================


class TestComputeDiff:
    def test_main_chain_only(self) -> None:
        features = {INDICATOR_MAIN_CHAIN: 0.5}
        diff = compute_diff_with_weights(features, DEFAULT_WEIGHTS)
        expected = 0.5 * DEFAULT_WEIGHTS[INDICATOR_MAIN_CHAIN]
        assert diff == pytest.approx(expected, abs=1e-9)

    def test_missing_keys_zero(self) -> None:
        diff = compute_diff_with_weights({}, DEFAULT_WEIGHTS)
        assert diff == 0.0

    def test_negative_death_risk(self) -> None:
        features = {INDICATOR_DEATH_RISK: 1.0}
        # DEATH_RISK の重みは負なので diff も負
        diff = compute_diff_with_weights(features, DEFAULT_WEIGHTS)
        assert diff < 0


# ============================
# Predictor
# ============================


class TestPredictors:
    def test_static_predictor_returns_pm1(self) -> None:
        pred = make_static_predictor(DEFAULT_WEIGHTS)
        assert pred({INDICATOR_MAIN_CHAIN: 1.0}, 0.0, 60.0) == 1
        assert pred({INDICATOR_MAIN_CHAIN: -1.0}, 0.0, 60.0) == -1

    def test_phase_aware_predictor_runs(self) -> None:
        pred = make_phase_aware_predictor(WEIGHT_MODE_OPTIMAL)
        out = pred({INDICATOR_MAIN_CHAIN: 1.0}, 30.0, 60.0)
        assert out in (1, -1)

    def test_ensemble_unanimous_agrees(self) -> None:
        # 両方 +1 を返す predictor → +1
        always_one: callable = lambda f, e, d: 1  # noqa: E731
        always_neg: callable = lambda f, e, d: -1  # noqa: E731
        ens_pos = make_ensemble_predictor(always_one, always_one)
        assert ens_pos({}, 0.0, 60.0) == 1
        ens_neg = make_ensemble_predictor(always_neg, always_neg)
        assert ens_neg({}, 0.0, 60.0) == -1

    def test_ensemble_50_50_split_defaults_to_positive(self) -> None:
        """primary +1, secondary -1, weight=0.5 → combined=0 → +1 (>=0)。"""
        plus: callable = lambda f, e, d: 1  # noqa: E731
        minus: callable = lambda f, e, d: -1  # noqa: E731
        ens = make_ensemble_predictor(plus, minus, primary_weight=0.5)
        assert ens({}, 0.0, 60.0) == 1


# ============================
# evaluate_predictor
# ============================


class TestEvaluatePredictor:
    def _toy_rows(self) -> list[FeatureRow]:
        # 2 動画 × 2 試合 × 2 phase = 8 行 (label と main_chain を一致させる)
        rows: list[FeatureRow] = []
        for vid in ("01", "02"):
            for match in (1, 2):
                for tp in ("midpoint", "end_minus_5"):
                    label = 1 if match == 1 else -1
                    rows.append(FeatureRow(
                        video_id=vid,
                        match_idx=match,
                        time_phase=tp,
                        features={INDICATOR_MAIN_CHAIN: float(label)},
                        label=label,
                    ))
        return rows

    def test_perfect_predictor_acc_1(self) -> None:
        rows = self._toy_rows()
        durs = {("01", 1): 60.0, ("01", 2): 60.0,
                ("02", 1): 60.0, ("02", 2): 60.0}
        pred = make_static_predictor({INDICATOR_MAIN_CHAIN: 1.0})
        result = evaluate_predictor("perfect", pred, rows, durs)
        assert result.overall_accuracy == pytest.approx(1.0)
        # phase 集計が両方含まれる
        assert "midpoint" in result.per_phase
        assert "end_minus_5" in result.per_phase
        assert result.per_phase["midpoint"]["accuracy"] == 1.0
        # video 集計
        assert result.per_video["01"]["total"] == 4
        assert result.per_video["02"]["total"] == 4


# ============================
# build_predictors
# ============================


class TestBuildPredictors:
    def test_all_six_strategies(self) -> None:
        preds = build_predictors()
        expected = {
            "DEFAULT", "LEARNED_GLOBAL", "LEARNED_V3_GLOBAL",
            "PhaseAware_learned", "PhaseAware_optimal",
            "ENSEMBLE_optimal_default",
        }
        assert set(preds.keys()) == expected

    def test_each_predictor_callable(self) -> None:
        preds = build_predictors()
        for name, p in preds.items():
            out = p({INDICATOR_MAIN_CHAIN: 0.3}, 30.0, 60.0)
            assert out in (1, -1), f"{name} returned {out}"
