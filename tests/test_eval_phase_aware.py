"""
scripts/eval_phase_aware.py / scripts/eval_ensemble.py の smoke テスト。

実データに依存しない最小データで評価ロジックが動くことを確認する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval_ensemble import (
    evaluate_alpha,
    grid_search_alpha,
    weighted_diff,
)
from scripts.eval_phase_aware import (
    FeatureRow,
    evaluate_strategy,
    load_feature_rows,
    load_match_durations,
    phase_to_elapsed,
    predict_with_weights,
    run_all_strategies,
)
from src.scorer import (
    DEFAULT_WEIGHTS,
    LEARNED_WEIGHTS_GLOBAL,
    PhaseAwareScorer,
)


# ============================
# 入出力ヘルパ
# ============================


@pytest.fixture
def csv_with_two_rows(tmp_path: Path) -> Path:
    """end_minus_5 行 2 件を含む最小 CSV を生成する。"""
    p = tmp_path / "match_features.csv"
    header = (
        "video_id,match_idx,time_phase,main_chain_maturity,extension_potential,"
        "sub_chain_quality,harassment_resistance,death_risk,offset_power,"
        "second_chain_potential,field_efficiency,next_acceptance,shape_score,"
        "touching_density,tail_height,color_variance,key_flexibility,"
        "sub_chain_independence,chain_timing_pressure,label\n"
    )
    body = (
        # 1P 圧勝サンプル (label=1)
        "01,1,end_minus_5,0.5,0.4,0.3,0.2,-0.5,0.4,0.3,0.2,"
        "0.0,0.1,0.1,0.0,0.0,0.0,0.0,0.0,1\n"
        # 2P 圧勝サンプル (label=-1)
        "01,2,end_minus_5,-0.5,-0.4,-0.3,-0.2,0.5,-0.4,-0.3,-0.2,"
        "0.0,-0.1,-0.1,0.0,0.0,0.0,0.0,0.0,-1\n"
    )
    p.write_text(header + body, encoding="utf-8")
    return p


@pytest.fixture
def boundaries_root(tmp_path: Path) -> Path:
    """video_01/matches.tsv だけを持つ最小 boundaries root。"""
    root = tmp_path / "match_boundaries"
    sub = root / "video_01"
    sub.mkdir(parents=True)
    (sub / "matches.tsv").write_text(
        "idx\tstart_sec\tend_sec\tduration_sec\n"
        "1\t0.0\t60.0\t60.0\n"
        "2\t60.0\t120.0\t60.0\n",
        encoding="utf-8",
    )
    return root


# ============================
# load 系
# ============================


class TestLoad:
    def test_load_feature_rows(self, csv_with_two_rows: Path) -> None:
        rows = load_feature_rows(csv_with_two_rows)
        assert len(rows) == 2
        assert isinstance(rows[0], FeatureRow)
        assert rows[0].label == 1
        assert rows[1].label == -1

    def test_load_match_durations(self, boundaries_root: Path) -> None:
        durations = load_match_durations(boundaries_root)
        assert durations[("01", 1)] == pytest.approx(60.0)
        assert durations[("01", 2)] == pytest.approx(60.0)


# ============================
# 予測ロジック
# ============================


class TestPredictionLogic:
    def test_predict_with_weights_1p(self) -> None:
        features = {n: 0.5 for n in DEFAULT_WEIGHTS}
        # ほぼすべての DEFAULT 重みは正なので 1P 予測になるはず
        pred = predict_with_weights(features, DEFAULT_WEIGHTS)
        assert pred == 1

    def test_predict_with_weights_2p(self) -> None:
        features = {n: -0.5 for n in DEFAULT_WEIGHTS}
        pred = predict_with_weights(features, DEFAULT_WEIGHTS)
        assert pred == -1

    def test_phase_to_elapsed_end(self) -> None:
        assert phase_to_elapsed("end_minus_5", 60.0) == pytest.approx(55.0)

    def test_phase_to_elapsed_start(self) -> None:
        assert phase_to_elapsed("start_plus_20", 60.0) == pytest.approx(20.0)

    def test_phase_to_elapsed_midpoint(self) -> None:
        assert phase_to_elapsed("midpoint", 60.0) == pytest.approx(30.0)


# ============================
# evaluate_strategy / run_all_strategies
# ============================


class TestEvaluateStrategy:
    def test_evaluate_strategy_smoke(
        self,
        csv_with_two_rows: Path,
        boundaries_root: Path,
    ) -> None:
        rows = load_feature_rows(csv_with_two_rows)
        durations = load_match_durations(boundaries_root)
        result = evaluate_strategy(
            "DEFAULT", rows, durations, weights=DEFAULT_WEIGHTS,
        )
        assert result.overall_total == 2
        assert 0.0 <= result.overall_accuracy <= 1.0

    def test_run_all_strategies_smoke(
        self,
        csv_with_two_rows: Path,
        boundaries_root: Path,
    ) -> None:
        rows = load_feature_rows(csv_with_two_rows)
        durations = load_match_durations(boundaries_root)
        results = run_all_strategies(rows, durations)
        # 4 戦略すべて返ること
        names = {r.strategy for r in results}
        assert names == {
            "DEFAULT", "LEARNED_GLOBAL",
            "PhaseAware_discrete", "PhaseAware_interpolated",
        }


# ============================
# eval_ensemble.py
# ============================


class TestEnsemble:
    def test_weighted_diff_zero_for_zero_features(self) -> None:
        features = {n: 0.0 for n in DEFAULT_WEIGHTS}
        assert weighted_diff(features, DEFAULT_WEIGHTS) == pytest.approx(0.0)

    def test_evaluate_alpha_smoke(
        self,
        csv_with_two_rows: Path,
        boundaries_root: Path,
    ) -> None:
        rows = load_feature_rows(csv_with_two_rows)
        durations = load_match_durations(boundaries_root)
        pa = PhaseAwareScorer(interpolate=True)
        result = evaluate_alpha(0.5, rows, durations, pa)
        assert result.overall_total == 2
        assert 0.0 <= result.overall_accuracy <= 1.0

    def test_grid_search_returns_best(
        self,
        csv_with_two_rows: Path,
        boundaries_root: Path,
    ) -> None:
        rows = load_feature_rows(csv_with_two_rows)
        durations = load_match_durations(boundaries_root)
        results, best = grid_search_alpha(rows, durations)
        # ALPHA_GRID 数だけ結果が返る
        assert len(results) == 5
        # best は results 内のいずれか
        assert best in results
