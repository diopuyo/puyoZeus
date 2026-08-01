"""#24 打ち合い計測器 共通評価ハーネス (scripts/exchange_meter_eval_harness.py) の単体テスト。

測定器事故4件目 (train_board_cnn の AUC近似バグ) の教訓に従い、
DeLong検定内部の厳密AUC計算が sklearn.roc_auc_score と完全一致することを
必ず確認する。GroupKFold のグループ非交差・ブートストラップの動画クラスタ
単位性・reliability table のビン集計も検証する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from scripts.exchange_meter_eval_harness import (
    MIN_PHASE_N_FOR_POWER,
    POWER_INSUFFICIENT_LABEL,
    PredictorPredictions,
    _fast_delong,
    bootstrap_ci_by_video,
    bootstrap_diff_ci_by_video,
    build_pairwise_delong_table,
    compare_predictors,
    compute_reliability_table,
    delong_paired_test,
    exact_auc,
    group_kfold_splits,
    phase_power_flag,
)

RANDOM_SEED = 12345


def _make_synthetic_binary(n: int, seed: int = RANDOM_SEED, ties: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """AUC検定用の合成二値データ (正例スコアがやや高い) を作る。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    score = rng.normal(loc=0.0, scale=1.0, size=n) + y * 0.8
    if ties:
        score = np.round(score, 1)  # 同点を意図的に作る
    return y, score


# =============================================================================
# 測定器事故4件目対策: 厳密AUC が sklearn と一致するか
# =============================================================================

class TestExactAucMatchesSklearn:
    """exact_auc / _fast_delong の内部AUCが sklearn.roc_auc_score と一致するか。"""

    @pytest.mark.parametrize("n", [20, 100, 500, 5000])
    def test_exact_auc_matches_sklearn(self, n: int) -> None:
        y, score = _make_synthetic_binary(n)
        expected = roc_auc_score(y, score)
        assert exact_auc(y, score) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("n", [50, 500])
    def test_exact_auc_matches_sklearn_with_ties(self, n: int) -> None:
        y, score = _make_synthetic_binary(n, ties=True)
        expected = roc_auc_score(y, score)
        assert exact_auc(y, score) == pytest.approx(expected, abs=1e-9)

    def test_fast_delong_auc_matches_sklearn(self) -> None:
        y, score = _make_synthetic_binary(300)
        pos_mask = y == 1
        order = np.concatenate([np.where(pos_mask)[0], np.where(~pos_mask)[0]])
        aucs, _ = _fast_delong(score[order].reshape(1, -1), int(pos_mask.sum()))
        assert aucs[0] == pytest.approx(roc_auc_score(y, score), abs=1e-9)

    def test_exact_auc_single_class_returns_nan(self) -> None:
        y = np.zeros(10)
        score = np.random.default_rng(0).normal(size=10)
        assert np.isnan(exact_auc(y, score))


# =============================================================================
# DeLong ペアード検定
# =============================================================================

class TestDeLongPairedTest:
    """delong_paired_test の妥当性 (同一スコア=差0、明確な優劣=有意)。"""

    def test_identical_scores_gives_zero_diff(self) -> None:
        y, score = _make_synthetic_binary(200)
        result = delong_paired_test(y, score, score.copy())
        assert result.auc_diff == pytest.approx(0.0, abs=1e-9)

    def test_clearly_better_model_gives_significant_result(self) -> None:
        rng = np.random.default_rng(1)
        n = 400
        y = rng.integers(0, 2, size=n)
        score_good = rng.normal(size=n) + y * 3.0  # 分離良好
        score_bad = rng.normal(size=n)  # ランダム (AUC~0.5)
        result = delong_paired_test(y, score_good, score_bad)
        assert result.auc_a > 0.9
        assert result.auc_diff > 0.3
        assert result.p_value < 0.01

    def test_single_class_returns_nan(self) -> None:
        y = np.ones(10)
        score_a = np.random.default_rng(0).normal(size=10)
        score_b = np.random.default_rng(1).normal(size=10)
        result = delong_paired_test(y, score_a, score_b)
        assert np.isnan(result.auc_diff)
        assert np.isnan(result.p_value)


# =============================================================================
# GroupKFold (video_id 単位)
# =============================================================================

class TestGroupKFoldSplits:
    """グループ (video_id) が train/test 間で交差しないことを確認する。"""

    def test_no_group_overlap_between_folds(self) -> None:
        rng = np.random.default_rng(2)
        groups = np.repeat([f"v{i}" for i in range(10)], 20)
        rng.shuffle(groups)
        splits = group_kfold_splits(len(groups), groups, n_splits=5)
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups)

    def test_clamps_splits_to_available_groups(self) -> None:
        groups = np.array(["v0", "v0", "v1", "v1"])
        splits = group_kfold_splits(len(groups), groups, n_splits=5)
        assert len(splits) == 2  # 動画2本しかないので2foldにクランプ


# =============================================================================
# ブートストラップ CI (動画クラスタ単位)
# =============================================================================

class TestBootstrapCiByVideo:
    """動画クラスタ単位のブートストラップが正しく機能するか。"""

    def test_point_estimate_matches_direct_metric(self) -> None:
        rng = np.random.default_rng(3)
        n = 300
        video_ids = rng.choice([f"v{i}" for i in range(15)], size=n)
        y_true = rng.normal(size=n)
        y_pred = y_true + rng.normal(scale=0.1, size=n)

        def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean(np.abs(y_true - y_pred)))

        result = bootstrap_ci_by_video(
            _mae, video_ids, {"y_true": y_true, "y_pred": y_pred}, n_resamples=50,
        )
        assert result.point == pytest.approx(_mae(y_true, y_pred))
        assert result.ci_low <= result.point <= result.ci_high

    def test_diff_ci_point_equals_metric_difference(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        video_ids = rng.choice([f"v{i}" for i in range(10)], size=n)
        y_true = rng.normal(size=n)
        pred_a = y_true + rng.normal(scale=0.05, size=n)
        pred_b = y_true + rng.normal(scale=0.5, size=n)

        def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean(np.abs(y_true - y_pred)))

        result = bootstrap_diff_ci_by_video(
            _mae, video_ids,
            {"y_true": y_true, "y_pred": pred_a},
            {"y_true": y_true, "y_pred": pred_b},
            n_resamples=50,
        )
        expected = _mae(y_true, pred_a) - _mae(y_true, pred_b)
        assert result.point == pytest.approx(expected)
        assert result.point < 0  # pred_a の方が誤差が小さいので差は負

    def test_resamples_by_video_not_by_event(self) -> None:
        """同一動画内は常に同じ値を持つ変数で、CIが単一動画由来の値しか取らないことを確認する。"""
        video_ids = np.array(["v0"] * 5 + ["v1"] * 5 + ["v2"] * 5)
        # 動画ごとに定数値 (イベント単位でばらけていたら平均は連続値になるはず)
        per_video_value = {"v0": 1.0, "v1": 2.0, "v2": 3.0}
        y_pred = np.array([per_video_value[v] for v in video_ids])
        y_true = np.zeros(len(video_ids))

        def _mean_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean(y_pred))

        result = bootstrap_ci_by_video(
            _mean_pred, video_ids, {"y_true": y_true, "y_pred": y_pred}, n_resamples=500,
        )
        # 3動画から復元抽出した平均は 1.0,2.0,3.0 の組合せ平均のみを取りうる
        possible_means = {round(np.mean(c), 6) for c in
                           __import__("itertools").product([1.0, 2.0, 3.0], repeat=3)}
        assert round(result.point, 6) in possible_means


# =============================================================================
# reliability table
# =============================================================================

class TestReliabilityTable:
    """compute_reliability_table のビン集計が正しいか。"""

    def test_bin_counts_and_means(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_prob = np.array([0.05, 0.05, 0.95, 0.95, 0.55, 0.55])
        table = compute_reliability_table(y_true, y_prob, n_bins=10)
        assert table["n"].sum() == len(y_true)
        bin_0 = table[table["bin"] == "0.0-0.1"].iloc[0]
        assert bin_0["n"] == 2
        assert bin_0["actual_rate"] == pytest.approx(0.0)
        bin_9 = table[table["bin"] == "0.9-1.0"].iloc[0]
        assert bin_9["n"] == 2
        assert bin_9["actual_rate"] == pytest.approx(1.0)


# =============================================================================
# 位相別検定力フラグ (silent cap 禁止)
# =============================================================================

class TestPhasePowerFlag:
    """MIN_PHASE_N_FOR_POWER 未満で警告ラベルが付与されるか。"""

    def test_below_threshold_flagged(self) -> None:
        assert phase_power_flag(MIN_PHASE_N_FOR_POWER - 1) == POWER_INSUFFICIENT_LABEL

    def test_at_threshold_not_flagged(self) -> None:
        assert phase_power_flag(MIN_PHASE_N_FOR_POWER) == ""


# =============================================================================
# 統合テスト: compare_predictors (三つ巴比較の実行系)
# =============================================================================

class TestComparePredictorsIntegration:
    """compare_predictors が最後まで実行でき、必要な出力を生成するか。"""

    def _make_df(self, n: int = 600, n_videos: int = 12) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        video_ids = rng.choice([f"video_v{i}" for i in range(n_videos)], size=n)
        phases = rng.choice(["序", "中", "終"], size=n, p=[0.2, 0.3, 0.5])
        taiou = rng.integers(0, 2, size=n)
        net_ojama = rng.normal(loc=50.0, scale=30.0, size=n)
        return pd.DataFrame({
            "video_id": video_ids, "phase": phases,
            "taiou_success": taiou, "net_ojama_after": net_ojama,
        })

    def test_compare_predictors_runs_end_to_end(self, tmp_path) -> None:
        df = self._make_df()
        rng = np.random.default_rng(6)
        pred_d = PredictorPredictions(
            name="案D",
            prob_taiou_success=np.clip(df["taiou_success"].values * 0.6 + rng.normal(scale=0.2, size=len(df)), 0, 1),
            net_ojama_after_pred=df["net_ojama_after"].values + rng.normal(scale=10.0, size=len(df)),
        )
        pred_sim = PredictorPredictions(
            name="修正シミュ",
            prob_taiou_success=np.clip(rng.uniform(size=len(df)), 0, 1),
            net_ojama_after_pred=df["net_ojama_after"].values + rng.normal(scale=40.0, size=len(df)),
        )
        out_dir = tmp_path / "exchange_meter_compare"
        result = compare_predictors(df, [pred_d, pred_sim], out_dir, n_resamples=20)

        assert set(result.keys()) == {"scope", "delong_pairs", "bootstrap_pairs"}
        assert (out_dir / "comparison_report.md").exists()
        assert (out_dir / "reliability_diagrams.png").exists()
        assert len(result["scope"]) == 4 * 2  # (全体+3位相) x 2予測器
        assert len(result["delong_pairs"]) == 4  # (全体+3位相) x 1ペア

    def test_phase_with_few_events_flagged_in_delong_table(self, tmp_path) -> None:
        rng = np.random.default_rng(7)
        n = 100
        df = pd.DataFrame({
            "video_id": rng.choice(["va", "vb", "vc"], size=n),
            "phase": ["序"] * n,  # 全て同一位相かつ n<200 -> 参考値フラグ
            "taiou_success": rng.integers(0, 2, size=n),
            "net_ojama_after": rng.normal(size=n),
        })
        pred_a = PredictorPredictions("A", rng.uniform(size=n), rng.normal(size=n))
        pred_b = PredictorPredictions("B", rng.uniform(size=n), rng.normal(size=n))
        table = build_pairwise_delong_table(df, [pred_a, pred_b])
        row = table[table["範囲"] == "序"].iloc[0]
        assert row["備考"] == POWER_INSUFFICIENT_LABEL
