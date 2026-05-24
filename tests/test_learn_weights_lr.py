"""
scripts/learn_weights_lr.py のロジックテスト

合成 MatchSample で sklearn LR 学習パイプラインを検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from scripts.learn_weights_lr import (  # noqa: E402
    DEFAULT_REGULARIZATION,
    LR_FEATURE_NAMES,
    FeatureMatrix,
    LearnedModel,
    build_features,
    fit_lr,
    kfold_lr_scores,
    predict_with_weights,
    run_holdout_lr,
)
from scripts.tune_weights import MatchSample  # noqa: E402
from src.indicators import ALL_INDICATOR_NAMES  # noqa: E402


# ============================
# サンプル生成ユーティリティ
# ============================


def _make_samples(n: int = 50, separation: float = 0.6) -> list[MatchSample]:
    """合成 MatchSample を生成する (奇数=1P 勝、偶数=2P 勝)。"""
    out: list[MatchSample] = []
    empty = {n_: 0.0 for n_ in ALL_INDICATOR_NAMES}
    extra_zero = {
        "shape_score": 0.0, "touching_density": 0.0,
        "tail_height": 0.0, "color_variance": 0.0,
    }
    high, low = separation, 1.0 - separation
    for i in range(1, n + 1):
        is_p1 = (i % 2 == 1)
        winner = "1P" if is_p1 else "2P"
        p1 = {**empty, **extra_zero}
        p2 = {**empty, **extra_zero}
        for k in ALL_INDICATOR_NAMES:
            if k == "death_risk":
                p1[k] = low if is_p1 else high
                p2[k] = high if is_p1 else low
            else:
                p1[k] = high if is_p1 else low
                p2[k] = low if is_p1 else high
        out.append(MatchSample(
            idx=i, end_sec=float(i * 60),
            winner=winner, p1_scores=p1, p2_scores=p2,
        ))
    return out


# ============================
# build_features
# ============================


class TestBuildFeatures:
    def test_shapes(self):
        samples = _make_samples(n=10)
        fm = build_features(samples)
        assert isinstance(fm, FeatureMatrix)
        assert fm.X.shape == (10, len(LR_FEATURE_NAMES))
        assert fm.y.shape == (10,)
        assert isinstance(fm.X, np.ndarray)

    def test_labels_are_pm1(self):
        samples = _make_samples(n=10)
        fm = build_features(samples)
        assert set(fm.y.tolist()) <= {1, -1}

    def test_diff_signs(self):
        samples = _make_samples(n=4)
        fm = build_features(samples)
        # idx=1 (1P 勝) → main_chain は p1 - p2 = high - low > 0
        col = LR_FEATURE_NAMES.index("main_chain_maturity")
        assert fm.X[0, col] > 0
        assert fm.y[0] == 1
        # idx=2 (2P 勝) → main_chain は p1 - p2 < 0
        assert fm.X[1, col] < 0
        assert fm.y[1] == -1


# ============================
# fit_lr
# ============================


class TestFitLR:
    def test_returns_numpy_coef(self):
        samples = _make_samples(n=30)
        fm = build_features(samples)
        model = fit_lr(fm, regularization=0.5)
        assert isinstance(model, LearnedModel)
        assert isinstance(model.coef, np.ndarray)
        assert model.coef.shape == (len(LR_FEATURE_NAMES),)
        assert isinstance(model.intercept, float)

    def test_high_train_acc_separable(self):
        samples = _make_samples(n=30, separation=0.9)
        fm = build_features(samples)
        model = fit_lr(fm, regularization=0.1)
        assert model.train_acc >= 0.9

    def test_to_weights_dict(self):
        samples = _make_samples(n=30)
        fm = build_features(samples)
        model = fit_lr(fm)
        w = model.to_weights_dict()
        assert set(w.keys()) == set(LR_FEATURE_NAMES)
        for v in w.values():
            assert isinstance(v, float)

    def test_regularization_strength(self):
        """強い正則化で係数 L2 ノルムが小さくなる。"""
        samples = _make_samples(n=30, separation=0.8)
        fm = build_features(samples)
        weak = fit_lr(fm, regularization=0.01)
        strong = fit_lr(fm, regularization=10.0)
        weak_norm = float(np.linalg.norm(weak.coef))
        strong_norm = float(np.linalg.norm(strong.coef))
        assert strong_norm <= weak_norm


# ============================
# predict_with_weights
# ============================


class TestPredictWithWeights:
    def test_perfect_weights(self):
        samples = _make_samples(n=10, separation=0.9)
        # main_chain だけ正の重みでも分離可能
        weights = {n: 0.0 for n in LR_FEATURE_NAMES}
        weights["main_chain_maturity"] = 1.0
        weights["death_risk"] = -1.0
        acc = predict_with_weights(samples, weights, intercept=0.0)
        assert acc == pytest.approx(1.0)

    def test_zero_weights(self):
        samples = _make_samples(n=4)
        weights = {n: 0.0 for n in LR_FEATURE_NAMES}
        # diff = 0 はスキップされ total=0 → 0.0
        acc = predict_with_weights(samples, weights, intercept=0.0)
        assert acc == 0.0


# ============================
# run_holdout_lr / kfold_lr_scores
# ============================


class TestRunHoldoutLR:
    def test_keys_and_ranges(self):
        samples = _make_samples(n=40, separation=0.7)
        result = run_holdout_lr(
            samples, train_ratio=0.5, regularization=0.5, seed=42,
        )
        for key in (
            "train_acc", "test_acc", "weights", "intercept",
            "n_train", "n_test", "generalization_gap", "overfit_flag",
        ):
            assert key in result
        assert 0.0 <= result["train_acc"] <= 1.0
        assert 0.0 <= result["test_acc"] <= 1.0
        assert result["n_train"] == 20
        assert result["n_test"] == 20

    def test_separable_high_test_acc(self):
        samples = _make_samples(n=40, separation=0.9)
        result = run_holdout_lr(
            samples, train_ratio=0.5, regularization=0.5, seed=42,
        )
        assert result["test_acc"] >= 0.85


class TestKFoldLRScores:
    def test_kfold_disjoint(self):
        samples = _make_samples(n=20)
        result = kfold_lr_scores(samples, k=5, seed=42)
        assert result["k"] == 5
        assert len(result["fold_results"]) == 5
        n_test_total = sum(f["n_test"] for f in result["fold_results"])
        assert n_test_total == len(samples)

    def test_kfold_reproducible(self):
        samples = _make_samples(n=20)
        a = kfold_lr_scores(samples, k=5, seed=42)
        b = kfold_lr_scores(samples, k=5, seed=42)
        assert a["test_mean"] == pytest.approx(b["test_mean"])
        assert a["test_std"] == pytest.approx(b["test_std"])


# ============================
# 設定整合性
# ============================


class TestConfig:
    def test_default_regularization_positive(self):
        assert DEFAULT_REGULARIZATION > 0.0

    def test_feature_names_nonempty(self):
        assert len(LR_FEATURE_NAMES) >= 8
