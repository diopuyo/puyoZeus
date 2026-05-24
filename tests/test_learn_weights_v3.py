"""scripts.learn_weights_v3 のスモークテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.eda_features import Dataset
from scripts.learn_weights_v3 import (
    DROPPED_FEATURES,
    PHYSICALLY_NEGATIVE_FEATURES,
    PHYSICALLY_POSITIVE_FEATURES,
    fit_elastic_net,
    fit_ridge_classifier,
    fit_sign_constrained_logreg,
    kfold_eval,
    per_phase_acc,
    reduce_features,
    reduced_to_dataset,
)
from scripts.learn_weights_v2 import Split, fit_lr_eval


def _make_full_dataset(seed: int = 0) -> Dataset:
    """全特徴量 (next_acceptance=0 を含む) のダミーデータ.

    Phase H1 (2026-05-08) で FEATURE_NAMES が 29 → 45 に拡張されたため、
    n を 200 → 400 に増やし overfit を防ぐ。テストは基本的なフィット動作の
    スモークなので、十分な信号 (label = main_chain > 0) を与えれば
    Ridge / ElasticNet で test_acc > 0.5 を達成可能。
    """
    from scripts.generate_training_dataset import FEATURE_NAMES
    rng = np.random.default_rng(seed)
    n = 400
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    # next_acceptance を 0 に固定 (VIF=inf の状況再現)
    na_idx = FEATURE_NAMES.index("next_acceptance")
    X[:, na_idx] = 0.0
    # main_chain_maturity と offset_power を強相関
    main_idx = FEATURE_NAMES.index("main_chain_maturity")
    off_idx = FEATURE_NAMES.index("offset_power")
    X[:, off_idx] = X[:, main_idx] + 0.05 * rng.standard_normal(n)
    y = np.where(X[:, main_idx] > 0, 1, -1)
    return Dataset(
        feature_names=FEATURE_NAMES,
        X=X, y=y,
        video_ids=["01"] * (n // 2) + ["03"] * (n - n // 2),
        time_phases=["midpoint"] * n,
    )


def test_reduce_features_drops_listed() -> None:
    """指定特徴量が削除されること。"""
    ds = _make_full_dataset()
    rd = reduce_features(ds)
    for name in DROPPED_FEATURES:
        assert name not in rd.feature_names
    assert len(rd.feature_names) == len(ds.feature_names) - len(DROPPED_FEATURES)
    assert rd.X.shape[1] == len(rd.feature_names)


def test_fit_elastic_net_returns_weights() -> None:
    """Elastic Net で重みが学習されること。"""
    ds = _make_full_dataset()
    rd = reduce_features(ds)
    sub = reduced_to_dataset(rd)
    split = Split(
        train_idx=np.arange(300), test_idx=np.arange(300, 400),
        label="random",
    )
    res = fit_elastic_net(sub, split, l1_ratio=0.3, C=1.0)
    assert res.model == "elasticnet"
    assert len(res.weights) == len(rd.feature_names)
    # main_chain_maturity が支配的なはず → 重み非ゼロ
    assert abs(res.weights["main_chain_maturity"]) > 0


def test_fit_ridge_classifier_works() -> None:
    """RidgeClassifier が学習・推論できること。"""
    ds = _make_full_dataset()
    rd = reduce_features(ds)
    sub = reduced_to_dataset(rd)
    split = Split(
        train_idx=np.arange(300), test_idx=np.arange(300, 400),
        label="random",
    )
    res = fit_ridge_classifier(sub, split, alpha=1.0)
    assert 0.0 <= res.test_acc <= 1.0
    assert res.test_acc > 0.5  # ランダム以上に学習できているはず


def test_fit_sign_constrained_returns_positive_for_constrained() -> None:
    """物理的に正の指標は重みが非負になっていること。"""
    ds = _make_full_dataset()
    rd = reduce_features(ds)
    sub = reduced_to_dataset(rd)
    split = Split(
        train_idx=np.arange(300), test_idx=np.arange(300, 400),
        label="random",
    )
    res = fit_sign_constrained_logreg(sub, split, alpha=1.0)
    # 正値制約された指標の重みは >= -1e-6 (浮動小数誤差許容)
    for n in PHYSICALLY_POSITIVE_FEATURES:
        if n in res.weights:
            assert res.weights[n] >= -1e-6, (
                f"{n} の重みが負: {res.weights[n]}"
            )
    # 負値制約された指標は <= +1e-6
    for n in PHYSICALLY_NEGATIVE_FEATURES:
        if n in res.weights:
            assert res.weights[n] <= 1e-6, (
                f"{n} の重みが正: {res.weights[n]}"
            )


def test_kfold_eval_returns_mean_std() -> None:
    """K-fold 評価が平均/分散を返すこと。"""
    ds = _make_full_dataset()
    rd = reduce_features(ds)
    sub = reduced_to_dataset(rd)
    out = kfold_eval(sub, k=3, model_fn=fit_lr_eval, penalty="l2", C=1.0)
    assert "mean" in out and "std" in out
    assert 0.0 <= out["mean"] <= 1.0
    assert len(out["folds"]) == 3


def test_per_phase_acc_returns_dict() -> None:
    """phase 別 acc が辞書形式で返ること。"""
    ds = _make_full_dataset()
    weights = np.zeros(ds.X.shape[1])
    out = per_phase_acc(ds, weights)
    assert "midpoint" in out
    assert "n" in out["midpoint"] and "acc" in out["midpoint"]
