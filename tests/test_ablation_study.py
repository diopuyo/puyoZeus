"""
scripts.ablation_study のスモークテスト
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.old.ablation_study import (
    IMPORTANT_DROP_THRESHOLD,
    NEGLIGIBLE_DROP_THRESHOLD,
    V3_DROPPED_FEATURES,
    classify_contribution,
    fit_ridge_acc,
    reduce_features,
    run_ablation,
    video_holdout_split,
)
from scripts.old.eda_features import Dataset
from scripts.old.generate_training_dataset import FEATURE_NAMES


# ============================
# テスト用 Dataset 構築
# ============================


def _make_dataset(n: int = 60, seed: int = 0) -> Dataset:
    """16 特徴量 × n サンプルの線形分離可能なダミーデータセット。"""
    rng = np.random.default_rng(seed)
    d = len(FEATURE_NAMES)
    X = rng.normal(size=(n, d))
    # 第 0 列の符号でラベルを決める (重要特徴量)
    y = np.where(X[:, 0] >= 0, 1, -1)
    # 動画 ID は 60 件中 40 件を 01,02 / 20 件を 03 に
    video_ids = ["01"] * 20 + ["02"] * 20 + ["03"] * (n - 40)
    time_phases = ["midpoint"] * n
    return Dataset(
        feature_names=FEATURE_NAMES,
        X=X, y=y, video_ids=video_ids, time_phases=time_phases,
    )


# ============================
# テスト
# ============================


class TestVideoHoldoutSplit:
    def test_split_correct(self):
        ds = _make_dataset()
        tr, te = video_holdout_split(ds, "03")
        assert len(tr) == 40 and len(te) == 20

    def test_no_overlap(self):
        ds = _make_dataset()
        tr, te = video_holdout_split(ds, "03")
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


class TestFitRidgeAcc:
    def test_returns_acc_keys(self):
        ds = _make_dataset()
        tr, te = video_holdout_split(ds, "03")
        res = fit_ridge_acc(ds.X[tr], ds.y[tr], ds.X[te], ds.y[te])
        assert "train_acc" in res and "test_acc" in res
        assert 0.0 <= res["test_acc"] <= 1.0

    def test_zero_features_dummy_predict(self):
        ds = _make_dataset()
        tr, te = video_holdout_split(ds, "03")
        empty = np.zeros((len(tr), 0))
        empty_te = np.zeros((len(te), 0))
        res = fit_ridge_acc(empty, ds.y[tr], empty_te, ds.y[te])
        assert "test_acc" in res


class TestRunAblation:
    def test_baseline_and_n_ablation(self):
        ds = _make_dataset()
        tr, te = video_holdout_split(ds, "03")
        res = run_ablation(ds, tr, te)
        assert "baseline" in res
        assert len(res["ablation"]) == len(ds.feature_names)
        # ソート後、先頭が最大 drop
        drops = [r["test_acc_drop"] for r in res["ablation"]]
        assert drops == sorted(drops, reverse=True)


class TestClassifyContribution:
    def test_three_classes(self):
        items = [
            {"removed": "a", "test_acc_drop": IMPORTANT_DROP_THRESHOLD + 0.01},
            {"removed": "b", "test_acc_drop": NEGLIGIBLE_DROP_THRESHOLD + 0.001},
            {"removed": "c", "test_acc_drop": -0.01},
        ]
        c = classify_contribution(items)
        assert "a" in c["important"]
        assert "b" in c["minor"]
        assert "c" in c["redundant"]


class TestReduceFeatures:
    def test_drop_v3(self):
        ds = _make_dataset()
        rd = reduce_features(ds, V3_DROPPED_FEATURES)
        assert all(n not in rd.feature_names for n in V3_DROPPED_FEATURES)
        assert len(rd.feature_names) == len(ds.feature_names) - len(V3_DROPPED_FEATURES)
