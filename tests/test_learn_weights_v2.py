"""
scripts/learn_weights_v2.py のロジックテスト

合成データで多モデル学習・split 関数・正規化処理を検証する。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from scripts.eda_features import Dataset, load_dataset  # noqa: E402
from scripts.generate_training_dataset import (  # noqa: E402
    DEFAULT_TIME_PHASES,
    FEATURE_NAMES,
)
from scripts.learn_weights_v2 import (  # noqa: E402
    accuracy,
    best_by_test_acc,
    default_weights_vector,
    evaluate_default,
    fit_lr_eval,
    fit_phase_lr,
    fit_rf_eval,
    normalize_weights_to_default_scale,
    phase_filter,
    predict_with_weights,
    random_split,
    search_lr,
    video_level_split,
)


def _make_csv(tmp_path: Path, n: int = 120) -> Path:
    """学習可能な合成 CSV (第 1, 2 特徴量で完全分離)。"""
    csv_path = tmp_path / "match_features.csv"
    rng = np.random.default_rng(42)
    header = ["video_id", "match_idx", "time_phase"]
    header.extend(FEATURE_NAMES)
    header.append("label")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            label = 1 if i % 2 == 0 else -1
            video = ["01", "02", "03"][i % 3]
            phase = DEFAULT_TIME_PHASES[i % 5]
            row: list[object] = [video, i, phase]
            for j, _name in enumerate(FEATURE_NAMES):
                if j == 0:
                    row.append(float(label) * 1.5 + rng.normal(0, 0.1))
                elif j == 1:
                    row.append(float(label) * 0.8 + rng.normal(0, 0.2))
                else:
                    row.append(rng.normal(0, 1.0))
            row.append(label)
            w.writerow(row)
    return csv_path


def _ds(tmp_path: Path) -> Dataset:
    return load_dataset(_make_csv(tmp_path))


# ============================
# split
# ============================


def test_video_level_split_separates_test_video(tmp_path: Path) -> None:
    """test_video の行が test に、それ以外が train に入る。"""
    ds = _ds(tmp_path)
    split = video_level_split(ds, test_video="03")
    assert all(ds.video_ids[i] == "03" for i in split.test_idx)
    assert all(ds.video_ids[i] != "03" for i in split.train_idx)


def test_random_split_disjoint(tmp_path: Path) -> None:
    """train / test が重ならない。"""
    ds = _ds(tmp_path)
    split = random_split(ds, train_ratio=0.7)
    assert set(split.train_idx).isdisjoint(set(split.test_idx))
    assert (
        len(split.train_idx) + len(split.test_idx) == len(ds.y)
    )


def test_phase_filter(tmp_path: Path) -> None:
    """phase 指定でその時刻だけ抽出される。"""
    ds = _ds(tmp_path)
    idx = phase_filter(ds, DEFAULT_TIME_PHASES[0])
    assert len(idx) > 0
    assert all(ds.time_phases[i] == DEFAULT_TIME_PHASES[0] for i in idx)


# ============================
# 評価
# ============================


def test_default_weights_vector_length() -> None:
    """重みベクトルが FEATURE_NAMES と同じ長さ。"""
    v = default_weights_vector()
    assert v.shape == (len(FEATURE_NAMES),)


def test_predict_with_weights_basic() -> None:
    """予測関数が +1/-1 のみを返す。"""
    X = np.array([[1.0], [-1.0]])
    w = np.array([1.0])
    pred = predict_with_weights(X, w)
    assert set(pred.tolist()) == {1, -1}


def test_accuracy() -> None:
    """精度計算。"""
    a = accuracy(np.array([1, 1, -1]), np.array([1, -1, -1]))
    assert a == pytest.approx(2 / 3)


def test_evaluate_default(tmp_path: Path) -> None:
    """DEFAULT_WEIGHTS の評価が float を返す。"""
    ds = _ds(tmp_path)
    split = video_level_split(ds)
    res = evaluate_default(ds, split)
    assert "train_acc" in res and "test_acc" in res


# ============================
# モデル学習
# ============================


def test_fit_lr_eval_returns_weights(tmp_path: Path) -> None:
    """LR が重み辞書と高精度を返す。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    res = fit_lr_eval(ds, split, "l2", C=1.0)
    assert res.test_acc > 0.7   # 合成データなのでそこそこ高精度
    assert len(res.weights) == len(ds.feature_names)


def test_fit_rf_eval_returns_feature_importance(tmp_path: Path) -> None:
    """RF が feature_importance を返す。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    res = fit_rf_eval(ds, split, max_depth=4, n_estimators=20)
    assert res.feature_importance is not None
    assert len(res.feature_importance) == len(ds.feature_names)


def test_search_lr_returns_multiple_results(tmp_path: Path) -> None:
    """search_lr が複数 ModelResult を返す。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    results = search_lr(ds, split)
    assert len(results) >= 4


def test_best_by_test_acc(tmp_path: Path) -> None:
    """best_by_test_acc は test_acc 最大を選ぶ。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    results = search_lr(ds, split)
    best = best_by_test_acc(results)
    for r in results:
        assert r.test_acc <= best.test_acc


# ============================
# normalize_weights
# ============================


def test_normalize_weights_zero_returns_input() -> None:
    """全 0 重みを与えると元のまま返す (ゼロ除算回避)。"""
    w = {n: 0.0 for n in FEATURE_NAMES}
    out = normalize_weights_to_default_scale(w)
    assert out == w


def test_normalize_weights_preserves_sign(tmp_path: Path) -> None:
    """符号は保たれ、スケールが調整される。"""
    ds = _ds(tmp_path)
    split = random_split(ds)
    res = fit_lr_eval(ds, split, "l2", C=1.0)
    norm = normalize_weights_to_default_scale(res.weights)
    for name, w in res.weights.items():
        if w == 0:
            continue
        assert (w > 0) == (norm[name] > 0)


# ============================
# phase 別モデル
# ============================


def test_fit_phase_lr_returns_dict(tmp_path: Path) -> None:
    """phase 別 LR が辞書を返す。"""
    ds = _ds(tmp_path)
    res = fit_phase_lr(ds, DEFAULT_TIME_PHASES[0])
    assert "phase" in res
    if not res.get("skipped"):
        assert "test_acc" in res
        assert "weights" in res
