"""Phase H3 ensemble スクリプト (LR + HGBT + RF の確率出力 ensemble).

目的:
    各モデルの video_holdout 確率出力を平均/加重平均し、ensemble の test acc が
    単独最良を超えるか確認する。

入力:
    --csv data/training/match_features_phase_h2_quick_phased.csv
出力:
    --out data/verify/phase_h3_ensemble.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ============================
# 定数
# ============================
META_COLS = {"video_id", "match_idx", "time_phase", "frame_idx", "timestamp", "label"}
N_TEST_VIDEOS = 3
RANDOM_SEED = 0
LR_C = 0.5
HGBT_PARAMS = {
    "max_depth": 6, "learning_rate": 0.05, "max_iter": 300,
    "min_samples_leaf": 20, "max_leaf_nodes": 31,
}
RF_PARAMS = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 10}
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_h2_csv(path: Path) -> dict[str, Any]:
    """H2 csv を読み込む (他スクリプトと同仕様)."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    feat_cols = [c for c in fieldnames if c not in META_COLS]
    n, d = len(rows), len(feat_cols)
    X = np.zeros((n, d), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    video_ids: list[str] = []
    for i, r in enumerate(rows):
        for j, c in enumerate(feat_cols):
            X[i, j] = float(r.get(c, 0.0) or 0.0)
        y[i] = int(r["label"])
        video_ids.append(r["video_id"])
    return {
        "X": X, "y": y,
        "video_ids": np.array(video_ids),
        "feat_cols": feat_cols,
        "n": n, "d": d,
    }


def video_holdout_split(
    video_ids: np.ndarray, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(video_ids)
    if len(uniq) <= n_test:
        n_test = max(1, len(uniq) // 3)
    test_videos = rng.choice(uniq, size=n_test, replace=False)
    test_mask = np.isin(video_ids, test_videos)
    return ~test_mask, test_mask


def fit_proba(model_name: str, X_tr, y_tr, X_te) -> np.ndarray:
    """指定モデルを学習して X_te の class=1 確率を返す."""
    y_tr_b = (y_tr > 0).astype(int)
    if model_name == "lr":
        clf = LogisticRegression(
            C=LR_C, penalty="l2", max_iter=2000, random_state=RANDOM_SEED
        )
    elif model_name == "hgbt":
        clf = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **HGBT_PARAMS)
    elif model_name == "rf":
        clf = RandomForestClassifier(
            random_state=RANDOM_SEED, n_jobs=4, **RF_PARAMS
        )
    else:
        raise ValueError(f"unknown model {model_name}")
    clf.fit(X_tr, y_tr_b)
    proba = clf.predict_proba(X_te)
    return proba[:, 1]


def grid_search_weights(
    probas: dict[str, np.ndarray], y_te_b: np.ndarray
) -> tuple[dict[str, float], float]:
    """3 モデルの重み (合計 1) を grid search で探索."""
    best_w = {"lr": 1.0, "hgbt": 0.0, "rf": 0.0}
    best_acc = 0.0
    for w_lr, w_hgbt, w_rf in product(WEIGHT_GRID, repeat=3):
        total = w_lr + w_hgbt + w_rf
        if total <= 0:
            continue
        wl, wh, wr = w_lr / total, w_hgbt / total, w_rf / total
        merged = wl * probas["lr"] + wh * probas["hgbt"] + wr * probas["rf"]
        pred = (merged >= 0.5).astype(int)
        acc = float(np.mean(pred == y_te_b))
        if acc > best_acc:
            best_acc = acc
            best_w = {"lr": wl, "hgbt": wh, "rf": wr}
    return best_w, best_acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = load_h2_csv(args.csv)
    print(f"[load] n={ds['n']}, d={ds['d']}, videos={len(np.unique(ds['video_ids']))}")

    train_mask, test_mask = video_holdout_split(
        ds["video_ids"], n_test=N_TEST_VIDEOS, seed=RANDOM_SEED
    )
    X_tr, X_te = ds["X"][train_mask], ds["X"][test_mask]
    y_tr, y_te = ds["y"][train_mask], ds["y"][test_mask]
    y_te_b = (y_te > 0).astype(int)

    probas: dict[str, np.ndarray] = {}
    individual: dict[str, float] = {}
    for name in ("lr", "hgbt", "rf"):
        print(f"[fit] {name}")
        p = fit_proba(name, X_tr, y_tr, X_te)
        probas[name] = p
        acc = float(np.mean((p >= 0.5).astype(int) == y_te_b))
        individual[name] = acc
        print(f"  {name} vh={acc:.3f}")

    # 等分平均 ensemble
    eq = (probas["lr"] + probas["hgbt"] + probas["rf"]) / 3.0
    eq_acc = float(np.mean((eq >= 0.5).astype(int) == y_te_b))
    print(f"\n[ensemble] equal-weight vh={eq_acc:.3f}")

    # 加重最適化
    best_w, best_acc = grid_search_weights(probas, y_te_b)
    print(f"[ensemble] grid best vh={best_acc:.3f} w={best_w}")

    payload = {
        "n": ds["n"],
        "d": ds["d"],
        "individual_video_holdout": individual,
        "ensemble_equal_weight": eq_acc,
        "ensemble_grid_best": best_acc,
        "ensemble_grid_weights": best_w,
        "improvement_vs_best_individual": best_acc - max(individual.values()),
        "baseline_h2_lr_vh": 0.7396,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
