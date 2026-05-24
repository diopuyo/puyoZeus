"""Phase H2 学習スクリプト (280 features 対応).

H2 csv は学習スクリプトの FEATURE_NAMES (45) に縛られず、全数値列を
自動的に feature として扱う。LightGBM + LR L2 + Phase Aware を実装。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ============================
# 定数
# ============================
META_COLS = {"video_id", "match_idx", "time_phase", "frame_idx", "timestamp", "label"}
PHASES = ("start_plus_20", "mid_minus_20", "midpoint", "mid_plus_20", "end_minus_5")
GO_THRESHOLD = 0.04
PHASE_GROUPS: dict[str, tuple[str, ...]] = {
    "start": ("start_plus_20",),
    "mid": ("mid_minus_20", "midpoint", "mid_plus_20"),
    "end": ("end_minus_5",),
}


def load_h2_csv(path: Path) -> dict[str, Any]:
    """H2 csv を読み込み、自動で feature columns を抽出."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    feat_cols = [c for c in fieldnames if c not in META_COLS]
    n = len(rows)
    d = len(feat_cols)

    X = np.zeros((n, d), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    video_ids = []
    time_phases = []
    for i, r in enumerate(rows):
        for j, c in enumerate(feat_cols):
            X[i, j] = float(r.get(c, 0.0) or 0.0)
        y[i] = int(r["label"])
        video_ids.append(r["video_id"])
        time_phases.append(r.get("time_phase", "midpoint"))

    return {
        "X": X, "y": y,
        "video_ids": np.array(video_ids),
        "time_phases": np.array(time_phases),
        "feat_cols": feat_cols,
        "n": n, "d": d,
    }


def fit_lgbm(X_tr, y_tr, X_te, y_te, params: dict) -> tuple[float, float, Any]:
    y_tr_b = (y_tr > 0).astype(int)
    y_te_b = (y_te > 0).astype(int)
    clf = HistGradientBoostingClassifier(random_state=0, **params)
    clf.fit(X_tr, y_tr_b)
    train = float(clf.score(X_tr, y_tr_b))
    test = float(clf.score(X_te, y_te_b)) if len(X_te) > 0 else 0.0
    return train, test, clf


def fit_lr(X_tr, y_tr, X_te, y_te, C: float) -> tuple[float, float, Any]:
    y_tr_b = (y_tr > 0).astype(int)
    y_te_b = (y_te > 0).astype(int)
    clf = LogisticRegression(C=C, penalty="l2", max_iter=2000, random_state=0)
    clf.fit(X_tr, y_tr_b)
    train = float(clf.score(X_tr, y_tr_b))
    test = float(clf.score(X_te, y_te_b)) if len(X_te) > 0 else 0.0
    return train, test, clf


def video_holdout_split(video_ids: np.ndarray, n_test: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(video_ids)
    if len(uniq) <= n_test:
        n_test = max(1, len(uniq) // 3)
    test_videos = rng.choice(uniq, size=n_test, replace=False)
    test_mask = np.isin(video_ids, test_videos)
    return ~test_mask, test_mask


def loov_evaluate(ds: dict, fit_fn, params: dict) -> tuple[float, float]:
    uniq = np.unique(ds["video_ids"])
    accs = []
    for vid in uniq:
        test_mask = ds["video_ids"] == vid
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        try:
            _, te, _ = fit_fn(
                ds["X"][train_mask], ds["y"][train_mask],
                ds["X"][test_mask], ds["y"][test_mask],
                params,
            )
            accs.append(te)
        except Exception as e:
            print(f"  LOOV vid={vid} skip: {e}")
    if not accs:
        return 0.0, 0.0
    return float(np.mean(accs)), float(np.std(accs))


def loov_phase_evaluate(ds: dict, fit_fn, params: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for phase_name, phases in PHASE_GROUPS.items():
        phase_mask = np.isin(ds["time_phases"], phases)
        if phase_mask.sum() == 0:
            out[phase_name] = (0.0, 0.0)
            continue
        sub = {
            "X": ds["X"][phase_mask],
            "y": ds["y"][phase_mask],
            "video_ids": ds["video_ids"][phase_mask],
            "time_phases": ds["time_phases"][phase_mask],
        }
        mean, std = loov_evaluate(sub, fit_fn, params)
        out[phase_name] = (mean, std)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = load_h2_csv(args.csv)
    print(f"[load] n={ds['n']}, d={ds['d']}")
    print(f"  unique videos: {len(np.unique(ds['video_ids']))}")

    results: dict[str, Any] = {"n": ds['n'], "d": ds['d']}

    # ========== LGBM ==========
    print("\n=== LGBM video holdout ===")
    lgbm_params_set = [
        {"max_depth": 8, "learning_rate": 0.03, "max_iter": 500, "min_samples_leaf": 30, "max_leaf_nodes": 63},
        {"max_depth": 6, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 20, "max_leaf_nodes": 31},
        {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 20, "max_leaf_nodes": 15},
    ]
    train_mask, test_mask = video_holdout_split(ds["video_ids"], n_test=3, seed=0)
    best_acc = 0.0
    best_params = None
    best_clf = None
    for params in lgbm_params_set:
        tr, te, clf = fit_lgbm(
            ds["X"][train_mask], ds["y"][train_mask],
            ds["X"][test_mask], ds["y"][test_mask],
            params,
        )
        print(f"  {params}: train={tr:.3f}, test={te:.3f}")
        if te > best_acc:
            best_acc = te
            best_params = params
            best_clf = clf
    results["lgbm_video_holdout_test"] = best_acc
    results["lgbm_best_params"] = best_params
    print(f"\n[best LGBM] test={best_acc:.3f}")

    # LGBM LOOV (slow, skip if videos < 5)
    n_videos = len(np.unique(ds["video_ids"]))
    if n_videos >= 5:
        print("\n=== LGBM LOOV ===")
        loov_mean, loov_std = loov_evaluate(ds, fit_lgbm, best_params)
        results["lgbm_loov_mean"] = loov_mean
        results["lgbm_loov_std"] = loov_std
        print(f"  LOOV mean={loov_mean:.3f} std={loov_std:.3f}")

        print("\n=== LGBM Phase LOOV ===")
        phase_loov = loov_phase_evaluate(ds, fit_lgbm, best_params)
        for phase, (m, s) in phase_loov.items():
            print(f"  {phase}: mean={m:.3f} std={s:.3f}")
        results["lgbm_phase_loov"] = {p: {"mean": m, "std": s} for p, (m, s) in phase_loov.items()}
        avg = float(np.mean([m for m, _ in phase_loov.values()]))
        results["lgbm_phase_loov_avg"] = avg
        print(f"  average: {avg:.3f}")

    # ========== LR L2 ==========
    print("\n=== LR L2 video holdout ===")
    best_lr_acc = 0.0
    best_C = 1.0
    for C in [0.05, 0.1, 0.5, 1.0, 5.0]:
        try:
            tr, te, clf = fit_lr(
                ds["X"][train_mask], ds["y"][train_mask],
                ds["X"][test_mask], ds["y"][test_mask], C,
            )
            print(f"  C={C}: train={tr:.3f}, test={te:.3f}")
            if te > best_lr_acc:
                best_lr_acc = te
                best_C = C
        except Exception as e:
            print(f"  C={C} fail: {e}")
    results["lr_video_holdout_test"] = best_lr_acc
    results["lr_best_C"] = best_C

    if n_videos >= 5:
        print("\n=== LR Phase LOOV ===")
        phase_loov_lr = loov_phase_evaluate(ds, fit_lr, best_C)
        for phase, (m, s) in phase_loov_lr.items():
            print(f"  {phase}: mean={m:.3f} std={s:.3f}")
        results["lr_phase_loov"] = {p: {"mean": m, "std": s} for p, (m, s) in phase_loov_lr.items()}
        avg = float(np.mean([m for m, _ in phase_loov_lr.values()]))
        results["lr_phase_loov_avg"] = avg
        print(f"  average: {avg:.3f}")

    # ========== Permutation Importance ==========
    if best_clf is not None:
        print("\n=== Permutation Importance (LGBM, top 20) ===")
        try:
            y_test = (ds["y"][test_mask] > 0).astype(int)
            perm = permutation_importance(
                best_clf, ds["X"][test_mask], y_test,
                n_repeats=5, random_state=0, n_jobs=4,
            )
            order = np.argsort(perm.importances_mean)[::-1][:20]
            top = []
            for idx in order:
                fname = ds["feat_cols"][idx]
                m = float(perm.importances_mean[idx])
                s = float(perm.importances_std[idx])
                print(f"  {fname[:40]:<40} mean={m:+.4f} std={s:.4f}")
                top.append({"feature": fname, "importance_mean": m, "importance_std": s})
            results["lgbm_permutation_top20"] = top
        except Exception as e:
            print(f"  perm fail: {e}")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
