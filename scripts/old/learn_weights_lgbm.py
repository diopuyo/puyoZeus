"""M-A: HistGradientBoosting (sklearn の histogram-based GBM) で非線形学習.

LightGBM と同じアルゴリズムで、libgomp 不要・純 python で動作。
線形 LR L2 では test_acc 0.66 で頭打ち (E-2 結果)。
GBM で指標間の非線形相互作用を捕捉し、0.72+ を狙う。

評価軸:
    - video holdout (1 動画 hold out)
    - leave-one-video-out CV (LOOV)
    - per-phase 精度 (start / mid / end)
    - permutation feature importance (sklearn)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_lgbm \
        --csv data/training/match_features_phase_e_v01-40.csv \
        --out data/verify/learned_weights_gbm.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

from scripts.old.eda_features import load_dataset  # noqa: E402


# E-3 推奨削減
DROPPED_FEATURES: tuple[str, ...] = (
    "incoming_ojama_pressure",
    "required_puyo_to_fire",
    "offset_power",
    "touching_density",
    "opponent_chain_threat",
)

# Phase 定義 (D-C 相対化後も phase 名は同じ)
PHASE_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "start": ("start_plus_20",),
    "mid": ("mid_minus_20", "midpoint", "mid_plus_20"),
    "end": ("end_minus_5",),
}

# HistGradientBoosting ハイパー grid
GBM_GRID: list[dict] = [
    {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200,
     "min_samples_leaf": 20, "max_leaf_nodes": 15},
    {"max_depth": 6, "learning_rate": 0.05, "max_iter": 300,
     "min_samples_leaf": 20, "max_leaf_nodes": 31},
    {"max_depth": 8, "learning_rate": 0.03, "max_iter": 500,
     "min_samples_leaf": 30, "max_leaf_nodes": 63},
    {"max_depth": None, "learning_rate": 0.05, "max_iter": 400,
     "min_samples_leaf": 20, "max_leaf_nodes": 31},
]


def reduce_dataset(ds, drop):
    keep_idx = [
        i for i, n in enumerate(ds.feature_names) if n not in drop
    ]
    feature_names = tuple(ds.feature_names[i] for i in keep_idx)
    return feature_names, ds.X[:, keep_idx]


def fit_gbm(X_tr, y_tr, X_te, y_te, params: dict):
    """label を ±1 → 0/1 にして学習."""
    y_tr_b = ((np.asarray(y_tr) + 1) // 2).astype(np.int32)
    y_te_b = ((np.asarray(y_te) + 1) // 2).astype(np.int32)
    clf = HistGradientBoostingClassifier(
        random_state=42, **params,
    )
    clf.fit(X_tr, y_tr_b)
    train_acc = float(clf.score(X_tr, y_tr_b))
    test_acc = float(clf.score(X_te, y_te_b))
    return train_acc, test_acc, clf


def video_split(ds, holdout: str):
    test_mask = np.array([v == holdout for v in ds.video_ids])
    return ~test_mask, test_mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path,
        default=_ROOT / "data/training/match_features_phase_e_v01-40.csv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_ROOT / "data/verify/learned_weights_lgbm.json",
    )
    parser.add_argument("--holdout-video", type=str, default="03")
    parser.add_argument("--no-drop", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    print(f"[load] n={len(ds.y)} d={len(ds.feature_names)}")

    if args.no_drop:
        feature_names = ds.feature_names
        X = ds.X
    else:
        feature_names, X = reduce_dataset(ds, DROPPED_FEATURES)
    print(f"[features] kept={len(feature_names)}")

    # 1) video holdout でハイパー grid 探索
    tr_mask, te_mask = video_split(ds, args.holdout_video)
    best_video = (0.0, None, None)
    for params in GBM_GRID:
        tr_acc, te_acc, clf = fit_gbm(
            X[tr_mask], ds.y[tr_mask],
            X[te_mask], ds.y[te_mask], params,
        )
        print(
            f"  lgbm {params}: train={tr_acc:.3f}, "
            f"test={te_acc:.3f}"
        )
        if te_acc > best_video[0]:
            best_video = (te_acc, params, clf)
    print(f"\n[best video_holdout] params={best_video[1]} "
          f"test={best_video[0]:.3f}")
    best_params = best_video[1]

    # 2) leave-one-video-out CV (best_params で)
    print("\n[LOOV] running...")
    loo_accs: dict[str, float] = {}
    for v in sorted(set(ds.video_ids)):
        tr_m, te_m = video_split(ds, v)
        if te_m.sum() < 10:
            continue
        _, te, _ = fit_gbm(
            X[tr_m], ds.y[tr_m], X[te_m], ds.y[te_m], best_params,
        )
        loo_accs[v] = te
    loo_mean = float(np.mean(list(loo_accs.values())))
    loo_std = float(np.std(list(loo_accs.values())))
    print(f"  LOOV: mean={loo_mean:.3f} std={loo_std:.3f}")

    # 3) phase 別 LOOV
    print("\n[phase LOOV]")
    phase_results: dict[str, dict] = {}
    for ph_name, ph_list in PHASE_DEFINITIONS.items():
        ph_idx = np.array([
            i for i, p in enumerate(ds.time_phases) if p in ph_list
        ])
        if len(ph_idx) == 0:
            continue
        Xp = X[ph_idx]
        yp = ds.y[ph_idx]
        vp = [ds.video_ids[i] for i in ph_idx]
        accs = {}
        for v in sorted(set(vp)):
            tr_m_p = np.array([vv != v for vv in vp])
            te_m_p = ~tr_m_p
            if te_m_p.sum() < 5:
                continue
            _, te, _ = fit_gbm(
                Xp[tr_m_p], yp[tr_m_p],
                Xp[te_m_p], yp[te_m_p], best_params,
            )
            accs[v] = te
        m, s = float(np.mean(list(accs.values()))), float(
            np.std(list(accs.values())),
        )
        print(f"  {ph_name:6s}: mean={m:.3f} std={s:.3f}  (n={len(ph_idx)})")
        phase_results[ph_name] = {
            "mean": m, "std": s, "n": int(len(ph_idx)), "per_video": accs,
        }
    overall = float(np.mean(
        [r["mean"] for r in phase_results.values()]
    ))
    print(f"  average: {overall:.3f}")

    # 4) 最終モデル (全データ学習) で permutation feature importance
    y_all_b = ((np.asarray(ds.y) + 1) // 2).astype(np.int32)
    final_clf = HistGradientBoostingClassifier(
        random_state=42, **best_params,
    )
    final_clf.fit(X, y_all_b)
    print("\n[permutation importance] computing...")
    perm = permutation_importance(
        final_clf, X, y_all_b, n_repeats=5,
        random_state=42, n_jobs=-1,
    )
    importance: list[dict] = []
    for i, name in enumerate(feature_names):
        importance.append({
            "name": name,
            "mean": float(perm.importances_mean[i]),
            "std": float(perm.importances_std[i]),
        })
    importance.sort(key=lambda r: -r["mean"])
    print("\n[feature importance (permutation)]")
    for r in importance[:10]:
        print(
            f"  {r['name']:30s} "
            f"mean={r['mean']:+.4f} std={r['std']:.4f}"
        )

    # 5) 出力
    out = {
        "csv": str(args.csv),
        "n_samples": len(ds.y),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "dropped": list(DROPPED_FEATURES) if not args.no_drop else [],
        "best_video_holdout": {
            "params": best_params,
            "test_acc": float(best_video[0]),
        },
        "loov_overall": {
            "mean": loo_mean, "std": loo_std,
            "per_video": loo_accs,
        },
        "phase_results": phase_results,
        "overall_phase_avg": overall,
        "feature_importance": importance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[save] {to_windows_path(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
