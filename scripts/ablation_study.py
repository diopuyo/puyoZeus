"""
段階 4-3: 1 指標除外 ablation study (限界貢献度分析)

目的:
    LEARNED_V3_GLOBAL ベース構成 (16 指標 と V3 削除 3 指標除外後の 13 指標
    の両方) で、1 指標を 1 つずつ除外して RidgeClassifier (alpha=0.1) を再学習し、
    video_holdout test_acc 落ち幅を計測する。
    「除外したら精度が落ちる指標」 = 重要 / 「除外しても変わらない」 = 冗長。

入力:
    data/training/match_features_v2.csv

出力:
    data/verify/ablation_study.json
    data/verify/ablation_study_chart.png        (16 指標版)
    data/verify/ablation_study_chart_v3.png     (V3 13 指標版)

使用モデル:
    RidgeClassifier(alpha=0.1) + StandardScaler + class_weight='balanced'
    test split: video 03 を holdout
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.eda_features import Dataset, load_dataset  # noqa: E402
from scripts.generate_training_dataset import FEATURE_NAMES  # noqa: E402

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features_v2.csv")
DEFAULT_OUTPUT_JSON: Path = Path("data/verify/ablation_study.json")
DEFAULT_OUTPUT_PNG: Path = Path("data/verify/ablation_study_chart.png")
DEFAULT_OUTPUT_PNG_V3: Path = Path("data/verify/ablation_study_chart_v3.png")
TEST_VIDEO: str = "03"
RIDGE_ALPHA: float = 0.1

# V3 で多重共線性により削除済の 3 指標
V3_DROPPED_FEATURES: tuple[str, ...] = (
    "next_acceptance",
    "offset_power",
    "touching_density",
)

# 「冗長」と判定する精度落ち幅の閾値 (絶対値、ポイント単位)
NEGLIGIBLE_DROP_THRESHOLD: float = 0.005  # 0.5%
# 「重要」と判定する閾値
IMPORTANT_DROP_THRESHOLD: float = 0.015   # 1.5%


# ============================
# データ split
# ============================


def video_holdout_split(ds: Dataset, test_video: str) -> tuple[np.ndarray, np.ndarray]:
    """video_id == test_video を test に。残りを train に。"""
    train_mask = np.array([v != test_video for v in ds.video_ids])
    test_mask = ~train_mask
    return np.where(train_mask)[0], np.where(test_mask)[0]


# ============================
# モデル
# ============================


def fit_ridge_acc(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, float]:
    """RidgeClassifier を学習し train/test 精度を返す。"""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    if X_tr.shape[1] == 0:
        # 全特徴量除外時はクラス比に基づくダミー予測
        majority = 1 if (y_tr == 1).mean() >= 0.5 else -1
        pred_te = np.full_like(y_te, majority)
        return {
            "train_acc": float((y_tr == majority).mean()),
            "test_acc": float((pred_te == y_te).mean()),
        }
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)
    clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
    clf.fit(Xs_tr, y_tr)
    return {
        "train_acc": float(clf.score(Xs_tr, y_tr)),
        "test_acc": float(clf.score(Xs_te, y_te)),
    }


# ============================
# Ablation
# ============================


def run_ablation(
    ds: Dataset, train_idx: np.ndarray, test_idx: np.ndarray,
) -> dict[str, Any]:
    """全特徴量 + 1 指標除外 × N 試行を実行する。"""
    feature_names = list(ds.feature_names)
    X_tr_full = ds.X[train_idx]
    X_te_full = ds.X[test_idx]
    y_tr = ds.y[train_idx]
    y_te = ds.y[test_idx]

    baseline = fit_ridge_acc(X_tr_full, y_tr, X_te_full, y_te)
    base_test = baseline["test_acc"]
    print(f"[baseline] {len(feature_names)} features, test_acc={base_test:.4f}")

    ablation: list[dict[str, Any]] = []
    for i, name in enumerate(feature_names):
        keep_idx = [k for k in range(len(feature_names)) if k != i]
        X_tr = X_tr_full[:, keep_idx]
        X_te = X_te_full[:, keep_idx]
        res = fit_ridge_acc(X_tr, y_tr, X_te, y_te)
        drop = base_test - res["test_acc"]
        ablation.append({
            "removed": name,
            "train_acc": res["train_acc"],
            "test_acc": res["test_acc"],
            "test_acc_drop": float(drop),
        })
        print(f"  ablate {name:30s} test={res['test_acc']:.4f} drop={drop:+.4f}")

    # drop が大きい順 (= 重要) でソート
    ablation.sort(key=lambda r: -r["test_acc_drop"])
    return {
        "baseline": baseline,
        "ablation": ablation,
    }


# ============================
# 分類
# ============================


def classify_contribution(
    ablation: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """落ち幅から「重要 / 寄与小 / 冗長 (= 0 か負)」に分類する。"""
    important: list[str] = []
    minor: list[str] = []
    redundant: list[str] = []
    for entry in ablation:
        drop = entry["test_acc_drop"]
        name = entry["removed"]
        if drop >= IMPORTANT_DROP_THRESHOLD:
            important.append(name)
        elif drop > NEGLIGIBLE_DROP_THRESHOLD:
            minor.append(name)
        else:
            redundant.append(name)
    return {
        "important": important,
        "minor": minor,
        "redundant": redundant,
    }


# ============================
# 描画
# ============================


def _ensure_matplotlib():
    """matplotlib を遅延 import する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_drop_chart(
    ablation: list[dict[str, Any]], baseline_acc: float, out_path: Path,
) -> None:
    """test_acc 落ち幅を棒グラフで描画する。"""
    plt = _ensure_matplotlib()
    names = [r["removed"] for r in ablation]
    drops = [r["test_acc_drop"] for r in ablation]
    colors = ["#d4453a" if d >= IMPORTANT_DROP_THRESHOLD
              else "#fbc02d" if d > NEGLIGIBLE_DROP_THRESHOLD
              else "#9e9e9e" for d in drops]
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, drops, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.axvline(IMPORTANT_DROP_THRESHOLD, color="red",
               linewidth=0.4, linestyle="--", alpha=0.5)
    ax.axvline(NEGLIGIBLE_DROP_THRESHOLD, color="orange",
               linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"test_acc drop (baseline={baseline_acc:.3f})")
    ax.set_title("Ablation Study (1-feature removal, RidgeClassifier α=0.1)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================
# main
# ============================


def reduce_features(
    ds: Dataset, dropped: tuple[str, ...],
) -> Dataset:
    """指定特徴量を除いた Dataset を返す。"""
    keep_idx = [i for i, n in enumerate(ds.feature_names) if n not in dropped]
    return Dataset(
        feature_names=tuple(ds.feature_names[i] for i in keep_idx),
        X=ds.X[:, keep_idx], y=ds.y,
        video_ids=list(ds.video_ids),
        time_phases=list(ds.time_phases),
    )


def run_phase(
    ds: Dataset, train_idx: np.ndarray, test_idx: np.ndarray,
    label: str, png_path: Path,
) -> dict[str, Any]:
    """1 フェーズの ablation 実行 + 描画 + 結果集約。"""
    print(f"\n=== {label} phase: d={len(ds.feature_names)} ===")
    res = run_ablation(ds, train_idx, test_idx)
    classes = classify_contribution(res["ablation"])
    plot_drop_chart(res["ablation"], res["baseline"]["test_acc"], png_path)
    return {
        "label": label,
        "n_features": len(ds.feature_names),
        "feature_names": list(ds.feature_names),
        "baseline_test_acc": res["baseline"]["test_acc"],
        "baseline_train_acc": res["baseline"]["train_acc"],
        "ablation_ranking": res["ablation"],
        "contribution_classes": classes,
        "top5_important": [r["removed"] for r in res["ablation"][:5]],
        "redundant_features": classes["redundant"],
        "chart_png": str(png_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--png-v3", type=Path, default=DEFAULT_OUTPUT_PNG_V3)
    parser.add_argument("--test-video", type=str, default=TEST_VIDEO)
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    print(f"[load] n={len(ds.y)}, d={len(ds.feature_names)}")
    train_idx, test_idx = video_holdout_split(ds, args.test_video)
    print(f"[split] train={len(train_idx)}, test={len(test_idx)}")

    full_phase = run_phase(ds, train_idx, test_idx, "full16", args.png)
    ds_v3 = reduce_features(ds, V3_DROPPED_FEATURES)
    v3_phase = run_phase(ds_v3, train_idx, test_idx, "v3_reduced13", args.png_v3)

    out: dict[str, Any] = {
        "csv_path": str(args.csv),
        "n_samples": int(len(ds.y)),
        "test_video": args.test_video,
        "model": f"RidgeClassifier(alpha={RIDGE_ALPHA}) + StandardScaler + class_weight='balanced'",
        "thresholds": {
            "important_drop": IMPORTANT_DROP_THRESHOLD,
            "negligible_drop": NEGLIGIBLE_DROP_THRESHOLD,
        },
        "v3_dropped_features": list(V3_DROPPED_FEATURES),
        "phases": {"full16": full_phase, "v3_reduced13": v3_phase},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[full16 baseline]    {full_phase['baseline_test_acc']:.4f}")
    print(f"[full16 top5]        {full_phase['top5_important']}")
    print(f"[full16 redundant]   {full_phase['redundant_features']}")
    print(f"\n[v3_reduced baseline] {v3_phase['baseline_test_acc']:.4f}")
    print(f"[v3_reduced top5]     {v3_phase['top5_important']}")
    print(f"[v3_reduced redundant]{v3_phase['redundant_features']}")
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
