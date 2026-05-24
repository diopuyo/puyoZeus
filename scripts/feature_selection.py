"""
段階 4: 特徴量選別 (Pareto frontier 分析)

目的:
    16 特徴量を 5/8/10/12/16 個に削減して LR L2 を再学習し、
    test_acc が最も高い特徴量数を発見する (overfit 回避)。

選別法:
    1. RF feature_importance ランキングで上位 N 個を選ぶ。
    2. L1 LR の sparsity を強制し、自動的に非零係数を抽出する。
    3. それぞれで video holdout / random split の両方で test_acc 測定。

出力:
    data/verify/feature_selection.json
    data/verify/feature_selection_pareto.png
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.eda_features import Dataset, load_dataset  # noqa: E402
from scripts.learn_weights_v2 import (  # noqa: E402
    DEFAULT_INPUT_CSV,
    fit_lr_eval,
    fit_rf_eval,
    random_split,
    video_level_split,
)

# sklearn 1.8 の deprecation 警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ============================
# 定数
# ============================

DEFAULT_OUTPUT_JSON: Path = Path("data/verify/feature_selection.json")
DEFAULT_OUTPUT_PARETO: Path = Path(
    "data/verify/feature_selection_pareto.png",
)

# 評価する特徴量数の段階
FEATURE_COUNT_GRID: tuple[int, ...] = (3, 5, 8, 10, 12, 16)

# RF importance ランキング作成時のハイパラ
RF_RANK_MAX_DEPTH: int = 8

# L1 LR の C (sparsity 強度)
L1_C_GRID: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0)

# LR L2 評価時の C (固定)
EVAL_LR_C: float = 1.0


# ============================
# 特徴量サブセット切り出し
# ============================


def select_subset(
    ds: Dataset, feature_indices: list[int],
) -> Dataset:
    """指定インデックスの特徴量だけ残した部分データセットを返す。"""
    return Dataset(
        feature_names=tuple(
            ds.feature_names[i] for i in feature_indices
        ),
        X=ds.X[:, feature_indices],
        y=ds.y,
        video_ids=ds.video_ids,
        time_phases=ds.time_phases,
    )


# ============================
# 特徴量ランキング
# ============================


def rank_by_rf_importance(
    ds: Dataset, split,
) -> list[tuple[str, float]]:
    """RF を学習し、feature_importance の降順リストを返す。"""
    res = fit_rf_eval(ds, split, max_depth=RF_RANK_MAX_DEPTH)
    fi = res.feature_importance or {}
    return sorted(fi.items(), key=lambda x: -x[1])


def select_indices_top_n(
    feature_names: tuple[str, ...],
    ranking: list[tuple[str, float]],
    n: int,
) -> list[int]:
    """ranking から上位 N 個の特徴量インデックスを返す。"""
    top = {name for name, _ in ranking[:n]}
    return [
        i for i, name in enumerate(feature_names)
        if name in top
    ]


# ============================
# L1 sparsity 選別
# ============================


@dataclass
class L1SelectionResult:
    """L1 LR で選ばれた特徴量名と係数。"""
    C: float
    selected_features: list[str]
    weights: dict[str, float]
    train_acc: float
    test_acc: float


def select_via_l1(
    ds: Dataset, split, C: float,
) -> L1SelectionResult:
    """L1 LR を学習し、非零係数の特徴量を返す。"""
    res = fit_lr_eval(ds, split, "l1", C)
    selected = [
        name for name, w in res.weights.items() if abs(w) > 1e-6
    ]
    return L1SelectionResult(
        C=C,
        selected_features=selected,
        weights=res.weights,
        train_acc=res.train_acc,
        test_acc=res.test_acc,
    )


# ============================
# 特徴量数 vs 精度 (Pareto frontier)
# ============================


@dataclass
class ParetoPoint:
    """Pareto frontier の 1 点 (特徴量数 vs 精度)。"""
    method: str
    n_features: int
    split_label: str
    train_acc: float
    test_acc: float
    selected: list[str]


def evaluate_subset(
    ds: Dataset, indices: list[int], split, method: str, n: int,
) -> ParetoPoint:
    """指定特徴量サブセットで LR L2 を学習・評価する。"""
    sub_ds = select_subset(ds, indices)
    sub_split = type(split)(
        train_idx=split.train_idx,
        test_idx=split.test_idx,
        label=split.label,
    )
    res = fit_lr_eval(sub_ds, sub_split, "l2", EVAL_LR_C)
    return ParetoPoint(
        method=method,
        n_features=n,
        split_label=split.label,
        train_acc=res.train_acc,
        test_acc=res.test_acc,
        selected=list(sub_ds.feature_names),
    )


def sweep_pareto(
    ds: Dataset, split,
) -> list[ParetoPoint]:
    """RF ランキングと L1 sparsity で Pareto 点を取得する。"""
    points: list[ParetoPoint] = []
    ranking = rank_by_rf_importance(ds, split)
    for n in FEATURE_COUNT_GRID:
        if n > len(ranking):
            continue
        idx = select_indices_top_n(ds.feature_names, ranking, n)
        pt = evaluate_subset(ds, idx, split, "rf_topN", n)
        points.append(pt)
    for C in L1_C_GRID:
        sel = select_via_l1(ds, split, C)
        n = len(sel.selected_features)
        if n == 0:
            continue
        idx = [
            i for i, name in enumerate(ds.feature_names)
            if name in sel.selected_features
        ]
        pt = evaluate_subset(ds, idx, split, f"l1_C{C}", n)
        points.append(pt)
    return points


# ============================
# Pareto frontier 描画
# ============================


def plot_pareto(
    points: list[ParetoPoint], out_path: Path,
) -> None:
    """特徴量数 × test_acc の散布図を描画する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    by_split: dict[str, list[ParetoPoint]] = {}
    for p in points:
        by_split.setdefault(p.split_label, []).append(p)
    for split_label, pts in by_split.items():
        for method in {p.method for p in pts}:
            sub = sorted(
                [p for p in pts if p.method == method],
                key=lambda x: x.n_features,
            )
            xs = [p.n_features for p in sub]
            ys = [p.test_acc for p in sub]
            ax.plot(
                xs, ys, marker="o", linestyle="-",
                label=f"{split_label}/{method}",
                alpha=0.8,
            )
    ax.set_xlabel("number of features")
    ax.set_ylabel("test accuracy")
    ax.set_title(
        "Pareto frontier: feature count vs test accuracy",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================
# 結果集約
# ============================


def best_pareto_point(
    points: list[ParetoPoint], split_label: str | None = None,
) -> ParetoPoint:
    """test_acc 最大の Pareto 点を返す。split_label でフィルタ可能。"""
    candidates = (
        [p for p in points if p.split_label == split_label]
        if split_label else points
    )
    if not candidates:
        candidates = points
    return max(candidates, key=lambda p: p.test_acc)


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="特徴量選別 + Pareto frontier",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_INPUT_CSV,
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT_JSON,
    )
    parser.add_argument(
        "--pareto", type=Path, default=DEFAULT_OUTPUT_PARETO,
    )
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    splits = [
        video_level_split(ds, test_video="03"),
        random_split(ds, train_ratio=0.7),
    ]
    all_points: list[ParetoPoint] = []
    for split in splits:
        print(f"\n[split={split.label}]")
        points = sweep_pareto(ds, split)
        for p in points:
            print(
                f"  method={p.method}, n={p.n_features}, "
                f"train={p.train_acc:.3f}, test={p.test_acc:.3f}"
            )
        all_points.extend(points)

    best_video = best_pareto_point(
        all_points, split_label="video_holdout_03",
    )
    best_random = best_pareto_point(
        all_points, split_label="random_0.70",
    )
    print(
        f"\n[best video_holdout] method={best_video.method}, "
        f"n={best_video.n_features}, test_acc={best_video.test_acc:.3f}"
    )
    print(
        f"[best random] method={best_random.method}, "
        f"n={best_random.n_features}, test_acc={best_random.test_acc:.3f}"
    )
    plot_pareto(all_points, args.pareto)

    out = {
        "n_samples": len(ds.y),
        "feature_names": list(ds.feature_names),
        "feature_count_grid": list(FEATURE_COUNT_GRID),
        "points": [asdict(p) for p in all_points],
        "best_video_holdout": asdict(best_video),
        "best_random": asdict(best_random),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[save] {args.out}")
    print(f"[save] {args.pareto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
