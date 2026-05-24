"""
段階 2: 特徴量データセット EDA (探索的データ分析)

目的:
    生成された data/training/match_features.csv に対し、
        - 各特徴量と勝者ラベルの Pearson 相関 (符号付き)
        - 時刻フェーズ別の相関 (midpoint / end / start で重要度変化)
        - 特徴量間の相関行列 (冗長指標の検出)
    を計算し、ヒートマップとレポートを出力する。

入力:
    data/training/match_features.csv

出力:
    data/verify/feature_correlation.png   : 特徴量×勝者ラベル相関ヒートマップ
    data/verify/feature_correlation_pairwise.png  : 特徴量間相関ヒートマップ
    data/verify/eda_report.md            : テキストレポート

実行例:
    python -m scripts.eda_features
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.generate_training_dataset import (  # noqa: E402
    DEFAULT_TIME_PHASES,
    FEATURE_NAMES,
)

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features.csv")
DEFAULT_OUTPUT_HEATMAP: Path = Path("data/verify/feature_correlation.png")
DEFAULT_OUTPUT_PAIRWISE: Path = Path(
    "data/verify/feature_correlation_pairwise.png",
)
DEFAULT_OUTPUT_REPORT: Path = Path("data/verify/eda_report.md")

# 特徴量間相関で「冗長」と判断する閾値 (絶対値)
REDUNDANCY_THRESHOLD: float = 0.85


# ============================
# CSV 読み込み
# ============================


@dataclass(frozen=True)
class Dataset:
    """EDA 用データ構造。"""
    feature_names: tuple[str, ...]
    X: np.ndarray            # shape=(n, d)
    y: np.ndarray            # shape=(n,) +1/-1
    video_ids: list[str]
    time_phases: list[str]


def load_dataset(csv_path: Path) -> Dataset:
    """CSV を読み込み Dataset を返す。"""
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError(f"空のデータセット: {csv_path}")
    n = len(rows)
    d = len(FEATURE_NAMES)
    x_arr = np.zeros((n, d), dtype=np.float64)
    y_arr = np.zeros((n,), dtype=np.int64)
    video_ids: list[str] = []
    time_phases: list[str] = []
    for i, r in enumerate(rows):
        for j, name in enumerate(FEATURE_NAMES):
            x_arr[i, j] = float(r.get(name, "0") or 0.0)
        y_arr[i] = int(r["label"])
        video_ids.append(r["video_id"])
        time_phases.append(r["time_phase"])
    return Dataset(
        feature_names=FEATURE_NAMES,
        X=x_arr, y=y_arr,
        video_ids=video_ids,
        time_phases=time_phases,
    )


# ============================
# 相関計算
# ============================


def pearson_with_label(
    X: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """各特徴量と y の Pearson 相関を返す (shape=(d,))。"""
    if X.shape[0] < 2:
        return np.zeros(X.shape[1])
    y_centered = y - y.mean()
    y_std = y.std()
    if y_std == 0:
        return np.zeros(X.shape[1])
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        col_centered = col - col.mean()
        col_std = col.std()
        if col_std == 0:
            out[j] = 0.0
            continue
        out[j] = float(
            (col_centered * y_centered).mean() / (col_std * y_std)
        )
    return out


def pairwise_correlation(X: np.ndarray) -> np.ndarray:
    """特徴量間の相関行列を返す (shape=(d,d))。"""
    if X.shape[0] < 2:
        return np.eye(X.shape[1])
    return np.corrcoef(X.T)


def correlations_per_phase(
    ds: Dataset,
) -> dict[str, np.ndarray]:
    """time_phase 別に特徴量×ラベル相関を計算する。"""
    out: dict[str, np.ndarray] = {}
    for phase in DEFAULT_TIME_PHASES:
        mask = np.array([p == phase for p in ds.time_phases])
        if mask.sum() < 2:
            out[phase] = np.zeros(len(ds.feature_names))
            continue
        out[phase] = pearson_with_label(ds.X[mask], ds.y[mask])
    return out


# ============================
# ヒートマップ描画
# ============================


def _ensure_matplotlib():
    """matplotlib を遅延 import する (テスト時に依存解決を遅らせる)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_label_correlation_heatmap(
    feature_names: Sequence[str],
    overall_corr: np.ndarray,
    phase_corr: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """[全体, 各 phase] × [特徴量] の相関行列ヒートマップを描画する。"""
    plt = _ensure_matplotlib()
    rows_labels = ["overall"] + list(phase_corr.keys())
    matrix = np.vstack(
        [overall_corr] + [phase_corr[p] for p in phase_corr.keys()],
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(feature_names) * 0.6),
                 max(2.5, len(rows_labels) * 0.6)),
    )
    img = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows_labels)))
    ax.set_yticklabels(rows_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:+.2f}",
                ha="center", va="center", fontsize=7,
                color="black",
            )
    ax.set_title("Feature × Label Pearson correlation (per time phase)")
    fig.colorbar(img, ax=ax, fraction=0.025)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pairwise_heatmap(
    feature_names: Sequence[str],
    corr_matrix: np.ndarray,
    out_path: Path,
) -> None:
    """特徴量間相関行列のヒートマップを描画する。"""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(
        figsize=(max(10, len(feature_names) * 0.6),
                 max(10, len(feature_names) * 0.6)),
    )
    img = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(
                j, i, f"{corr_matrix[i, j]:+.2f}",
                ha="center", va="center", fontsize=6,
                color="black",
            )
    ax.set_title("Pairwise feature Pearson correlation")
    fig.colorbar(img, ax=ax, fraction=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================
# 冗長ペア検出
# ============================


def find_redundant_pairs(
    feature_names: Sequence[str],
    corr_matrix: np.ndarray,
    threshold: float = REDUNDANCY_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """|相関| >= threshold な特徴量ペアを返す (i<j のみ)。"""
    out: list[tuple[str, str, float]] = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(corr_matrix[i, j])
            if abs(v) >= threshold:
                out.append((feature_names[i], feature_names[j], v))
    out.sort(key=lambda x: -abs(x[2]))
    return out


# ============================
# レポート生成
# ============================


def write_report(
    ds: Dataset,
    overall_corr: np.ndarray,
    phase_corr: dict[str, np.ndarray],
    redundant_pairs: list[tuple[str, str, float]],
    out_path: Path,
) -> None:
    """Markdown レポートを書き出す。"""
    lines: list[str] = []
    lines.append("# 特徴量 EDA レポート")
    lines.append("")
    lines.append(f"- サンプル数: {len(ds.y)}")
    lines.append(f"- 1P 勝率: {(ds.y == 1).mean():.3f}")
    lines.append(f"- 動画別件数: {_video_breakdown(ds)}")
    lines.append(f"- 時刻別件数: {_phase_breakdown(ds)}")
    lines.append("")
    lines.append("## 各特徴量 × 勝者ラベル Pearson 相関 (overall)")
    lines.append("")
    lines.append("| 順位 | 特徴量 | 相関 |")
    lines.append("|---:|:---|---:|")
    ranking = sorted(
        zip(ds.feature_names, overall_corr),
        key=lambda x: -abs(x[1]),
    )
    for rank, (name, val) in enumerate(ranking, start=1):
        lines.append(f"| {rank} | {name} | {val:+.4f} |")
    lines.append("")
    lines.append("## 時刻フェーズ別 相関 (絶対値 ≥ 0.05 のみ抜粋)")
    for phase, corr in phase_corr.items():
        lines.append(f"### {phase}")
        sig = [
            (n, c) for n, c in zip(ds.feature_names, corr) if abs(c) >= 0.05
        ]
        sig.sort(key=lambda x: -abs(x[1]))
        if not sig:
            lines.append("(有意な相関なし)")
        for n, c in sig:
            lines.append(f"- {n}: {c:+.4f}")
        lines.append("")
    lines.append("## 冗長な特徴量ペア (|r| ≥ "
                 f"{REDUNDANCY_THRESHOLD})")
    if not redundant_pairs:
        lines.append("(検出なし)")
    for a, b, v in redundant_pairs:
        lines.append(f"- {a} ↔ {b}: r = {v:+.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _video_breakdown(ds: Dataset) -> str:
    """動画別件数の文字列表現。"""
    counts: dict[str, int] = {}
    for v in ds.video_ids:
        counts[v] = counts.get(v, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _phase_breakdown(ds: Dataset) -> str:
    """時刻別件数の文字列表現。"""
    counts: dict[str, int] = {}
    for p in ds.time_phases:
        counts[p] = counts.get(p, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description="特徴量 EDA")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--heatmap", type=Path, default=DEFAULT_OUTPUT_HEATMAP,
    )
    parser.add_argument(
        "--pairwise", type=Path, default=DEFAULT_OUTPUT_PAIRWISE,
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_OUTPUT_REPORT,
    )
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    overall = pearson_with_label(ds.X, ds.y)
    phase_corr = correlations_per_phase(ds)
    pair_corr = pairwise_correlation(ds.X)
    redundant = find_redundant_pairs(ds.feature_names, pair_corr)

    plot_label_correlation_heatmap(
        ds.feature_names, overall, phase_corr, args.heatmap,
    )
    plot_pairwise_heatmap(
        ds.feature_names, pair_corr, args.pairwise,
    )
    write_report(ds, overall, phase_corr, redundant, args.report)
    print(f"[save] {args.heatmap}")
    print(f"[save] {args.pairwise}")
    print(f"[save] {args.report}")
    print(f"\n[summary] n={len(ds.y)}, "
          f"overall top3 features (|r|): "
          f"{[(n, round(c, 3)) for n, c in sorted(zip(ds.feature_names, overall), key=lambda x: -abs(x[1]))[:3]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
