"""
段階 4-2: sub_chain_quality と sub_chain_independence の重複度測定

目的:
    - 1390 サンプル上で 2 つの「副砲」系指標の Pearson 相関を測定
    - 線形モデル R^2 で「片方が他方の関数として表現できるか」を評価
    - 散布図 PNG を作成し、統合 / 削除推奨を JSON で出力

入力:
    data/training/match_features_v2.csv

出力:
    data/verify/diagnose_sub_chain.json
    data/verify/diagnose_sub_chain_scatter.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features_v2.csv")
DEFAULT_OUTPUT_JSON: Path = Path("data/verify/diagnose_sub_chain.json")
DEFAULT_OUTPUT_PNG: Path = Path("data/verify/diagnose_sub_chain_scatter.png")
COL_QUALITY: str = "sub_chain_quality"
COL_INDEP: str = "sub_chain_independence"

# 統合判定閾値: |r| >= 0.85 なら強い重複、>= 0.5 なら中程度
HIGH_REDUNDANCY_THRESHOLD: float = 0.85
MODERATE_REDUNDANCY_THRESHOLD: float = 0.50


# ============================
# CSV 読み込み
# ============================


def load_two_columns(
    csv_path: Path, col_a: str, col_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """2 列を float ndarray で取得する。"""
    a: list[float] = []
    b: list[float] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            a.append(float(r.get(col_a, "0") or 0.0))
            b.append(float(r.get(col_b, "0") or 0.0))
    return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


# ============================
# 統計
# ============================


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相関係数を返す。"""
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = x.std()
    sy = y.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """y = a*x + b で最小二乗フィットし、(a, b, R^2) を返す。"""
    if x.size < 2 or x.std() == 0:
        return 0.0, float(y.mean() if y.size else 0.0), 0.0
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)


def col_stats(values: np.ndarray) -> dict[str, float]:
    """1 列の基本統計。"""
    return {
        "n": int(values.size),
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "abs_mean": float(np.abs(values).mean()) if values.size else 0.0,
    }


# ============================
# scatter PNG 描画
# ============================


def _ensure_matplotlib():
    """matplotlib を遅延 import する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_scatter(
    x: np.ndarray, y: np.ndarray,
    a: float, b: float, r2: float,
    out_path: Path,
) -> None:
    """sub_chain_quality vs sub_chain_independence の散布図を保存。"""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=8, alpha=0.4, color="#3477eb")
    if x.size >= 2:
        xs = np.linspace(float(x.min()), float(x.max()), 50)
        ax.plot(xs, a * xs + b, color="red", linewidth=1.5,
                label=f"y={a:.3f}x+{b:.3f}, R^2={r2:.3f}")
    ax.set_xlabel(COL_QUALITY)
    ax.set_ylabel(COL_INDEP)
    ax.set_title("sub_chain_quality vs sub_chain_independence")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================
# 判定
# ============================


def decide_recommendation(
    r: float, r2: float,
    quality_stats: dict[str, float],
    indep_stats: dict[str, float],
) -> dict[str, Any]:
    """相関と分散から「両方残す / 一方削除 / 統合」を判定する。"""
    abs_r = abs(r)
    if abs_r >= HIGH_REDUNDANCY_THRESHOLD:
        decision = "DROP_ONE"
        target = COL_INDEP if quality_stats["abs_mean"] >= indep_stats["abs_mean"] else COL_QUALITY
        rationale = (
            f"|r|={abs_r:.3f} >= {HIGH_REDUNDANCY_THRESHOLD} で高重複。"
            f"より情報量が小さい側 ({target}) を削除推奨。"
        )
    elif abs_r >= MODERATE_REDUNDANCY_THRESHOLD:
        decision = "REVIEW"
        target = None
        rationale = (
            f"|r|={abs_r:.3f} は中程度。ablation で精度寄与を比較し決定。"
        )
    else:
        decision = "KEEP_BOTH"
        target = None
        rationale = (
            f"|r|={abs_r:.3f} は弱い相関。両方独立な情報を持つため保持推奨。"
        )
    return {
        "decision": decision,
        "drop_target": target,
        "rationale": rationale,
        "abs_pearson": abs_r,
        "linear_r2": r2,
    }


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description="sub_chain 系の重複測定")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--png", type=Path, default=DEFAULT_OUTPUT_PNG)
    args = parser.parse_args()

    quality, indep = load_two_columns(args.csv, COL_QUALITY, COL_INDEP)
    q_stats = col_stats(quality)
    i_stats = col_stats(indep)
    r = pearson_corr(quality, indep)
    a, b, r2 = linear_fit_r2(quality, indep)
    plot_scatter(quality, indep, a, b, r2, args.png)
    decision = decide_recommendation(r, r2, q_stats, i_stats)

    out: dict[str, Any] = {
        "csv_path": str(args.csv),
        "n_samples": int(quality.size),
        "sub_chain_quality_stats": q_stats,
        "sub_chain_independence_stats": i_stats,
        "pearson_r": r,
        "linear_fit": {"a": a, "b": b, "r2": r2},
        "recommendation": decision,
        "scatter_png": str(args.png),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[stats] r={r:+.3f} R^2={r2:.3f} decision={decision['decision']}")
    print(f"[save] {args.out}")
    print(f"[save] {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
