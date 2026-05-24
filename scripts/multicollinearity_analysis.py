"""
段階 1: 多重共線性 (Multicollinearity) 分析

目的:
    LEARNED_GLOBAL の重みが counter-intuitive (offset=-0.347、
    touching_density=-2.701 等) であることから、特徴量間の
    冗長性 (multicollinearity) を定量化し、削除/統合候補を提言する。

実施内容:
    1. VIF (Variance Inflation Factor) 計算: 各特徴量について
       VIF_j = 1 / (1 - R^2_j) where R^2_j は他特徴量で j を説明する回帰
    2. 相関行列 + クラスタリング: |r| > 0.8 の特徴量グループ抽出
    3. PCA で実効次元数を測定
    4. 重複っぽい特徴量ペアを特定:
       - main_chain_maturity vs second_chain_potential (連鎖系)
       - touching_density vs key_flexibility (連結系)
       - shape_score vs color_variance (形状系)

出力:
    data/verify/multicollinearity.json
    data/verify/multicollinearity_heatmap.png
    data/verify/multicollinearity.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.eda_features import Dataset, load_dataset  # noqa: E402

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features.csv")
DEFAULT_OUTPUT_JSON: Path = Path("data/verify/multicollinearity.json")
DEFAULT_OUTPUT_HEATMAP: Path = Path(
    "data/verify/multicollinearity_heatmap.png",
)
DEFAULT_OUTPUT_REPORT: Path = Path("data/verify/multicollinearity.md")

# VIF 閾値: 通常 5〜10。10 超は深刻、5 超は要警戒
VIF_THRESHOLD_SEVERE: float = 10.0
VIF_THRESHOLD_WARN: float = 5.0
# クラスタリング閾値: |r| >= 0.8 を同一クラスタとみなす
CORR_CLUSTER_THRESHOLD: float = 0.8
# PCA 主成分の累積寄与率閾値 (実効次元数の判定)
PCA_VAR_THRESHOLD: float = 0.95


# ============================
# VIF 計算
# ============================


def compute_vif(X: np.ndarray) -> np.ndarray:
    """各特徴量の VIF を計算する。

    VIF_j = 1 / (1 - R^2_j) where R^2_j は X[:, j] を残り特徴量で
    線形回帰した決定係数。numpy のみで実装 (statsmodels 不要)。

    Args:
        X: shape=(n, d)。各列が特徴量。

    Returns:
        shape=(d,) の VIF ベクトル。R^2 が 1 に近い (完全多重共線性)
        場合は np.inf。
    """
    n, d = X.shape
    if n < d + 1:
        return np.full(d, np.nan)
    # 標準化 (定数列対策)
    Xs = _standardize(X)
    vif = np.zeros(d, dtype=np.float64)
    for j in range(d):
        mask = np.ones(d, dtype=bool)
        mask[j] = False
        # 線形回帰: Xs[:, j] = Xs[:, mask] @ beta + b
        A = np.hstack([Xs[:, mask], np.ones((n, 1))])
        target = Xs[:, j]
        coef, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((target - pred) ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        if ss_tot < 1e-12:
            vif[j] = float("nan")
            continue
        r2 = 1.0 - ss_res / ss_tot
        if r2 >= 1.0 - 1e-9:
            vif[j] = float("inf")
        else:
            vif[j] = 1.0 / max(1e-9, 1.0 - r2)
    return vif


def _standardize(X: np.ndarray) -> np.ndarray:
    """列毎に z-score 化。分散 0 列はそのままゼロ列にする。"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std_safe


# ============================
# 相関クラスタリング
# ============================


@dataclass
class CorrelationCluster:
    """相関でグルーピングされた特徴量集合。"""
    members: list[str]
    avg_abs_corr: float


def cluster_by_correlation(
    feature_names: list[str],
    corr_matrix: np.ndarray,
    threshold: float = CORR_CLUSTER_THRESHOLD,
) -> list[CorrelationCluster]:
    """|r| >= threshold で連結成分クラスタを抽出する (Union-Find)。"""
    n = len(feature_names)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    clusters: list[CorrelationCluster] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        sub = corr_matrix[np.ix_(members, members)]
        # 対角を除いた絶対相関平均
        mask = ~np.eye(len(members), dtype=bool)
        avg = float(np.abs(sub[mask]).mean()) if mask.any() else 0.0
        clusters.append(CorrelationCluster(
            members=[feature_names[i] for i in members],
            avg_abs_corr=avg,
        ))
    clusters.sort(key=lambda c: -c.avg_abs_corr)
    return clusters


# ============================
# PCA
# ============================


def pca_explained_variance(X: np.ndarray) -> np.ndarray:
    """PCA の主成分寄与率 (固有値正規化済み) を返す。"""
    Xs = _standardize(X)
    cov = np.cov(Xs.T)
    eig_vals, _ = np.linalg.eigh(cov)
    # 降順
    eig_vals = np.sort(eig_vals)[::-1]
    eig_vals = np.clip(eig_vals, 0.0, None)
    total = eig_vals.sum()
    if total <= 0:
        return np.zeros_like(eig_vals)
    return eig_vals / total


def effective_dim(explained: np.ndarray, threshold: float) -> int:
    """累積寄与率が threshold を超える最小の次元数を返す。"""
    cum = np.cumsum(explained)
    over = np.where(cum >= threshold)[0]
    if len(over) == 0:
        return len(explained)
    return int(over[0]) + 1


# ============================
# ヒートマップ描画
# ============================


def plot_correlation_heatmap(
    feature_names: list[str],
    corr_matrix: np.ndarray,
    vif: np.ndarray,
    out_path: Path,
) -> None:
    """相関行列ヒートマップ + VIF を併記して描画する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(feature_names)
    fig, axes = plt.subplots(
        1, 2, figsize=(max(12, n * 0.7), max(10, n * 0.6)),
        gridspec_kw={"width_ratios": [4, 1]},
    )
    ax = axes[0]
    img = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(feature_names, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{corr_matrix[i, j]:+.2f}",
                ha="center", va="center", fontsize=6,
                color="black",
            )
    ax.set_title("Pairwise feature correlation")
    fig.colorbar(img, ax=ax, fraction=0.04)

    # VIF 棒グラフ
    ax2 = axes[1]
    finite_vif = np.where(np.isfinite(vif), vif, 50.0)
    colors = [
        "red" if v >= VIF_THRESHOLD_SEVERE else
        "orange" if v >= VIF_THRESHOLD_WARN else "green"
        for v in vif
    ]
    ax2.barh(range(n), finite_vif, color=colors)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(feature_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.axvline(VIF_THRESHOLD_WARN, color="orange", linestyle="--", lw=0.8)
    ax2.axvline(VIF_THRESHOLD_SEVERE, color="red", linestyle="--", lw=0.8)
    ax2.set_xlabel("VIF (capped at 50)")
    ax2.set_title("Variance Inflation Factor")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================
# レポート生成
# ============================


def write_report(
    feature_names: list[str],
    vif: np.ndarray,
    clusters: list[CorrelationCluster],
    explained: np.ndarray,
    eff_dim: int,
    corr_matrix: np.ndarray,
    out_path: Path,
) -> None:
    """Markdown レポートを書き出す。"""
    lines: list[str] = []
    lines.append("# 多重共線性 (Multicollinearity) 分析レポート")
    lines.append("")
    lines.append(f"- 特徴量数: {len(feature_names)}")
    lines.append(
        f"- PCA 実効次元 ({int(PCA_VAR_THRESHOLD * 100)}% 分散): "
        f"{eff_dim} / {len(feature_names)}",
    )
    lines.append("")
    lines.append("## VIF (Variance Inflation Factor)")
    lines.append("")
    lines.append("| 順位 | 特徴量 | VIF | 判定 |")
    lines.append("|---:|:---|---:|:---|")
    order = np.argsort(-np.where(np.isfinite(vif), vif, 1e9))
    for rank, j in enumerate(order, start=1):
        v = float(vif[j])
        verdict = (
            "深刻 (削除推奨)" if not np.isfinite(v) or v >= VIF_THRESHOLD_SEVERE
            else "要警戒" if v >= VIF_THRESHOLD_WARN else "OK"
        )
        v_str = "inf" if not np.isfinite(v) else f"{v:.2f}"
        lines.append(f"| {rank} | {feature_names[j]} | {v_str} | {verdict} |")
    lines.append("")

    lines.append(f"## 相関クラスタ (|r| >= {CORR_CLUSTER_THRESHOLD})")
    lines.append("")
    if not clusters:
        lines.append(
            f"|r| >= {CORR_CLUSTER_THRESHOLD} のクラスタは検出されなかった。",
        )
    for k, c in enumerate(clusters, start=1):
        lines.append(
            f"### クラスタ {k} (avg|r|={c.avg_abs_corr:.3f}, "
            f"size={len(c.members)})",
        )
        lines.append(", ".join(c.members))
        lines.append("")

    lines.append("## PCA 主成分寄与率 (上位 10)")
    lines.append("")
    lines.append("| 主成分 | 寄与率 | 累積 |")
    lines.append("|---:|---:|---:|")
    cum = np.cumsum(explained)
    for i in range(min(10, len(explained))):
        lines.append(f"| PC{i + 1} | {explained[i]:.4f} | {cum[i]:.4f} |")
    lines.append("")

    lines.append("## 高相関ペア (|r| >= 0.7)")
    lines.append("")
    lines.append("| A | B | r |")
    lines.append("|:---|:---|---:|")
    pairs = _high_corr_pairs(feature_names, corr_matrix, threshold=0.7)
    for a, b, v in pairs[:30]:
        lines.append(f"| {a} | {b} | {v:+.3f} |")
    lines.append("")

    lines.append("## 提言 (削除/統合候補)")
    lines.append("")
    recos = _build_recommendations(feature_names, vif, clusters)
    for r in recos:
        lines.append(f"- {r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _high_corr_pairs(
    feature_names: list[str],
    corr: np.ndarray,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """|r| >= threshold のペアを |r| 降順で返す。"""
    out: list[tuple[str, str, float]] = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(corr[i, j])
            if abs(v) >= threshold:
                out.append((feature_names[i], feature_names[j], v))
    out.sort(key=lambda x: -abs(x[2]))
    return out


def _build_recommendations(
    feature_names: list[str],
    vif: np.ndarray,
    clusters: list[CorrelationCluster],
) -> list[str]:
    """削除/統合候補のテキスト提言を生成する。"""
    out: list[str] = []
    severe_idx = [
        j for j, v in enumerate(vif)
        if not np.isfinite(v) or v >= VIF_THRESHOLD_SEVERE
    ]
    if severe_idx:
        names = [feature_names[j] for j in severe_idx]
        out.append(
            "VIF >= 10 (深刻な共線性): "
            + ", ".join(names)
            + " — 各クラスタの代表 1 個に統合し残りを削除推奨",
        )
    for c in clusters:
        if c.avg_abs_corr >= CORR_CLUSTER_THRESHOLD:
            out.append(
                f"クラスタ ({', '.join(c.members)}) は "
                f"avg|r|={c.avg_abs_corr:.2f} で同一情報を表現。"
                f"代表 1 個に統合推奨。",
            )
    if not out:
        out.append("特に深刻な共線性は検出されなかった。")
    return out


# ============================
# JSON 整形
# ============================


def serialize_report(
    feature_names: list[str],
    vif: np.ndarray,
    corr: np.ndarray,
    clusters: list[CorrelationCluster],
    explained: np.ndarray,
    eff_dim: int,
) -> dict[str, Any]:
    """JSON 出力用の辞書を作る。"""
    return {
        "feature_names": list(feature_names),
        "vif": {
            n: (None if not np.isfinite(v) else float(v))
            for n, v in zip(feature_names, vif)
        },
        "correlation_matrix": corr.tolist(),
        "clusters": [
            {"members": c.members, "avg_abs_corr": c.avg_abs_corr}
            for c in clusters
        ],
        "pca_explained_variance_ratio": [float(x) for x in explained],
        "pca_effective_dim": int(eff_dim),
        "pca_variance_threshold": float(PCA_VAR_THRESHOLD),
        "thresholds": {
            "vif_warn": float(VIF_THRESHOLD_WARN),
            "vif_severe": float(VIF_THRESHOLD_SEVERE),
            "cluster_corr": float(CORR_CLUSTER_THRESHOLD),
        },
    }


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description="多重共線性分析")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--out-heatmap", type=Path, default=DEFAULT_OUTPUT_HEATMAP,
    )
    parser.add_argument(
        "--out-report", type=Path, default=DEFAULT_OUTPUT_REPORT,
    )
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    print(f"[load] n={len(ds.y)}, d={len(ds.feature_names)}")

    vif = compute_vif(ds.X)
    corr = np.corrcoef(ds.X.T)
    clusters = cluster_by_correlation(
        list(ds.feature_names), corr,
    )
    explained = pca_explained_variance(ds.X)
    eff_dim = effective_dim(explained, PCA_VAR_THRESHOLD)

    plot_correlation_heatmap(
        list(ds.feature_names), corr, vif, args.out_heatmap,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            serialize_report(
                list(ds.feature_names), vif, corr, clusters,
                explained, eff_dim,
            ),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        list(ds.feature_names), vif, clusters, explained, eff_dim,
        corr, args.out_report,
    )
    print(f"[save] {args.out_json}")
    print(f"[save] {args.out_heatmap}")
    print(f"[save] {args.out_report}")
    print(f"[summary] VIF top3: ", end="")
    order = np.argsort(-np.where(np.isfinite(vif), vif, 1e9))[:3]
    for j in order:
        v = vif[j]
        v_str = "inf" if not np.isfinite(v) else f"{v:.2f}"
        print(f"{ds.feature_names[j]}={v_str} ", end="")
    print(f"\n[summary] PCA effective dim ({int(PCA_VAR_THRESHOLD * 100)}%): "
          f"{eff_dim}/{len(ds.feature_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
