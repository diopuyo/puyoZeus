"""火力安定性「比率版」(ratio_kX = expected_fire_kX / near_future_fire_kX) の

win-AUC 軽量検証。

user指示 (2026-07-22, 選択A): 新規指標計算はせず、既存の2つのCSV
(expected_fire_verify_result.csv・near_future_prod_verify_result.csv) を
突合して割り算するだけの軽い検証。src本体には入れない (効けば後で指標化)。

ratio_kX = expected_fire_kX_raw / near_future_fire_kX_raw (K=1..4)。
near_future_fire_kX_raw == 0 (火力ゼロ盤面) の行は ratio=0 とする
(ゼロ割ガード、そのまま除外せず「火力が出ない=安定性も無い」として扱う)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_ratio_stability_verify
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_indicator_win import (  # noqa: E402
    TSUMO_EARLY_RATIO, TSUMO_LATE_RATIO, pair_sides_for_win, build_features,
)

EXPECTED_FIRE_CSV = Path("data/indicators_v2/study/expected_fire_verify_result.csv")
NEAR_FUTURE_CSV = Path("data/indicators_v2/study/near_future_prod_verify_result.csv")
MERGE_KEY: "list[str]" = ["video_id", "side", "game_idx", "t_sec"]
MAX_TDIFF: float = 1.0
K_LEVELS: "tuple[int, ...]" = (1, 2, 3, 4)

N_BOOTSTRAP: int = 500
BOOTSTRAP_SEED: int = 20260722
MIN_BOOTSTRAP_ROWS: int = 20
MIN_BOOTSTRAP_VALID: int = 20


def _load_merged() -> pd.DataFrame:
    """expected_fire (700行サブサンプル) と near_future (1913行全体) を突合し

    ratio_k1..k4 を計算する。突合キーは (video_id, side, game_idx, t_sec)。
    """
    ef = pd.read_csv(EXPECTED_FIRE_CSV)
    nf = pd.read_csv(NEAR_FUTURE_CSV)
    nf_cols = MERGE_KEY + [f"near_future_fire_k{k}_raw" for k in K_LEVELS]
    merged = ef.merge(nf[nf_cols], on=MERGE_KEY, how="inner", suffixes=("", "_nf"))
    for k in K_LEVELS:
        ef_col = f"expected_fire_k{k}_raw"
        nf_col = f"near_future_fire_k{k}_raw"
        ratio = np.where(
            merged[nf_col].values > 0.0,
            merged[ef_col].values / merged[nf_col].values.clip(min=1e-9),
            0.0,
        )
        merged[f"ratio_k{k}"] = ratio
    return merged


def _phase_masks(paired: pd.DataFrame) -> "dict[str, np.ndarray]":
    tsumo = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q33,
        "中盤": (tsumo > q33) & (tsumo <= q67),
        "終盤": tsumo > q67,
    }


def _diff_auc(
    paired: pd.DataFrame, feat_col: str, y: np.ndarray, mask: "np.ndarray | None" = None,
) -> "tuple[float, int]":
    feat = build_features(paired, [feat_col])
    score = feat[f"{feat_col}_diff"].fillna(0.0).values
    yy = y
    if mask is not None:
        score, yy = score[mask], y[mask]
    if len(score) < 20 or len(np.unique(yy)) < 2:
        return float("nan"), len(score)
    auc = float(roc_auc_score(yy, score))
    auc = max(auc, 1.0 - auc)
    return auc, len(score)


def _bootstrap_ci(
    paired: pd.DataFrame, feat_col: str, y: np.ndarray, mask: "np.ndarray | None",
) -> "tuple[float, float]":
    if mask is not None:
        sub = paired[mask].reset_index(drop=True)
        yy = y[mask]
    else:
        sub = paired.reset_index(drop=True)
        yy = y
    videos = sub["video_id_1p"].unique()
    if len(videos) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    aucs: "list[float]" = []
    video_arr = sub["video_id_1p"].values
    for _ in range(N_BOOTSTRAP):
        sampled_videos = rng.choice(videos, size=len(videos), replace=True)
        idx = np.concatenate([np.where(video_arr == v)[0] for v in sampled_videos])
        if len(idx) < MIN_BOOTSTRAP_ROWS:
            continue
        sub_boot = sub.iloc[idx]
        y_boot = yy[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        auc, _ = _diff_auc(sub_boot, feat_col, y_boot)
        if not np.isnan(auc):
            aucs.append(auc)
    if len(aucs) < MIN_BOOTSTRAP_VALID:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    print("=== データ読み込み・突合・ratio計算 ===")
    merged = _load_merged()
    print(f"  expected_fire(700行) x near_future(1913行) 突合行数: {len(merged)}")
    merged = merged.dropna(subset=["video_id", "side", "won"]).copy()
    merged["won"] = merged["won"].astype(int)

    zero_rates = {
        k: float((merged[f"near_future_fire_k{k}_raw"] == 0.0).mean()) for k in K_LEVELS
    }
    print(f"  near_future_fire_kX=0 (ゼロ割ガード発動) の割合: {zero_rates}")

    paired = pair_sides_for_win(merged, MAX_TDIFF)
    n_videos = paired["video_id_1p"].nunique()
    y = paired["won_1p"].astype(int).values
    masks = _phase_masks(paired)
    print(f"  ペアリング後: {len(paired)} 行 (video数: {n_videos})")

    configs = [
        ("current_max_chain", "current_max_chain_raw"),
        ("near_future_fire_k4", "near_future_fire_k4_raw"),
        ("expected_fire_k4", "expected_fire_k4_raw"),
        ("ratio_k1", "ratio_k1"),
        ("ratio_k2", "ratio_k2"),
        ("ratio_k3", "ratio_k3"),
        ("ratio_k4", "ratio_k4"),
    ]

    print()
    print("=" * 100)
    print("  主指標: 単純diff値AUC (point-biserial相当、モデル無し)")
    print("=" * 100)
    header = f"  {'指標':<22}  {'全体':>7}  " + "  ".join(f"{p:>7}" for p in masks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    results: "dict[str, dict[str, tuple[float, int]]]" = {}
    for name, col in configs:
        row: "dict[str, tuple[float, int]]" = {}
        auc_all, n_all = _diff_auc(paired, col, y)
        row["全体"] = (auc_all, n_all)
        line = f"  {name:<22}  {auc_all:>7.4f}  "
        for phase, mask in masks.items():
            auc_p, n_p = _diff_auc(paired, col, y, mask)
            row[phase] = (auc_p, n_p)
            line += f"{auc_p:>7.4f}  "
        print(line)
        results[name] = row

    print()
    print("  --- current_max_chain 比の差分 ---")
    base = results["current_max_chain"]
    for name, _col in configs[1:]:
        deltas = " ".join(
            f"{p}:{results[name][p][0] - base[p][0]:+.4f}" for p in ["全体", *masks]
        )
        print(f"  {name:<22}  {deltas}")

    print()
    print("=" * 100)
    print(f"  ノイズ幅 (video単位クラスタブートストラップ 95%CI, n_boot={N_BOOTSTRAP}, 対象video数={n_videos})")
    print("=" * 100)
    for name, col in configs:
        line = f"  {name:<22}\n"
        for phase, mask in masks.items():
            lo, hi = _bootstrap_ci(paired, col, y, mask)
            point = results[name][phase][0]
            line += f"    {phase}: 点推定={point:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]\n"
        print(line)

    print()
    print("=" * 100)
    print("  独立性チェック (核心): ratio_kX と near_future_fire_kX / expected_fire_kX の相関 (Pearson r)")
    print("=" * 100)
    for k in K_LEVELS:
        nf_diff = build_features(paired, [f"near_future_fire_k{k}_raw"])[
            f"near_future_fire_k{k}_raw_diff"
        ].fillna(0.0).values
        ef_diff = build_features(paired, [f"expected_fire_k{k}_raw"])[
            f"expected_fire_k{k}_raw_diff"
        ].fillna(0.0).values
        ratio_diff = build_features(paired, [f"ratio_k{k}"])[f"ratio_k{k}_diff"].fillna(0.0).values
        r_nf = float(np.corrcoef(nf_diff, ratio_diff)[0, 1])
        r_ef = float(np.corrcoef(ef_diff, ratio_diff)[0, 1])
        print(
            f"  ratio_k{k}: vs near_future_fire_k{k} r={r_nf:.4f}  "
            f"vs expected_fire_k{k} r={r_ef:.4f}",
        )

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
