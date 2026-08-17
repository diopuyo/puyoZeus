# -*- coding: utf-8 -*-
"""
指標大整理提案書 用のエビデンス計算スクリプト (2026-08-12)

data/verify/npz_light_smoke_2026-08-12/labeled_win_light63.csv
(幻盤面ガードON再生成63動画分、won ラベル100%付き) を対象に、
- 単変量AUC (プール全体 / 位相別 / 動画別中央値)
- 列間 Spearman 相関の冗長ペア検出 (|rho|>0.85)
- 欠損率・分散ゼロ列
- center_bulge の 終盤×高充填 層別再現
を計算する。

Windows直接実行ではなく WSL venv 経由で実行すること:
  wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && OMP_NUM_THREADS=1 ./venv/bin/python scripts/_analyze_reorg_evidence_2026-08-12.py"
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CSV_PATH = "data/verify/npz_light_smoke_2026-08-12/labeled_win_light63.csv"
META_COLS = ["video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won"]
CORR_THRESHOLD = 0.85


def compute_progress(df: pd.DataFrame) -> pd.Series:
    """(video_id, game_idx) 単位で t_sec を min-max 正規化した進行率 [0,1] を返す。"""
    grp = df.groupby(["video_id", "game_idx"])["t_sec"]
    t_min = grp.transform("min")
    t_max = grp.transform("max")
    rng = (t_max - t_min).replace(0, np.nan)
    progress = (df["t_sec"] - t_min) / rng
    progress = progress.fillna(0.5)  # 1フレームしかない試合は中間とみなす
    return progress.clip(0.0, 1.0)


def safe_auc(y: np.ndarray, x: np.ndarray) -> float | None:
    """y に両クラスが無い、または x が定数の場合は None。"""
    if len(np.unique(y)) < 2:
        return None
    if np.nanstd(x) == 0:
        return None
    mask = ~np.isnan(x)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y[mask], x[mask]))
    except ValueError:
        return None


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    n_rows = len(df)
    n_videos = df["video_id"].nunique()
    n_games = df[["video_id", "game_idx"]].drop_duplicates().shape[0]
    print(f"[info] rows={n_rows} videos={n_videos} games={n_games}")

    indicator_cols = [c for c in df.columns if c not in META_COLS]
    print(f"[info] indicator_cols({len(indicator_cols)}) = {indicator_cols}")

    df["progress"] = compute_progress(df)
    phase_labels = ["序盤", "中盤", "終盤"]
    df["phase"] = pd.cut(
        df["progress"],
        bins=[-0.001, 1 / 3, 2 / 3, 1.001],
        labels=phase_labels,
    )
    print(df["phase"].value_counts())

    y_all = df["won"].to_numpy()

    rows_out = []
    for col in indicator_cols:
        x_all = df[col].to_numpy(dtype=float)
        missing_pct = float(df[col].isna().mean()) * 100.0
        std_all = float(np.nanstd(x_all))
        zero_var = std_all == 0.0

        raw_auc = safe_auc(y_all, x_all)
        if raw_auc is None:
            direction = "N/A"
            overall_auc = None
        else:
            direction = "+" if raw_auc >= 0.5 else "-"
            overall_auc = raw_auc if raw_auc >= 0.5 else 1.0 - raw_auc

        phase_aucs = {}
        for ph in phase_labels:
            sub = df[df["phase"] == ph]
            a = safe_auc(sub["won"].to_numpy(), sub[col].to_numpy(dtype=float))
            if a is None:
                phase_aucs[ph] = None
            else:
                # overall の向きに合わせて表示 (符号反転していたら "*" を付けて要注意フラグ)
                a_signed = a if direction == "+" else (1.0 - a)
                flipped = (a >= 0.5) != (direction == "+")
                phase_aucs[ph] = (a_signed, flipped)

        # 動画別 AUC の中央値 (向きは overall と揃える)
        video_aucs = []
        for vid, sub in df.groupby("video_id"):
            a = safe_auc(sub["won"].to_numpy(), sub[col].to_numpy(dtype=float))
            if a is None:
                continue
            a_signed = a if direction == "+" else (1.0 - a)
            video_aucs.append(a_signed)
        video_median = float(np.median(video_aucs)) if video_aucs else None
        video_n = len(video_aucs)

        rows_out.append(
            {
                "col": col,
                "direction": direction,
                "overall_auc": overall_auc,
                "早_auc": phase_aucs[phase_labels[0]][0] if phase_aucs[phase_labels[0]] else None,
                "早_flip": phase_aucs[phase_labels[0]][1] if phase_aucs[phase_labels[0]] else None,
                "中_auc": phase_aucs[phase_labels[1]][0] if phase_aucs[phase_labels[1]] else None,
                "中_flip": phase_aucs[phase_labels[1]][1] if phase_aucs[phase_labels[1]] else None,
                "終_auc": phase_aucs[phase_labels[2]][0] if phase_aucs[phase_labels[2]] else None,
                "終_flip": phase_aucs[phase_labels[2]][1] if phase_aucs[phase_labels[2]] else None,
                "video_median_auc": video_median,
                "video_n": video_n,
                "missing_pct": missing_pct,
                "zero_var": zero_var,
                "std": std_all,
            }
        )

    res = pd.DataFrame(rows_out).sort_values("overall_auc", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== 単変量AUC 一覧 (overall降順) ===")
    print(
        res[
            [
                "col",
                "direction",
                "overall_auc",
                "早_auc",
                "中_auc",
                "終_auc",
                "video_median_auc",
                "video_n",
                "missing_pct",
                "zero_var",
            ]
        ].to_string(index=False)
    )

    print("\n=== flip フラグが立った (位相で向きが反転した) 列 ===")
    flip_rows = res[(res["早_flip"] == True) | (res["中_flip"] == True) | (res["終_flip"] == True)]
    if len(flip_rows):
        print(flip_rows[["col", "direction", "早_flip", "中_flip", "終_flip"]].to_string(index=False))
    else:
        print("(なし)")

    # --- Spearman相関の冗長ペア ---
    print(f"\n=== Spearman相関 |rho|>{CORR_THRESHOLD} のペア ===")
    corr = df[indicator_cols].corr(method="spearman")
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > CORR_THRESHOLD:
                pairs.append((cols[i], cols[j], round(float(r), 4)))
    pairs.sort(key=lambda t: -abs(t[2]))
    for a, b, r in pairs:
        print(f"{a} <-> {b} : rho={r}")

    # --- center_bulge 終盤×高充填 層別 ---
    print("\n=== center_bulge 終盤×高充填 層別 ===")
    fill_col = "board_puyo_total"
    high_fill_thresh = df[fill_col].quantile(2 / 3)
    print(f"[info] high_fill threshold (top tertile of {fill_col}) = {high_fill_thresh:.4f}")
    sub = df[(df["phase"] == "終盤") & (df[fill_col] >= high_fill_thresh)]
    print(f"[info] 終盤×高充填 サブセット行数 = {len(sub)}")
    for col in ["center_bulge", "center_bulge_raw"]:
        a = safe_auc(sub["won"].to_numpy(), sub[col].to_numpy(dtype=float))
        print(f"{col}: AUC(終盤×高充填) = {a}")
        # 全体との比較用に overall / 終盤全体も再掲
        a_overall = safe_auc(y_all, df[col].to_numpy(dtype=float))
        end_only = df[df["phase"] == "終盤"]
        a_end = safe_auc(end_only["won"].to_numpy(), end_only[col].to_numpy(dtype=float))
        print(f"  (参考) overall AUC={a_overall}, 終盤全体AUC={a_end}")

    res.to_csv(
        "data/verify/npz_light_smoke_2026-08-12/_reorg_evidence_summary_2026-08-12.csv",
        index=False,
    )
    print(
        "\n[info] 詳細サマリCSV書き出し: "
        "data/verify/npz_light_smoke_2026-08-12/_reorg_evidence_summary_2026-08-12.csv"
    )


if __name__ == "__main__":
    main()
