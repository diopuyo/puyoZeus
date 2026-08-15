# -*- coding: utf-8 -*-
"""同一 landing イベントの直前(着弾前)/直後(着弾後) win を対で比較する追加分析。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
OUT_DIR = Path("data/verify/w12_quantify_2026-08-16")
VISIBLE_ROWS = slice(1, 13)
ON_FIELD_CAP = 72
FORECAST_BUCKETS = [(-0.5, 0.5, "0"), (0.5, 11.5, "1-11"), (11.5, 29.5, "12-29"),
                    (29.5, 71.5, "30-71"), (71.5, 143.5, "72-143"),
                    (143.5, 215.5, "144-215"), (215.5, 1e9, "216+")]


def bucket_of(val, edges):
    for lo, hi, name in edges:
        if lo < val <= hi:
            return name
    return "?"


def load_all_truth_rows():
    files = sorted(NPZ_DIR.glob("*.npz"))
    rows_list = []
    for fp in files:
        d = np.load(fp)
        if "ojama_forecast" not in d.files:
            continue
        grids = d["grids"]
        video_id = d["video_id"]; side = d["side"]
        t_sec = d["t_sec"].astype(np.float64)
        game_idx = d["game_idx"]; won = d["won"].astype(np.float64)
        forecast = d["ojama_forecast"].astype(np.float64)
        tsumo_count = d["tsumo_count"].astype(np.float64)
        vis = grids[:, VISIBLE_ROWS, :]
        board_ojama = (vis == 9).sum(axis=(1, 2)).astype(np.float64)
        df = pd.DataFrame({
            "video_id": video_id, "side": side, "game_idx": game_idx,
            "t_sec": t_sec, "won": won, "forecast": forecast,
            "tsumo_count": tsumo_count, "board_ojama": board_ojama,
        })
        rows_list.append(df)
    return pd.concat(rows_list, ignore_index=True)


def main():
    df = load_all_truth_rows()
    df = df.dropna(subset=["won"]).reset_index(drop=True)
    grp_key = ["video_id", "game_idx", "side"]
    df_sorted = df.sort_values(grp_key + ["t_sec"]).reset_index(drop=True)
    df_sorted["prev_forecast"] = df_sorted.groupby(grp_key)["forecast"].shift(1)
    df_sorted["prev_board_ojama"] = df_sorted.groupby(grp_key)["board_ojama"].shift(1)
    df_sorted["prev_won"] = df_sorted.groupby(grp_key)["won"].shift(1)
    delta_f = df_sorted["prev_forecast"] - df_sorted["forecast"]
    delta_ojama = df_sorted["board_ojama"] - df_sorted["prev_board_ojama"]
    is_landing = (delta_f > 0.5) & (delta_ojama > 0.5)
    landing = df_sorted[is_landing].copy()
    landing["landed_amount"] = delta_f[is_landing]
    landing["landed_bucket"] = landing["landed_amount"].apply(lambda v: bucket_of(v, FORECAST_BUCKETS))
    order = [b[2] for b in FORECAST_BUCKETS]

    paired = landing.groupby("landed_bucket").agg(
        before_win=("prev_won", "mean"), after_win=("won", "mean"), n=("won", "count"),
    ).reindex(order)
    paired.to_csv(OUT_DIR / "paired_before_after_same_event.csv")
    print("=== paired: same landing event, frame just before vs frame of landing ===")
    print(paired)

    # 72+ プール
    mask72 = landing["landed_bucket"].isin(["72-143", "144-215", "216+"])
    sub = landing[mask72]
    print("\n72+ pooled: before=%.4f after=%.4f n=%d" % (
        sub["prev_won"].mean(), sub["won"].mean(), len(sub)))


if __name__ == "__main__":
    main()
