# -*- coding: utf-8 -*-
"""W12 P1 quantify script."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import json

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
OUT_DIR = Path("data/verify/w12_quantify_2026-08-16")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("test ok")

VISIBLE_ROWS = slice(1, 13)
ON_FIELD_CAP = 72

FORECAST_BUCKETS = [(-0.5, 0.5, "0"), (0.5, 11.5, "1-11"), (11.5, 29.5, "12-29"),
                    (29.5, 71.5, "30-71"), (71.5, 143.5, "72-143"),
                    (143.5, 215.5, "144-215"), (215.5, 1e9, "216+")]
CAP_BUCKETS = [(-0.5, 17.5, "0-17"), (17.5, 35.5, "18-35"),
               (35.5, 53.5, "36-53"), (53.5, 71.5, "54-71")]


def bucket_of(val, edges):
    for lo, hi, name in edges:
        if lo < val <= hi:
            return name
    return "?"


def load_all_truth_rows():
    files = sorted(NPZ_DIR.glob("*.npz"))
    rows_list = []
    n_files_truth = 0
    for fp in files:
        d = np.load(fp)
        if "ojama_forecast" not in d.files:
            continue
        n_files_truth += 1
        grids = d["grids"]
        video_id = d["video_id"]
        side = d["side"]
        t_sec = d["t_sec"].astype(np.float64)
        game_idx = d["game_idx"]
        won = d["won"].astype(np.float64)
        forecast = d["ojama_forecast"].astype(np.float64)
        tsumo_count = d["tsumo_count"].astype(np.float64)

        vis = grids[:, VISIBLE_ROWS, :]
        board_ojama = (vis == 9).sum(axis=(1, 2)).astype(np.float64)
        board_color = ((vis >= 1) & (vis <= 5)).sum(axis=(1, 2)).astype(np.float64)
        board_free = ON_FIELD_CAP - (vis != 0).sum(axis=(1, 2)).astype(np.float64)

        df = pd.DataFrame({
            "video_id": video_id, "side": side, "game_idx": game_idx,
            "t_sec": t_sec, "won": won, "forecast": forecast,
            "tsumo_count": tsumo_count,
            "board_ojama": board_ojama, "board_color": board_color,
            "board_free": board_free,
        })
        rows_list.append(df)
    all_df = pd.concat(rows_list, ignore_index=True)
    return all_df, n_files_truth, len(files)


def main():
    df, n_truth_files, n_total_files = load_all_truth_rows()
    print(f"truth npz files: {n_truth_files}/{n_total_files}")
    print(f"total truth rows: {len(df)}")
    df = df.dropna(subset=["won"]).reset_index(drop=True)
    print(f"won-dropna: {len(df)}")

    grp_key = ["video_id", "game_idx", "side"]
    max_tsumo = df.groupby(grp_key)["tsumo_count"].transform("max").replace(0, np.nan)
    df["progress"] = (df["tsumo_count"] / max_tsumo).fillna(0.0).clip(0.0, 1.0)
    df["phase"] = pd.cut(df["progress"], bins=[-0.01, 1/3, 2/3, 1.01],
                          labels=["early", "mid", "late"])

    df["forecast_bucket"] = df["forecast"].apply(lambda v: bucket_of(v, FORECAST_BUCKETS))
    df["cap_bucket"] = df["board_free"].apply(lambda v: bucket_of(v, CAP_BUCKETS))

    order = [b[2] for b in FORECAST_BUCKETS]
    t1 = df.groupby("forecast_bucket")["won"].agg(["mean", "count"]).reindex(order)
    t1.to_csv(OUT_DIR / "1_before_landing_overall.csv")
    print("\n=== 1a overall before-landing by forecast bucket ===")
    print(t1)

    t1p = df.groupby(["phase", "forecast_bucket"], observed=True)["won"].agg(["mean", "count"])
    t1p = t1p.reindex(pd.MultiIndex.from_product(
        [["early", "mid", "late"], order], names=["phase", "forecast_bucket"]))
    t1p.to_csv(OUT_DIR / "1_before_landing_by_phase.csv")
    print("\n=== 1b phase x forecast bucket ===")
    print(t1p)

    df_sorted = df.sort_values(grp_key + ["t_sec"]).reset_index(drop=True)
    df_sorted["prev_forecast"] = df_sorted.groupby(grp_key)["forecast"].shift(1)
    df_sorted["prev_board_ojama"] = df_sorted.groupby(grp_key)["board_ojama"].shift(1)
    delta_f = df_sorted["prev_forecast"] - df_sorted["forecast"]
    delta_ojama = df_sorted["board_ojama"] - df_sorted["prev_board_ojama"]
    is_landing = (delta_f > 0.5) & (delta_ojama > 0.5)
    landing = df_sorted[is_landing].copy()
    landing["landed_amount"] = delta_f[is_landing]
    landing["landed_bucket"] = landing["landed_amount"].apply(lambda v: bucket_of(v, FORECAST_BUCKETS))
    print(f"\nlanding events detected: {len(landing)}")

    t2 = landing.groupby("landed_bucket")["won"].agg(["mean", "count"]).reindex(order)
    t2.to_csv(OUT_DIR / "2_after_landing_overall.csv")
    print("\n=== 2a overall after-landing by landed bucket ===")
    print(t2)

    t2p = landing.groupby(["phase", "landed_bucket"], observed=True)["won"].agg(["mean", "count"])
    t2p = t2p.reindex(pd.MultiIndex.from_product(
        [["early", "mid", "late"], order], names=["phase", "landed_bucket"]))
    t2p.to_csv(OUT_DIR / "2_after_landing_by_phase.csv")
    print("\n=== 2b phase x landed bucket ===")
    print(t2p)

    cap_order = [b[2] for b in CAP_BUCKETS]
    t3 = df.groupby(["forecast_bucket", "cap_bucket"], observed=True)["won"].agg(["mean", "count"])
    t3 = t3.reindex(pd.MultiIndex.from_product([order, cap_order], names=["forecast_bucket", "cap_bucket"]))
    t3.to_csv(OUT_DIR / "3_forecast_x_capacity.csv")
    print("\n=== 3 before-landing forecast x capacity 2D ===")
    print(t3)

    landing["cap_bucket"] = landing["board_free"].apply(lambda v: bucket_of(v, CAP_BUCKETS))
    t3b = landing.groupby(["landed_bucket", "cap_bucket"], observed=True)["won"].agg(["mean", "count"])
    t3b = t3b.reindex(pd.MultiIndex.from_product([order, cap_order], names=["landed_bucket", "cap_bucket"]))
    t3b.to_csv(OUT_DIR / "3b_landed_x_capacity_after.csv")
    print("\n=== 3b after-landing landed x capacity-after 2D ===")
    print(t3b)

    df["color_bucket"] = pd.cut(df["board_color"], bins=[-0.5, 17.5, 35.5, 53.5, 72.5],
                                 labels=["0-17", "18-35", "36-53", "54-72"])
    t5 = df.groupby(["forecast_bucket", "color_bucket"], observed=True)["won"].agg(["mean", "count"])
    t5 = t5.reindex(pd.MultiIndex.from_product(
        [order, ["0-17", "18-35", "36-53", "54-72"]], names=["forecast_bucket", "color_bucket"]))
    t5.to_csv(OUT_DIR / "5_forecast_x_color.csv")
    print("\n=== 5 before-landing forecast x color 2D ===")
    print(t5)

    n_nonzero = (df["forecast"] > 0).sum()
    n_saturated = (df["forecast"] >= ON_FIELD_CAP).sum()
    print(f"\n=== 4 current ojama_forecast raw distribution ===")
    print(f"forecast>0 rows: {n_nonzero} ({100*n_nonzero/len(df):.2f}% of all)")
    print(f"forecast>=72 (saturated) rows: {n_saturated} ({100*n_saturated/len(df):.2f}% of all,"
          f" {100*n_saturated/max(n_nonzero,1):.2f}% of nonzero)")
    desc = df.loc[df["forecast"] > 0, "forecast"].describe(percentiles=[.5, .75, .9, .95, .99])
    print(desc)
    stats_out = {
        "n_rows": int(len(df)), "n_nonzero": int(n_nonzero),
        "n_saturated_ge72": int(n_saturated),
        "pct_nonzero": float(100*n_nonzero/len(df)),
        "pct_saturated_of_all": float(100*n_saturated/len(df)),
        "pct_saturated_of_nonzero": float(100*n_saturated/max(n_nonzero,1)),
        "describe_nonzero": {str(k): float(v) for k, v in desc.items()},
    }
    with open(OUT_DIR / "4_forecast_raw_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)

    df.to_csv(OUT_DIR / "truth_rows_full.csv", index=False)
    print("\ndone. out dir:", OUT_DIR)


if __name__ == "__main__":
    main()
