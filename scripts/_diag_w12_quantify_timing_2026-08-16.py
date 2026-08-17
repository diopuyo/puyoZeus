# -*- coding: utf-8 -*-
"""項目3: 猶予(予告が現れてから着弾するまでの経過時間)と勝率の関係。
プロキシ定義: 「forecastが直近0だった時点」から「着弾フレーム」までの経過秒数
を猶予とみなす(相手のアニメ残り時間そのものではないが、観測可能な待ち時間の
代理指標)。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
OUT_DIR = Path("data/verify/w12_quantify_2026-08-16")
VISIBLE_ROWS = slice(1, 13)
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
        vis = grids[:, VISIBLE_ROWS, :]
        board_ojama = (vis == 9).sum(axis=(1, 2)).astype(np.float64)
        df = pd.DataFrame({
            "video_id": video_id, "side": side, "game_idx": game_idx,
            "t_sec": t_sec, "won": won, "forecast": forecast,
            "board_ojama": board_ojama,
        })
        rows_list.append(df)
    return pd.concat(rows_list, ignore_index=True)


def main():
    df = load_all_truth_rows()
    df = df.dropna(subset=["won"]).reset_index(drop=True)
    grp_key = ["video_id", "game_idx", "side"]
    df_sorted = df.sort_values(grp_key + ["t_sec"]).reset_index(drop=True)

    is_zero = (df_sorted["forecast"] <= 0.0)
    run_id = is_zero.groupby([df_sorted[k] for k in grp_key]).cumsum()
    df_sorted["run_id"] = run_id
    episode_key = grp_key + ["run_id"]
    ep_start = df_sorted[~is_zero].groupby(episode_key)["t_sec"].transform("min")
    df_sorted.loc[~is_zero, "episode_start_t"] = ep_start

    df_sorted["prev_forecast"] = df_sorted.groupby(grp_key)["forecast"].shift(1)
    df_sorted["prev_board_ojama"] = df_sorted.groupby(grp_key)["board_ojama"].shift(1)
    delta_f = df_sorted["prev_forecast"] - df_sorted["forecast"]
    delta_ojama = df_sorted["board_ojama"] - df_sorted["prev_board_ojama"]
    is_landing = (delta_f > 0.5) & (delta_ojama > 0.5)
    landing = df_sorted[is_landing].copy()
    landing["landed_amount"] = delta_f[is_landing]
    landing["landed_bucket"] = landing["landed_amount"].apply(lambda v: bucket_of(v, FORECAST_BUCKETS))

    # 猶予 = 着弾フレームのt_sec - (着弾直前の非ゼロ区間の開始t_sec)
    # episode_start_t は着弾直前行(prev row)基準にすべきだが、簡便に着弾行自身の
    # 直前run(=着弾前は非ゼロだったはずなので)開始を使う。着弾行自体はforecastが
    # 下がった行なので、prev行のepisode_start_tを引く。
    prev_ep_start = df_sorted.groupby(grp_key)["episode_start_t"].shift(1)
    landing["grace_sec"] = landing["t_sec"] - prev_ep_start[is_landing]

    order = [b[2] for b in FORECAST_BUCKETS]
    print("=== grace_sec (猶予) 分布 (着弾イベントのみ) ===")
    print(landing["grace_sec"].describe(percentiles=[.25, .5, .75, .9]))

    grace_bins = [(-0.01, 3.0, "0-3s"), (3.0, 6.0, "3-6s"), (6.0, 12.0, "6-12s"),
                  (12.0, 1e9, "12s+")]
    landing["grace_bucket"] = landing["grace_sec"].apply(lambda v: bucket_of(v, grace_bins) if pd.notna(v) else "unknown")

    t = landing.groupby(["grace_bucket"], observed=True)["won"].agg(["mean", "count"])
    print("\n=== 猶予バケツ別 勝率 (全landed) ===")
    print(t)
    t.to_csv(OUT_DIR / "3_timing_grace_overall.csv")

    t72 = landing[landing["landed_bucket"].isin(["72-143", "144-215", "216+"])]
    t2 = t72.groupby(["grace_bucket"], observed=True)["won"].agg(["mean", "count"])
    print("\n=== 猶予バケツ別 勝率 (着弾72+のみ) ===")
    print(t2)
    t2.to_csv(OUT_DIR / "3_timing_grace_72plus.csv")


if __name__ == "__main__":
    main()
