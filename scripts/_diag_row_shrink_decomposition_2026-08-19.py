"""学習データ行数減少 (319,038→189,545) の内訳分解 診断スクリプト (2026-08-19)。

段階ごとの行数を42本全体で集計する:
  A. 旧npz行数 (boards_lean_phase_l_2026-08-11)
  B. 旧CSV行数 (labeled_win_old.csv、per-video)
  C. 新npz行数 (boards_lean_subset50_2026-08-19)
  D. 新npzのうち match_end_locked==1 / post_match_lockdown_active==1
  E. 新CSV行数 (labeled.csv、per-video) — C-D と一致するか検証
  F. 新CSVのうち won欠損 (学習時に落ちる分)
  G. 旧CSVのうち won欠損 (非対称確認)

出力: logs/diag_row_shrink_decomposition_2026-08-19.tsv + stdout要約
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD_NPZ_DIR = ROOT / "data/indicators_v2/boards_lean_phase_l_2026-08-11"
NEW_NPZ_DIR = ROOT / "data/indicators_v2/boards_lean_subset50_2026-08-19"
OLD_CSV = ROOT / "data/verify/retrain_subset42_2026-08-19/old/labeled_win_old.csv"
NEW_CSV = ROOT / "data/verify/labeled_win_subset42_2026-08-19/labeled.csv"
OUT_TSV = ROOT / "logs/diag_row_shrink_decomposition_2026-08-19.tsv"

TARGET_IDS = [
    "29", "36", "39", "52", "c100", "c101", "c102", "c103", "c104", "c105",
    "c106", "c107", "c108", "c109", "c11", "c110", "c111", "c112", "c113",
    "c114", "c115", "c116", "c117", "c118", "c119", "c125", "c126", "c127",
    "c128", "c129", "c13", "c130", "c131", "c132", "c133", "c134", "c135",
    "c136", "c137", "c96s1", "c96s2", "c96s3",
]


def main() -> int:
    old_csv = pd.read_csv(OLD_CSV, usecols=["video_id", "game_idx", "won", "t_sec", "side"])
    new_csv = pd.read_csv(NEW_CSV, usecols=["video_id", "game_idx", "won", "t_sec", "side"])
    old_by_vid = old_csv.groupby("video_id").size()
    new_by_vid = new_csv.groupby("video_id").size()
    old_won_nan = old_csv[old_csv["won"].isna()].groupby("video_id").size()
    new_won_nan = new_csv[new_csv["won"].isna()].groupby("video_id").size()

    rows = []
    for vid in TARGET_IDS:
        key = f"video_{vid}"
        rec: dict = {"video": vid}
        # 旧npz
        p_old = OLD_NPZ_DIR / f"{vid}.npz"
        if p_old.exists():
            d = np.load(p_old, allow_pickle=True)
            rec["old_npz"] = int(len(d["t_sec"]))
            rec["old_npz_won_nan"] = int(np.isnan(d["won"]).sum())
        else:
            rec["old_npz"] = -1
            rec["old_npz_won_nan"] = -1
        # 新npz
        p_new = NEW_NPZ_DIR / f"{vid}.npz"
        d = np.load(p_new, allow_pickle=True)
        n_new = int(len(d["t_sec"]))
        locked = (d["match_end_locked"] == 1) | (d["post_match_lockdown_active"] == 1)
        rec["new_npz"] = n_new
        rec["new_npz_locked"] = int(locked.sum())
        rec["new_npz_unlocked"] = int((~locked).sum())
        rec["new_npz_won_nan_unlocked"] = int(np.isnan(d["won"][~locked]).sum())
        # CSV
        rec["old_csv"] = int(old_by_vid.get(key, 0))
        rec["new_csv"] = int(new_by_vid.get(key, 0))
        rec["old_csv_won_nan"] = int(old_won_nan.get(key, 0))
        rec["new_csv_won_nan"] = int(new_won_nan.get(key, 0))
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["npz_ratio"] = df["new_npz"] / df["old_npz"]
    df["csv_ratio"] = df["new_csv"] / df["old_csv"]
    df.to_csv(OUT_TSV, sep="\t", index=False)

    t = df.sum(numeric_only=True)
    print("=== 42本全体の段階別行数 ===")
    print(f"A. 旧npz合計          : {int(t['old_npz']):>8,}")
    print(f"B. 旧CSV合計          : {int(t['old_csv']):>8,} (won欠損 {int(t['old_csv_won_nan']):,})")
    print(f"C. 新npz合計          : {int(t['new_npz']):>8,} (旧npz比 {t['new_npz']/t['old_npz']:.1%})")
    print(f"D. うちlocked除外      : {int(t['new_npz_locked']):>8,}")
    print(f"   unlocked           : {int(t['new_npz_unlocked']):>8,}")
    print(f"E. 新CSV合計          : {int(t['new_csv']):>8,} (D除外後と一致? {int(t['new_npz_unlocked'])==int(t['new_csv'])})")
    print(f"F. 新CSV won欠損       : {int(t['new_csv_won_nan']):>8,}")
    print(f"   新npz(unlocked)won欠損: {int(t['new_npz_won_nan_unlocked']):>8,}")
    print(f"G. 学習到達           : {int(t['new_csv'] - t['new_csv_won_nan']):>8,}")
    print()
    print("=== 内訳分解 (旧CSV 319,038 基準) ===")
    base = int(t["old_csv"])
    d_collect = int(t["old_npz"] - t["new_npz"])
    d_locked = int(t["new_npz_locked"])
    d_won = int(t["new_csv_won_nan"])
    reach = int(t["new_csv"] - t["new_csv_won_nan"])
    print(f"旧CSV基準            : {base:,}")
    print(f"旧npz→旧CSVの差      : {int(t['old_npz'])-base:+,} (旧npzと旧CSVの整合確認)")
    print(f"収集方式差 (npzレベル): -{d_collect:,} ({d_collect/base:.1%})")
    print(f"locked除外           : -{d_locked:,} ({d_locked/base:.1%})")
    print(f"won欠損 (学習時drop)  : -{d_won:,} ({d_won/base:.1%})")
    print(f"学習到達             : {reach:,} ({reach/base:.1%})")
    print(f"合計チェック          : {base - d_collect - d_locked - d_won + (int(t['old_npz'])-base):,} (=学習到達と一致するはず)")
    print()
    print("=== per-video (npz_ratio 昇順ワースト10) ===")
    cols = ["video", "old_npz", "new_npz", "npz_ratio", "new_npz_locked",
            "new_csv_won_nan", "old_csv", "new_csv", "csv_ratio"]
    print(df.sort_values("npz_ratio")[cols].head(10).to_string(index=False))
    print()
    print("=== per-video (csv_ratio 昇順ワースト10) ===")
    print(df.sort_values("csv_ratio")[cols].head(10).to_string(index=False))
    print(f"\n出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
