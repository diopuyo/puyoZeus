"""落ちている行の局面偏り 層別診断 (2026-08-19)。

新旧の「学習に到達する行」(新=新CSVのwon非欠損 / 旧=locked窓転写後CSV) を、
各 (video, side, game_idx) 内の相対進行率10分位で層別し、新/旧の行数比を出す。
均等に間引かれているなら全分位で同じ比率、特定局面が丸ごと欠けるなら偏る。

あわせて won欠損の単位 (行単位か試合単位か) と、試合数の新旧差を出す。
出力: logs/diag_row_drop_stratify_2026-08-19.log (tee で使う)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NEW_CSV = ROOT / "data/verify/labeled_win_subset42_2026-08-19/labeled.csv"
OLD_FILT_CSV = ROOT / "data/verify/retrain_subset42_2026-08-19/old_lockedfilt/labeled_win_old_lockedfilt.csv"

USECOLS = ["video_id", "game_idx", "t_sec", "side", "won"]
N_BINS = 10


def _decile_counts(df: pd.DataFrame) -> np.ndarray:
    """各 (video, side, game_idx) 内の t_sec 相対位置で10分位ヒストグラム。"""
    counts = np.zeros(N_BINS, dtype=np.int64)
    for _, g in df.groupby(["video_id", "side", "game_idx"], sort=False):
        t = g["t_sec"].to_numpy(dtype=float)
        if len(t) < 2:
            counts[0] += len(t)
            continue
        lo, hi = t.min(), t.max()
        rel = (t - lo) / max(hi - lo, 1e-9)
        idx = np.minimum((rel * N_BINS).astype(int), N_BINS - 1)
        np.add.at(counts, idx, 1)
    return counts


def main() -> int:
    new = pd.read_csv(NEW_CSV, usecols=USECOLS)
    old = pd.read_csv(OLD_FILT_CSV, usecols=USECOLS)

    # won欠損の単位: 試合単位か
    print("=== won欠損の単位 (新CSV) ===")
    per_game = new.groupby(["video_id", "game_idx", "side"])["won"].agg(
        n="size", nan=lambda s: s.isna().sum())
    all_nan = per_game[per_game["nan"] == per_game["n"]]
    partial = per_game[(per_game["nan"] > 0) & (per_game["nan"] < per_game["n"])]
    print(f"  (video,game,side)単位: 全欠損 {len(all_nan)}組 ({all_nan['n'].sum():,}行) / "
          f"部分欠損 {len(partial)}組 ({int(partial['nan'].sum()):,}行)")
    n_games_new = new.groupby(["video_id", "game_idx", "side"]).ngroups
    n_games_new_labeled = new.dropna(subset=["won"]).groupby(
        ["video_id", "game_idx", "side"]).ngroups
    n_games_old = old.groupby(["video_id", "game_idx", "side"]).ngroups
    print(f"  試合×side数: 新={n_games_new} (won有り{n_games_new_labeled}) / 旧(転写後)={n_games_old}")

    # 学習到達行のみで層別
    new_l = new.dropna(subset=["won"])
    old_l = old.dropna(subset=["won"])
    print(f"\n=== 相対進行率10分位の行数 (学習到達行) ===")
    cn = _decile_counts(new_l)
    co = _decile_counts(old_l)
    print(f"{'分位':>4} {'旧(転写後)':>10} {'新':>10} {'新/旧':>7}")
    for i in range(N_BINS):
        print(f"D{i+1:>3} {co[i]:>10,} {cn[i]:>10,} {cn[i]/max(co[i],1):>7.3f}")
    print(f"合計 {co.sum():>10,} {cn.sum():>10,} {cn.sum()/co.sum():>7.3f}")

    # per-video 比率の分布 (どの動画が欠けるか)
    print("\n=== per-video 学習到達行 新/旧比 ワースト8 ===")
    r = (new_l.groupby("video_id").size() / old_l.groupby("video_id").size()).sort_values()
    print(r.head(8).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
