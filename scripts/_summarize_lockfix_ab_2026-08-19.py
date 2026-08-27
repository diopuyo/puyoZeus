"""ラッチ修正 before/after 集計 (2026-08-19)。

before = data/indicators_v2/boards_lean_subset50_2026-08-19 (修正前収集)
after  = data/indicators_v2/boards_lean_lockfix_2026-08-19 (修正後収集)

per-video: locked行比率 / won欠損率 / 試合数 / 行数。
⚠️ 再DL動画 (c13/c113/c111/c135/c11) は内容ドリフトの可能性
(feedback_redownload_content_drift_2026-08-14) があるため * を付す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BEFORE = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
AFTER = ROOT / "data" / "indicators_v2" / "boards_lean_lockfix_2026-08-19"
REDOWNLOADED = {"c13", "c113", "c111", "c135", "c11"}


def stats(path: Path) -> tuple[int, float, int, int]:
    """(行数, locked行比率, 試合数, won全NaN試合数)。"""
    d = np.load(path, allow_pickle=False)
    n = len(d["game_idx"])
    if n == 0:
        return 0, -1.0, 0, 0
    lock = d["post_match_lockdown_active"]
    lr = float(np.mean(lock == 1))
    games = sorted(set(int(g) for g in d["game_idx"]))
    miss = sum(
        1 for g in games if np.all(np.isnan(d["won"][d["game_idx"] == g]))
    )
    return n, lr, len(games), miss


def main() -> None:
    print(f"{'vid':10} {'rowsB':>6} {'rowsA':>6} {'lockB%':>7} {'lockA%':>7} "
          f"{'gamesB':>6} {'gamesA':>6} {'missB%':>7} {'missA%':>7}")
    tot = dict(nb=0, na=0, lb=0.0, la=0.0, gb=0, ga=0, mb=0, ma=0)
    for f in sorted(AFTER.glob("*.npz")):
        stem = f.stem
        base = stem.split("_win")[0]
        bf = BEFORE / f"{base}.npz"
        nb, lb, gb, mb = stats(bf) if bf.exists() else (0, -1, 0, 0)
        na, la, ga, ma = stats(f)
        mark = "*" if base in REDOWNLOADED else ""
        win = " (窓)" if "_win" in stem else ""
        print(f"{stem + mark:10} {nb:6d} {na:6d} {lb * 100:7.1f} {la * 100:7.1f} "
              f"{gb:6d} {ga:6d} "
              f"{(mb / gb * 100) if gb else -1:7.1f} "
              f"{(ma / ga * 100) if ga else -1:7.1f}{win}")
        if "_win" not in stem and bf.exists():
            tot["nb"] += nb; tot["na"] += na
            tot["lb"] += lb * nb; tot["la"] += la * na
            tot["gb"] += gb; tot["ga"] += ga
            tot["mb"] += mb; tot["ma"] += ma
    if tot["nb"] and tot["na"]:
        print(
            f"{'TOTAL':10} {tot['nb']:6d} {tot['na']:6d} "
            f"{tot['lb'] / tot['nb'] * 100:7.1f} {tot['la'] / tot['na'] * 100:7.1f} "
            f"{tot['gb']:6d} {tot['ga']:6d} "
            f"{tot['mb'] / max(tot['gb'], 1) * 100:7.1f} "
            f"{tot['ma'] / max(tot['ga'], 1) * 100:7.1f}"
        )


if __name__ == "__main__":
    main()
