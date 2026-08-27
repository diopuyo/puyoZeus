"""lockfix A/B の before 側 (subset50) 基準値を先出しする診断 (2026-08-19)。

対象: 再収集15本 (r2 12本 + c11 + 39 + c109)。
出力: 行数 / locked行比率 / 試合数 / won全NaN試合数 / c109は1430-1930s窓の試合数も。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BEFORE = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
STEMS = ["29", "31", "32", "33", "34", "35", "37", "39",
         "c11", "c13", "c109", "c111", "c113", "c132", "c135"]


def main() -> None:
    print(f"{'vid':8} {'rows':>7} {'lock%':>6} {'games':>6} {'wonMiss':>7} {'miss%':>6}")
    for s in STEMS:
        p = BEFORE / f"{s}.npz"
        if not p.exists():
            print(f"{s:8} (before なし)")
            continue
        d = np.load(p, allow_pickle=False)
        n = len(d["game_idx"])
        lock = float(np.mean(d["post_match_lockdown_active"] == 1)) * 100
        games = sorted(set(int(g) for g in d["game_idx"]))
        miss = sum(1 for g in games
                   if np.all(np.isnan(d["won"][d["game_idx"] == g])))
        print(f"{s:8} {n:7d} {lock:6.1f} {len(games):6d} {miss:7d} "
              f"{miss / len(games) * 100 if games else 0:6.1f}")
        if s == "c109":
            t = d["t_sec"] if "t_sec" in d.files else d["time_sec"]
            m = (t >= 1430) & (t <= 1930)
            gw = sorted(set(int(g) for g in d["game_idx"][m]))
            print(f"  c109窓1430-1930s: 試合数={len(gw)} game_idx={gw}")


if __name__ == "__main__":
    main()
