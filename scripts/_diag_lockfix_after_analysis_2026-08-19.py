"""lockfix 修正3件の効果測定 (after 詳細、2026-08-19)。

_summarize_lockfix_ab_2026-08-19.py の補完:
1. c109 窓 (1430-1930s) の試合数 + 各試合の won 可用性 (正解=9試合)
2. locked 行の回復量 (before locked 行数 → after locked 行数、同一動画のみ)
3. 退行判定: before 正常動画 (lock% < 10) で lock%/games/miss% が悪化していないか
⚠️ 再DL動画 (c13/c113/c111/c135/c11) は行単位比較から除外 (* 付き参考値)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BEFORE = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
AFTER = ROOT / "data" / "indicators_v2" / "boards_lean_lockfix_2026-08-19"
REDOWNLOADED = {"c13", "c113", "c111", "c135", "c11"}
WIN_LO, WIN_HI = 1430.0, 1930.0


def load(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in ("game_idx", "won", "post_match_lockdown_active",
                              "t_sec")}


def c109_window() -> None:
    p = AFTER / "c109_win1400.npz"
    if not p.exists():
        print("[c109窓] npz なし (未完了)")
        return
    d = load(p)
    m = (d["t_sec"] >= WIN_LO) & (d["t_sec"] <= WIN_HI)
    games = sorted(set(int(g) for g in d["game_idx"][m]))
    print(f"[c109窓 1430-1930s] 試合数={len(games)} (正解9 / 修正前15)")
    for g in games:
        gm = m & (d["game_idx"] == g)
        w = d["won"][gm]
        avail = "won有" if not np.all(np.isnan(w)) else "won欠損"
        t0, t1 = float(d["t_sec"][gm].min()), float(d["t_sec"][gm].max())
        print(f"  game{g}: {t0:7.1f}-{t1:7.1f}s rows={int(gm.sum())} {avail}")


def locked_recovery() -> None:
    print("\n[locked行の回復] (同一動画のみ行単位比較、*=再DLで参考値)")
    tot_b = tot_a = 0
    for f in sorted(AFTER.glob("*.npz")):
        stem = f.stem
        if "_win" in stem:
            continue
        bf = BEFORE / f"{stem}.npz"
        if not bf.exists():
            continue
        db, da = load(bf), load(f)
        lb = int(np.sum(db["post_match_lockdown_active"] == 1))
        la = int(np.sum(da["post_match_lockdown_active"] == 1))
        mark = "*" if stem in REDOWNLOADED else ""
        print(f"  {stem + mark:8} lockedB={lb:5d} lockedA={la:5d} 差={lb - la:+6d} "
              f"(rowsB={len(db['game_idx'])} rowsA={len(da['game_idx'])})")
        if stem not in REDOWNLOADED:
            tot_b += lb
            tot_a += la
    print(f"  合計(再DL除く): lockedB={tot_b} -> lockedA={tot_a} "
          f"(回復候補 {tot_b - tot_a} 行)")


def regression_check() -> None:
    print("\n[退行判定] before lock% < 10 の正常動画:")
    for f in sorted(AFTER.glob("*.npz")):
        stem = f.stem
        if "_win" in stem:
            continue
        bf = BEFORE / f"{stem}.npz"
        if not bf.exists():
            continue
        db, da = load(bf), load(f)
        lb = float(np.mean(db["post_match_lockdown_active"] == 1)) * 100
        if lb >= 10:
            continue
        la = float(np.mean(da["post_match_lockdown_active"] == 1)) * 100
        gb = len(set(int(g) for g in db["game_idx"]))
        ga = len(set(int(g) for g in da["game_idx"]))
        mb = sum(1 for g in set(int(g) for g in db["game_idx"])
                 if np.all(np.isnan(db["won"][db["game_idx"] == g])))
        ma = sum(1 for g in set(int(g) for g in da["game_idx"])
                 if np.all(np.isnan(da["won"][da["game_idx"] == g])))
        verdict = "OK" if (la <= lb + 2 and ga <= gb and ma <= mb) else "要確認"
        mark = "*" if stem in REDOWNLOADED else ""
        print(f"  {stem + mark:8} lock {lb:5.1f}->{la:5.1f}%  games {gb:3d}->{ga:3d}  "
              f"wonMiss {mb:3d}->{ma:3d}  {verdict}")


if __name__ == "__main__":
    c109_window()
    locked_recovery()
    regression_check()
