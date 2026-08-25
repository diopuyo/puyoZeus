# -*- coding: utf-8 -*-
"""フラグ切り分けA/B結果の比較 (2026-08-19)。

c109 1430-1930s 区間の各構成npzについて:
- game数 / 断片境界数 (直前gameとのギャップ<6s or 開始スコア>=2000)
- won欠損game数
を並べる。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "logs" / "_diag_flag_ablation_2026-08-19"

FRAGMENT_START_SCORE = 2000
FRAGMENT_GAP_SEC = 6.0


def analyze(f: Path) -> str:
    d = dict(np.load(f, allow_pickle=False))
    if len(d["game_idx"]) == 0:
        return f"{f.stem}: (empty)"
    gidxs = sorted(set(int(g) for g in d["game_idx"]))
    n_frag = 0
    n_miss = 0
    detail = []
    prev_last = None
    for g in gidxs:
        rows = np.where(d["game_idx"] == g)[0]
        won = d["won"][rows]
        missing = bool(np.all(np.isnan(won)))
        starts = []
        for side in ("1P", "2P"):
            srows = rows[d["side"][rows] == side]
            if len(srows):
                s = int(d["score"][srows[0]])
                if s >= 0:
                    starts.append(s)
        ss = max(starts) if starts else -1
        t0, t1 = float(d["t_sec"][rows[0]]), float(d["t_sec"][rows[-1]])
        gap = (t0 - prev_last) if prev_last is not None else -1.0
        frag = (ss >= FRAGMENT_START_SCORE) or (0 <= gap < FRAGMENT_GAP_SEC)
        if frag:
            n_frag += 1
        if missing:
            n_miss += 1
        detail.append(
            f"  g{g}: t={t0:.0f}-{t1:.0f} gap={gap:.1f} start_score={ss} "
            f"frag={int(frag)} won={'NaN' if missing else 'ok'}"
        )
        prev_last = t1
    head = (f"{f.stem}: games={len(gidxs)} fragment_games={n_frag} "
            f"won_missing={n_miss}")
    return head + "\n" + "\n".join(detail)


def main() -> None:
    for name in ("A_full", "B_no_lockdown", "C_no_multisignal",
                 "D_no_b3", "E_legacy_boundary"):
        f = DIR / f"{name}.npz"
        print(analyze(f) if f.exists() else f"{name}: (not yet)")
        print()


if __name__ == "__main__":
    main()
