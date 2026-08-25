"""マージンタイム逓減が効く「長い試合」の区間を特定する (2026-08-09).

## なぜ必要か
マージンタイム逓減 (おじゃまレートの時間減衰) は **最初の1手から 95.5 秒**で
始まる。 ところが 1 試合は数十秒〜2 分程度で、 全 148 動画のうち 95.5 秒を
超えた試合は **5.1%** しかない。

先に行った A/B は「各動画の先頭 10 分」を収録して比較したため、 長い試合が
ほとんど含まれず **差がほぼ出なかった** (3 動画中 2 動画で完全一致)。
測る範囲の設計ミスだったので、 **長い試合が実際にある区間**を特定して
そこだけを収録し直す。

## 出し方
boards_lean npz の t_sec / game_idx / side から、 各試合の長さを出し、
95.5 秒を超えた試合の開始・終了時刻を列挙する。 読み取り専用。

出力: data/verify/long_matches_2026-08-09.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.scoring import MARGIN_TIME_START_FROM_FIRST_MOVE_SEC  # noqa: E402

NPZ_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-07"
OUT_TSV = _ROOT / "data" / "verify" / "long_matches_2026-08-09.tsv"
# 収録時に前後へ付ける余裕 (秒)
MARGIN_SEC: float = 5.0
# 上位何件を出すか
TOP_N: int = 20


def main() -> int:
    rows: list[tuple[float, str, int, float, float]] = []
    for p in sorted(NPZ_DIR.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        t = np.asarray(d["t_sec"], dtype=float)
        g = np.asarray(d["game_idx"], dtype=int)
        s = np.asarray(d["side"])
        for gi in np.unique(g):
            m = (g == gi) & (s == "1P")
            if m.sum() < 5:
                continue
            ts = t[m]
            dur = float(ts.max() - ts.min())
            if dur >= MARGIN_TIME_START_FROM_FIRST_MOVE_SEC:
                rows.append((dur, p.stem, int(gi), float(ts.min()), float(ts.max())))
    rows.sort(reverse=True)
    lines = ["video\tgame_idx\tduration_sec\tstart_sec\tend_sec\tclip_start\tclip_end"]
    for dur, vid, gi, st, en in rows:
        lines.append(
            f"{vid}\t{gi}\t{dur:.1f}\t{st:.1f}\t{en:.1f}\t"
            f"{max(0.0, st - MARGIN_SEC):.1f}\t{en + MARGIN_SEC:.1f}"
        )
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"95.5 秒を超えた試合: {len(rows)} 件")
    print(f"\n長い順 上位 {TOP_N}:")
    for dur, vid, gi, st, en in rows[:TOP_N]:
        print(f"  {vid:8s} game={gi:3d} {dur:6.1f}s  ({st:.1f} 〜 {en:.1f})")
    print(f"\n出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
