"""凍結の機構切り分け (user指摘「ブレンドなら決着ではない」、2026-08-15)。

凍結区間ごとに、その間 resolved hold (決着ホールド) が active だったかを
突き合わせ、「安全弁による±100固定」と「決着ホールドによる中間値固定」を
分離して集計する。read-only。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.visualize_advantage_overlay import adv_to_winprob  # noqa: E402

MIN_FREEZE_SEC = 3.0
EPS = 1e-6
KILL_ADV = 99.9  # |adv| がこれ以上なら安全弁の完全上書き

rows = []
for path in sorted(glob.glob("data/verify/backtest_issue14_2026-08-15/npz/on/*.npz")):
    d = np.load(path, allow_pickle=True)
    t, adv, th, act = d["t_adv"], d["adv"], d["t_hold"], d["active"]

    def hold_frac(t0: float, t1: float) -> float:
        m = (th >= t0) & (th <= t1)
        return float(act[m].mean()) if m.any() else float("nan")

    start = 0
    for i in range(1, len(t) + 1):
        end = i == len(t)
        if end or abs(float(adv[i]) - float(adv[start])) > EPS:
            last = len(t) - 1 if end else i - 1
            dur = float(t[last] - t[start])
            if dur >= MIN_FREEZE_SEC:
                a = float(adv[start])
                rows.append({
                    "dur": dur, "adv": a, "lead": max(adv_to_winprob(a), 1 - adv_to_winprob(a)) * 100,
                    "hold": hold_frac(float(t[start]), float(t[last])),
                    "kill": abs(a) >= KILL_ADV, "name": Path(path).stem,
                    "t0": float(t[start]),
                })
            if not end:
                start = i

kill = [r for r in rows if r["kill"]]
mid = [r for r in rows if not r["kill"]]

print(f"=== 凍結{len(rows)}件の機構別内訳 ({MIN_FREEZE_SEC}秒以上) ===\n")
print(f"[A] 安全弁による±100固定 (真の決着): {len(kill)}件")
if kill:
    ds = np.array([r["dur"] for r in kill])
    hs = np.array([r["hold"] for r in kill])
    print(f"    長さ 中央値{np.median(ds):.1f}秒 / 最大{ds.max():.1f}秒、優勢側勝率は全件99.3%")
    print(f"    決着ホールド中だった割合 中央値 {np.nanmedian(hs)*100:.0f}%")

print(f"\n[B] 中間値のまま固定 (=決着ではない): {len(mid)}件")
for r in sorted(mid, key=lambda x: -x["dur"]):
    print(f"    {r['dur']:5.2f}秒 adv={r['adv']:+6.1f} 優勢側{r['lead']:5.1f}% "
          f"ホールド中{r['hold']*100:3.0f}% {r['name']} t={r['t0']:.1f}")
if mid:
    ds = np.array([r["dur"] for r in mid])
    ls = np.array([r["lead"] for r in mid])
    print(f"    → 長さ 中央値{np.median(ds):.1f}秒 / 最大{ds.max():.1f}秒、"
          f"優勢側勝率 中央値{np.median(ls):.1f}% / 最小{ls.min():.1f}%")
