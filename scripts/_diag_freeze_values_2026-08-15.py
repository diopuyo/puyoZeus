"""凍結中の表示値の実測 (user質問「決着というからには勝率99%以上か?」、2026-08-15)。

バックテストON構成のnpzから「adv が変化しない連続区間 (凍結)」を検出し、
その間の表示勝率がどれだけ極端だったかを長さ別に集計する。read-only。
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

rows: list[tuple[float, float, float, str]] = []  # (長さ, adv, p1%, 出所)
for path in sorted(glob.glob("data/verify/backtest_issue14_2026-08-15/npz/on/*.npz")):
    d = np.load(path, allow_pickle=True)
    t, adv = d["t_adv"], d["adv"]
    if len(t) < 2:
        continue
    start = 0
    for i in range(1, len(t)):
        if abs(float(adv[i]) - float(adv[start])) > EPS:
            dur = float(t[i - 1] - t[start])
            if dur >= MIN_FREEZE_SEC:
                a = float(adv[start])
                rows.append((dur, a, adv_to_winprob(a) * 100.0, Path(path).stem))
            start = i
    dur = float(t[-1] - t[start])
    if dur >= MIN_FREEZE_SEC:
        a = float(adv[start])
        rows.append((dur, a, adv_to_winprob(a) * 100.0, Path(path).stem))

if not rows:
    print("凍結区間なし")
    raise SystemExit(0)

rows.sort(key=lambda r: -r[0])
print(f"=== ON構成: {MIN_FREEZE_SEC}秒以上の凍結区間 {len(rows)}件 ===")
print("長さ(秒)  adv      1P勝率   優勢側の勝率  動画")
for dur, a, p1, name in rows[:25]:
    lead = max(p1, 100.0 - p1)
    print(f"{dur:7.2f}  {a:+7.1f}  {p1:6.1f}%  {lead:6.1f}%       {name}")

leads = np.array([max(p, 100.0 - p) for _, _, p, _ in rows])
durs = np.array([d for d, _, _, _ in rows])
print()
print("=== 凍結中の「優勢側の勝率」分布 ===")
print(f"中央値 {np.median(leads):.1f}% / 最小 {leads.min():.1f}% / 最大 {leads.max():.1f}%")
for th in (99.0, 95.0, 90.0, 80.0):
    n = int((leads >= th).sum())
    sec = float(durs[leads >= th].sum())
    print(f"  {th:.0f}%以上: {n:3d}件 ({n/len(leads)*100:5.1f}%) 合計{sec:7.1f}秒")
print()
print("=== 長い凍結ほど極端か (長さ別) ===")
for lo, hi in ((3, 5), (5, 10), (10, 100)):
    m = (durs >= lo) & (durs < hi)
    if m.any():
        print(f"  {lo:2d}-{hi:3d}秒: {int(m.sum()):3d}件 優勢側勝率 中央値{np.median(leads[m]):.1f}% "
              f"最小{leads[m].min():.1f}%")
