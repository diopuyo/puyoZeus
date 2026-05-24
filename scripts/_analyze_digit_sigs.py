"""検出漏れ調査: summary.tsv から数値変化の実態を分析する。"""
from __future__ import annotations
from pathlib import Path

tsv = Path("data/verify/match_boundaries_v2/video_02/summary.tsv")
lines = tsv.read_text(encoding="utf-8").splitlines()[1:]

prev_L = None
prev_R = None
changes: list[float] = []
for line in lines:
    parts = line.split("\t")
    if len(parts) < 5:
        continue
    t, panel, hL, hR, ev = parts[0], parts[1], parts[2], parts[3], parts[4]
    if panel != "1":
        prev_L = prev_R = None
        continue
    t_sec = float(t)
    if prev_L is not None and (hL != prev_L or hR != prev_R):
        changes.append(t_sec)
    prev_L = hL
    prev_R = hR

print(f"TSV 上の数値 hash 変化点: {len(changes)}")
if changes:
    gaps = []
    for i in range(1, len(changes)):
        gaps.append(changes[i] - changes[i-1])
    gaps.sort()
    print(f"連続変化間 gap min={gaps[0]:.1f}s  max={gaps[-1]:.1f}s  median={gaps[len(gaps)//2]:.1f}s")
    # 最初の 30 変化点を表示
    print(f"変化点 (先頭 30):")
    for t in changes[:30]:
        print(f"  t={t}")
