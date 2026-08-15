"""demo2v2_m2 のuser質問2件の裏取り (read-only、ダンプ解析のみ):

1. m2デモ内45秒 (source≈331秒) の表示勝率22%の内訳
   (両者状態・未着弾おじゃま・残り容量・主因top3・生値vs表示値)
2. 「両者連鎖中」以外での勝率凍結の有無を全域スキャン
   (p1が0.2秒刻みで3秒以上不変、かつ両者CHAINでない窓を列挙)
"""
from __future__ import annotations

import numpy as np

DUMP = "data/verify/demo_fixed_2026-08-13/selfverify_demo2v2_fullscan_2026-08-14.npz"
M2_OFFSET = 286.0  # demo2v2_m2 デモ内0秒 = source 286秒 (分割56s + overlay開始230s)
FREEZE_MIN_SEC = 3.0
EPS = 1e-9

d = np.load(DUMP, allow_pickle=True)
t = d["t_sec"]
p1 = d["p1"]
p1_raw = d["p1_raw"]
s1 = d["state1"]
s2 = d["state2"]
pend1 = d["pending_p1"]
pend2 = d["pending_p2"]
room1 = d["room1"]
room2 = d["room2"]
names = d["drivers_top3_names"]
vals = d["drivers_top3_vals"]

# --- 1. m2 45秒付近 (source 326〜338秒) のタイムライン表 ---
print("=== [1] m2デモ内40〜52秒 (source 326〜338秒) タイムライン ===")
print("t_src  t_m2   p1表示  p1生    state1        state2        pend1 pend2 room1 room2  主因top1")
mask = (t >= 326.0) & (t <= 338.0)
idx = np.where(mask)[0]
for i in idx[::2]:  # 0.4秒刻みで間引き表示
    print(
        f"{t[i]:6.1f} {t[i]-M2_OFFSET:5.1f} {p1[i]*100:6.1f}% {p1_raw[i]*100:6.1f}% "
        f"{str(s1[i]):13s} {str(s2[i]):13s} {pend1[i]:5d} {pend2[i]:5d} "
        f"{room1[i]:5d} {room2[i]:5d}  {names[i][0]}={vals[i][0]:+.3f}"
    )

# 22%近傍の詳細 (p1が20〜25%の点の主因top3)
print("\n=== [1b] 該当窓でp1が18〜26%の時点の主因top3 ===")
for i in idx:
    if 0.18 <= p1[i] <= 0.26:
        top3 = ", ".join(f"{names[i][j]}={vals[i][j]:+.3f}" for j in range(3))
        print(f"t_src={t[i]:.1f} (m2 {t[i]-M2_OFFSET:.1f}s) p1={p1[i]*100:.1f}% | {top3}")

# --- 2. 凍結スキャン (全域) ---
print("\n=== [2] 勝率凍結スキャン (p1不変が3秒以上継続する窓) ===")
print("両者CHAIN=正当凍結 (決着ホールド) / それ以外=要調査")
runs = []
start = 0
for i in range(1, len(t)):
    if abs(p1[i] - p1[start]) > EPS:
        if t[i - 1] - t[start] >= FREEZE_MIN_SEC:
            runs.append((start, i - 1))
        start = i
if t[-1] - t[start] >= FREEZE_MIN_SEC:
    runs.append((start, len(t) - 1))

for a, b in runs:
    seg1 = set(str(x) for x in s1[a : b + 1])
    seg2 = set(str(x) for x in s2[a : b + 1])
    both_chain_frac = np.mean(
        [("CHAIN" in str(x)) and ("CHAIN" in str(y)) for x, y in zip(s1[a : b + 1], s2[a : b + 1])]
    )
    one_chain_frac = np.mean(
        [("CHAIN" in str(x)) != ("CHAIN" in str(y)) for x, y in zip(s1[a : b + 1], s2[a : b + 1])]
    )
    no_chain_frac = np.mean(
        [("CHAIN" not in str(x)) and ("CHAIN" not in str(y)) for x, y in zip(s1[a : b + 1], s2[a : b + 1])]
    )
    m2t = t[a] - M2_OFFSET
    label = "正当(両者連鎖)" if both_chain_frac > 0.8 else ("片側連鎖主体" if one_chain_frac > 0.5 else "★連鎖なし主体")
    print(
        f"t_src={t[a]:6.1f}〜{t[b]:6.1f} ({t[b]-t[a]:4.1f}s) p1={p1[a]*100:5.1f}% "
        f"[m2内 {m2t:5.1f}s] 両者連鎖{both_chain_frac*100:3.0f}% 片側{one_chain_frac*100:3.0f}% "
        f"無連鎖{no_chain_frac*100:3.0f}% → {label} | s1={sorted(seg1)} s2={sorted(seg2)}"
    )

# 1P 30%帯が続く場面の特定 (m2範囲 source 286〜342)
print("\n=== [2b] m2範囲でp1が25〜35%に3秒以上滞在する窓 ===")
m2mask = (t >= 286.0) & (t <= 342.0)
in_band = m2mask & (p1 >= 0.25) & (p1 <= 0.35)
start = None
for i in range(len(t)):
    if in_band[i] and start is None:
        start = i
    elif not in_band[i] and start is not None:
        if t[i - 1] - t[start] >= 3.0:
            uniq = len(set(np.round(p1[start:i], 4)))
            print(
                f"t_src={t[start]:.1f}〜{t[i-1]:.1f} ({t[i-1]-t[start]:.1f}s) "
                f"[m2内 {t[start]-M2_OFFSET:.1f}〜{t[i-1]-M2_OFFSET:.1f}s] "
                f"p1範囲={p1[start:i].min()*100:.1f}〜{p1[start:i].max()*100:.1f}% 異なる値の数={uniq}"
            )
        start = None
