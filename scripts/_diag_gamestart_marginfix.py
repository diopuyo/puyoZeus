"""各 npz の (video_id, game_idx) 別に開始 t_sec (1P基準) を調べる診断スクリプト。

game_idx==0 は本当にほぼ0点で開始しているか、game_idx>=1 はどれくらい
オフセットがあるかを実データで確認する。
"""
from pathlib import Path
import numpy as np

NPZ_DIR = Path("data/indicators_v2/boards_lean_fixed")

rows = []
for p in sorted(NPZ_DIR.glob("c*.npz"))[:15]:
    with np.load(p, allow_pickle=True) as d:
        sides = d["side"]
        t_secs = d["t_sec"].astype(np.float32)
        game_idxs = d["game_idx"].astype(np.int32)
        vids = d["video_id"]
    mask1p = sides == "1P"
    t1 = t_secs[mask1p]
    g1 = game_idxs[mask1p]
    vid = str(vids[mask1p][0]) if mask1p.any() else "?"
    for gid in np.unique(g1):
        gm = g1 == gid
        if not gm.any():
            continue
        start_t = float(t1[gm].min())
        end_t = float(t1[gm].max())
        rows.append((p.stem, vid, int(gid), start_t, end_t, end_t - start_t))

print(f"{'npz':20s} {'vid':10s} {'gid':4s} {'start_t':>10s} {'end_t':>10s} {'dur':>8s}")
for r in rows:
    print(f"{r[0]:20s} {r[1]:10s} {r[2]:4d} {r[3]:10.1f} {r[4]:10.1f} {r[5]:8.1f}")
