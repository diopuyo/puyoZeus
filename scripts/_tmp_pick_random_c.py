# 存在する c* 動画から25本を再現可能ランダムで選定
import glob
import os
import random

paths = sorted(glob.glob("data/frames/video_c*.mp4"))
ids = [os.path.basename(p)[:-4] for p in paths]  # video_cN
print(f"存在するc*動画: {len(ids)}本")

rng = random.Random(20260721)  # 再現可能シード
picked = sorted(rng.sample(ids, 25), key=lambda s: int(s.replace("video_c", "")))
print("選定25本:")
print(" ".join(picked))

# 元のlean収集のsample_interval手掛かり(boards_lean_fixed npzの時間刻み)を確認
import numpy as np
d = np.load("data/indicators_v2/boards_lean_fixed/c1.npz", allow_pickle=False)
ts = d["t_sec"]
if len(ts) > 5:
    diffs = np.diff(np.unique(ts))
    print(f"\nc1 npz スナップ数={len(ts)} t範囲=[{ts.min():.1f},{ts.max():.1f}] "
          f"中央時間刻み={np.median(diffs):.3f}s")
print("npz keys:", list(d.files))
