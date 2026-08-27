import importlib, sys
from pathlib import Path
sys.path.insert(0, ".")
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")
import numpy as np

v4_dir = Path("data/verify/board_labels_v4F_yardstick_2026-08-17")
for npz_path in sorted(v4_dir.glob("c13_g*.npz")):
    idx = _MC._load_npz_index(npz_path)
    if idx is None:
        continue
    mask = (idx.sides == "2P")
    cand = np.where(mask)[0]
    if len(cand) == 0:
        continue
    label_t = 3121.782958984375
    dt = np.abs(idx.t_secs[cand] - label_t)
    best = int(np.argmin(dt))
    if float(dt[best]) > 0.35:
        continue
    print(npz_path.name, "dt=", float(dt[best]), "frame_idx=", int(idx.frame_idxs[cand[best]]))
    g = idx.grids[cand[best]]
    print(g[5:13])
    # 前後の近傍数枚も見る (何が起きているか)
    order = np.argsort(idx.t_secs[cand])
    ordered = cand[order]
    pos = int(np.where(ordered == cand[best])[0][0])
    lo = max(0, pos-3)
    hi = min(len(ordered), pos+4)
    print("周辺スナップショット (t_sec, おじゃま数):")
    for i in range(lo, hi):
        gi = idx.grids[ordered[i]]
        print(f"  t={idx.t_secs[ordered[i]]:.3f} frame={idx.frame_idxs[ordered[i]]} ojama_count={(gi==9).sum()}")
