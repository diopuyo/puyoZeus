import importlib, sys
from pathlib import Path
sys.path.insert(0, ".")
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")
YS = importlib.import_module("scripts._measure_yardstick_v4_2026-08-05")

rows = YS.load_yardstick_rows()
row = [r for r in rows if r.video_stem == "c13" and r.side == "2P" and r.frame_idx == 187307][0]
rec = YS._reconstruct_correct_grid(row, YS.BASELINE_NPZ_DIR)
correct, label_t = rec
print("label_t =", label_t)

v4_dir = Path("data/verify/board_labels_v4F_yardstick_2026-08-17")
for npz_path in sorted(v4_dir.glob(f"{row.video_stem}_g*.npz")):
    idx = _MC._load_npz_index(npz_path)
    if idx is None:
        continue
    match = _MC._find_by_frame_idx_exact(idx, row.side, row.frame_idx)
    print(npz_path.name, "exact_match=", match is not None)
    near = YS._find_nearest_in_time(idx, row.side, label_t)
    if near is not None:
        # 見つけたグリッドのidx内でのt_secとframe_idxも出したい
        import numpy as np
        mask = (idx.sides == row.side)
        cand = np.where(mask)[0]
        dt = np.abs(idx.t_secs[cand] - label_t)
        best = int(np.argmin(dt))
        print("  nearest dt=", float(dt[best]), "t_sec=", float(idx.t_secs[cand[best]]))
        if hasattr(idx, "frame_idxs"):
            print("  nearest frame_idx=", idx.frame_idxs[cand[best]])
