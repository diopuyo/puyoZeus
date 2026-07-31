# 一時検証: スモーク出力 CSV に XII 新列が正しく入っているか確認
import csv

import numpy as np

XII_COLS = [
    "saturated_chain_count", "ignition_point_count", "multi_color_ignition",
    "sub_chain_count", "simultaneous_pop_richness",
]

rows = list(csv.DictReader(open("data/indicators_v2/_smoke_v29_xii.csv", encoding="utf-8")))
print("rows:", len(rows), " cols:", len(rows[0]))
for c in XII_COLS:
    vals = np.array([float(r[c]) for r in rows])
    raws = np.array([float(r[c + "_raw"]) for r in rows])
    ok = (vals >= 0).all() and (vals <= 1).all() and not np.isnan(vals).any()
    print(f"{c}: range=[{vals.min():.3f},{vals.max():.3f}] mean={vals.mean():.3f} "
          f"raw_max={raws.max():.1f} 0-1ok={ok} nonzero={float((vals > 0).mean()):.1%}")

# npz 対応も確認
d = np.load("data/indicators_v2/_smoke_v29_xii.npz", allow_pickle=False)
print("npz grids:", d["grids"].shape, " csv rows:", len(rows), " 一致:", d["grids"].shape[0] == len(rows))
