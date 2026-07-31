# 一時検証: スモーク(60s)とベースライン v29.csv の同区間を比較
import csv

import numpy as np

base = list(csv.DictReader(open("data/indicators_v2/study_backup_2026-07-20/v29.csv", encoding="utf-8")))
base60 = [r for r in base if float(r["t_sec"]) <= 60.0]
print("baseline v29 t<=60s rows:", len(base60))
print("baseline 先頭10行 t_sec:", [round(float(r["t_sec"]), 1) for r in base60[:10]])

smoke = list(csv.DictReader(open("data/indicators_v2/_smoke_v29_xii.csv", encoding="utf-8")))
print("smoke rows:", len(smoke))
for r in smoke:
    print("  t_sec:", r["t_sec"], "side:", r["side"], "puyo_total_raw:", r["board_puyo_total_raw"],
          "cur_max_chain_raw:", r["current_max_chain_raw"], "sat_raw:", r["saturated_chain_count_raw"])

# スモーク npz の盤面を目視
d = np.load("data/indicators_v2/_smoke_v29_xii.npz", allow_pickle=False)
for i in range(d["grids"].shape[0]):
    g = d["grids"][i]
    print(f"--- grid {i} (side={d['side'][i]}, t={d['t_sec'][i]:.1f}) puyo数={int((g > 0).sum())}")
    print(g)
