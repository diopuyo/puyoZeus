# 一時検証: 再収集した v29.csv の XII 列と旧CSVとの整合を確認
import csv

import numpy as np

new = list(csv.DictReader(open("data/indicators_v2/study/v29.csv", encoding="utf-8")))
old = list(csv.DictReader(open("data/indicators_v2/study_backup_2026-07-20/v29.csv", encoding="utf-8")))
print(f"rows new={len(new)} old={len(old)}  cols new={len(new[0])} old={len(old[0])}")

# メタ整合 (t_sec/side が同一順序か)
same_meta = all(
    abs(float(a["t_sec"]) - float(b["t_sec"])) < 0.05 and a["side"] == b["side"]
    for a, b in zip(new, old)
)
print("メタ(t_sec/side)完全一致:", same_meta)

# 既存指標の一致率 (認識の再現性確認、代表3列)
for col in ["board_puyo_total_raw", "current_max_chain_raw"]:
    a = np.array([float(r[col]) for r in new])
    b = np.array([float(r[col]) for r in old])
    n = min(len(a), len(b))
    print(f"{col}: 一致率={float(np.mean(a[:n] == b[:n])):.1%}")

# XII 新列の分布
for col in ["saturated_chain_count", "ignition_point_count", "multi_color_ignition",
            "sub_chain_count", "simultaneous_pop_richness"]:
    v = np.array([float(r[col]) for r in new])
    print(f"{col}: mean={v.mean():.3f} max={v.max():.3f} nonzero={float((v > 0).mean()):.1%}")
raw = np.array([float(r["saturated_chain_count_raw"]) for r in new])
print("saturated_chain_count_raw: max=", raw.max(), " mean=", round(raw.mean(), 2))
