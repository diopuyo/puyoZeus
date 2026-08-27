from __future__ import annotations
import csv
import numpy as np
from pathlib import Path

rows = []
with open("data/verify/recognition_diag_chain_anim_duration_multi/events_raw.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["status"] == "ok" and r["visual_duration_sec"] != "":
            rows.append((int(r["chain_count"]), float(r["visual_duration_sec"])))
with open("data/verify/chain_anim_duration_2026-08-14/new_events_raw.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["status"] == "ok" and r["visual_duration_sec"] != "":
            rows.append((int(r["chain_count"]), float(r["visual_duration_sec"])))

x = np.array([r[0] for r in rows], dtype=float)
y = np.array([r[1] for r in rows], dtype=float)
print("n_total=", len(x))
b, a = np.polyfit(x, y, 1)
pred = a + b * x
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
print(f"linear: a={a:.3f} b={b:.3f} r2={r2:.4f}")

b0 = np.sum(x * y) / np.sum(x * x)
pred0 = b0 * x
r2_0 = 1 - np.sum((y - pred0) ** 2) / np.sum((y - np.mean(y)) ** 2)
print(f"origin: b={b0:.3f} r2={r2_0:.4f}")

# P75較正 (安全側=長め予算)
for n in range(1, 16):
    mask = x == n
    if mask.sum() == 0:
        continue
    yy = y[mask]
    print(f"n={n:2d} count={mask.sum():3d} median={np.median(yy):.2f} p75={np.percentile(yy,75):.2f} cv={np.std(yy)/np.mean(yy):.2f}")
