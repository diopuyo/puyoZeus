"""Gate 3R-5 の親エージェント自己検収 (使い捨て検証器)。"""
import collections
import itertools

import numpy as np

OFF = "data/verify/gate3r5_off_bitidentical_2026-08-25/"
ON = "data/verify/gate3r5_on_realdata_check_2026-08-25/dump_on.npz"


def _eq(x: np.ndarray, y: np.ndarray) -> bool:
    if x.shape != y.shape:
        return False
    if x.dtype.kind in "fc":
        return np.array_equal(np.nan_to_num(x, nan=-9e99), np.nan_to_num(y, nan=-9e99))
    return np.array_equal(x, y)


r = [np.load(OFF + f"dump_r{i}.npz", allow_pickle=True) for i in (1, 2, 3)]
keys = sorted(r[0].files)
print(f"OFF: キー数={len(keys)} 行数={len(r[0][keys[0]])}")
bad = 0
pairs = 0
for a, b in itertools.combinations(range(3), 2):
    pairs += 1
    for k in keys:
        if not _eq(r[a][k], r[b][k]):
            bad += 1
            print(f"  不一致 r{a+1} vs r{b+1}: {k}")
print(f"OFF 3run: 不一致 {bad}/{len(keys)*pairs} (母数={len(keys)}キー×{pairs}ペア)")
gross_off = [k for k in keys if k.startswith("gross")]
print(f"OFF の gross 列: {len(gross_off)}/{len(keys)} -> {gross_off}")

on = np.load(ON, allow_pickle=True)
onk = sorted(on.files)
og = sorted(k for k in onk if k.startswith("gross"))
print(f"\nON: キー数={len(onk)} 行数={len(on[onk[0]])} / gross列={len(og)}")
for k in og:
    print("   ", k)
shared = [k for k in keys if k in onk]
diff = 0
for k in shared:
    if not _eq(r[0][k], on[k]):
        diff += 1
        print(f"  ON/OFF 不一致: {k}")
print(f"ON vs OFF 共通列: 不一致 {diff}/{len(shared)} (母数={len(shared)}共通キー)")
if "gross_inspected_sides" in onk:
    print("検査母数の分布:", dict(collections.Counter(on["gross_inspected_sides"].tolist())))
for k in ("gross_residual_p1", "gross_residual_p2", "gross_wiped_p1", "gross_wiped_p2"):
    if k in onk:
        v = np.nan_to_num(on[k].astype(float), nan=0.0)
        print(f"{k}: 非0={int((v != 0).sum())}/{len(v)} 合計={v.sum():.0f}")
