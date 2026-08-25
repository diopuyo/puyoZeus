import pandas as pd
import numpy as np

df = pd.read_csv("data/verify/judgment_scan_zenchi_2026-08-22/suspects.tsv", sep="\t")
# 解放判定(release blocker)のみ = stage in (display, both)
blockers = df[df["stage"].isin(["display","both"])].copy()
print("release-blocker件数:", len(blockers))
blockers = blockers.sort_values("t_sec")

# エピソード化: 同一 detector で連続する suspect の間隔が2秒以内なら同一エピソードにまとめる
def episodes(sub):
    sub = sub.sort_values("t_sec")
    eps = []
    cur_start = None
    cur_end = None
    cur_rows = []
    for _, row in sub.iterrows():
        t = row["t_sec"]
        if cur_start is None:
            cur_start = t; cur_end = t; cur_rows=[row]
        elif t - cur_end <= 2.0:
            cur_end = t; cur_rows.append(row)
        else:
            eps.append((cur_start, cur_end, len(cur_rows)))
            cur_start = t; cur_end = t; cur_rows=[row]
    if cur_start is not None:
        eps.append((cur_start, cur_end, len(cur_rows)))
    return eps

for det in ["D1a", "D1b"]:
    sub = blockers[blockers["detector"]==det]
    eps = episodes(sub)
    print(f"\n=== {det} エピソード数(連続<=2s間隔でまとめた): {len(eps)} (元suspect行数={len(sub)}) ===")
    for s,e,n in eps:
        print(f"  t={s:.2f}~{e:.2f} (継続{e-s:.2f}s, {n}行)")

# セグメント境界(既知の8分割点)からの相対時刻でクラスタ確認
SEG_STARTS_GLOBAL_APPROX = [0, 893.7, 1738.3, 2637.3, 3626.0, 4379.5, 5255.6, 6131.6]
print("\n--- release-blocker のうちセグメント開始+30秒(warmup)近傍(±5s)の件数 ---")
near_warmup = 0
for t in blockers["t_sec"]:
    for s in SEG_STARTS_GLOBAL_APPROX:
        if abs(t - (s+30)) <= 5.0:
            near_warmup += 1
            break
print(near_warmup, "/", len(blockers))
