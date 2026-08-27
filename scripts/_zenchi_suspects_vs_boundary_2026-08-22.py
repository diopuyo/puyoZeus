import pandas as pd
import numpy as np

BOUNDARIES = np.array([86.67,141.07,232.47,278.10,335.97,416.47,487.87,514.23,609.63,695.27,751.23,828.37,
              958.90,985.90,1045.13,1097.67,1174.63,1235.77,1300.07,1356.87,1396.90,1432.20,1505.60,
              1582.03,1625.27,1798.53,1879.03,1918.40,1983.23,2034.13,2125.17,2172.07,2219.40,2275.87,
              2351.27,2393.20,2453.73,2544.43,2579.07,2706.17,2782.20,2812.30,2870.97,2898.23,2948.70,
              2988.57,3025.40,3051.70,3122.87,3170.83,3239.13,3273.13,3330.73,3356.20,3417.13,3625.80,
              3673.23,3749.57,3806.73,3894.00,3998.20,4040.60,4084.70,4142.83,4218.57,4293.20,4453.07,
              4550.37,4623.63,4665.27,4730.03,4767.40,4818.43,4871.23,4924.03,4981.33,5026.87,5061.83,
              5132.33,5194.47,5288.10,5366.17,5425.20,5487.00,5530.33,5606.13,5646.23,5744.97,5788.73,
              5832.47,5898.47,5935.73,6005.83,6071.53,6204.73,6281.70,6404.27,6456.20,6522.50,6567.83,
              6631.80,6664.17,6729.27,6755.53,6788.90,6832.90,6874.57,6936.60,6972.97, 0.0])
# 各セグメント開始も「試合開始扱いになりうる」実質的境界として一応候補に足す(0.0含む)

df = pd.read_csv("data/verify/judgment_scan_zenchi_2026-08-22/suspects.tsv", sep="\t")
blockers = df[df["stage"].isin(["display","both"])].copy()

def nearest_dist(t):
    return np.min(np.abs(BOUNDARIES - t))

blockers["dist_to_boundary"] = blockers["t_sec"].apply(nearest_dist)
print("release-blocker t_secの境界最近接距離 分布:")
print(blockers["dist_to_boundary"].describe())
print("\n<=2s (既存ガード内のはずだが漏れ):", (blockers["dist_to_boundary"]<=2.0).sum())
print("2-6s:", ((blockers["dist_to_boundary"]>2.0)&(blockers["dist_to_boundary"]<=6.0)).sum())
print("6-15s:", ((blockers["dist_to_boundary"]>6.0)&(blockers["dist_to_boundary"]<=15.0)).sum())
print(">15s (境界と無関係=深い試合中):", (blockers["dist_to_boundary"]>15.0).sum())

deep = blockers[blockers["dist_to_boundary"]>15.0].sort_values("t_sec")
print(f"\n=== 境界から15秒以上離れた(=深い試合中の)件数: {len(deep)} ===")
for _, row in deep.head(30).iterrows():
    print(f"  t={row['t_sec']:.2f} det={row['detector']} stage={row['stage']} dist={row['dist_to_boundary']:.1f} evidence={row['evidence'][:90]}")
