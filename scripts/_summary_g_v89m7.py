"""cycle 56 G の v89m7 結果集計 (= shell escape 回避用 script)."""
import json
import glob
from pathlib import Path

g_path = Path("data/verify/cycle56_g_viz/v89m7.json")
g = json.load(open(g_path, encoding="utf-8"))
s = g.get("summary", {})
print("=== v89m7 G (= 定数拡張 + KC metric) ===")
print(f"critical: {s.get('critical')}")
print(f"warning: {s.get('warning')}")
print(f"info: {s.get('info')}")
print(f"verdict: {g.get('verdict')}")
print()
print("by_metric:")
for k, v in s.get("by_metric", {}).items():
    print(f"  {k}: {v}")
print()
print("=== static_color_flicker 詳細 ===")
for v in g.get("violations", []):
    if v.get("metric") == "static_color_flicker":
        print(f"side={v.get('side')} severity={v.get('severity')}")
        ex = v.get("extra", {})
        print(f"  total_flips: {ex.get('total_flips')}")
        print(f"  stable_frames: {ex.get('stable_frames')}")
        pcs = ex.get("pair_counts", {})
        sorted_pairs = sorted(pcs.items(), key=lambda kv: -kv[1])
        print(f"  pair_counts (top 10): {sorted_pairs[:10]}")
print()
print("=== 比較対象 (= c56_v3b 単独 v89m7 board log) ===")
for p in glob.glob("logs/cycle56_v3b*/viz_v89m7*") + glob.glob("data/verify/cycle55_viz/*v89m7*"):
    print(f"  found: {p}")
