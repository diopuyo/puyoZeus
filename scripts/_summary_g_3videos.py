"""cycle 56 G の 3 動画 (= ユーザー目視用) 結果集計."""
import json
from pathlib import Path

videos = ["v89m7", "v30_match11", "v97_match11"]
print("=" * 70)
print("cycle 56 G (= 定数 N=8/votes=5 + 新 metric KC) — 3 動画集計")
print("=" * 70)
for v in videos:
    p = Path(f"data/verify/cycle56_g_viz/{v}.json")
    if not p.exists():
        print(f"\n[{v}] NOT FOUND")
        continue
    d = json.load(open(p, encoding="utf-8"))
    s = d.get("summary", {})
    print(f"\n[{v}]")
    print(f"  critical: {s.get('critical')}")
    print(f"  warning: {s.get('warning')}")
    print(f"  verdict: {d.get('verdict')}")
    by_m = s.get("by_metric", {})
    print("  by_metric:")
    for k, n in sorted(by_m.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {n}")

    # static_color_flicker 統合
    total_flips_1p = 0
    total_flips_2p = 0
    stable_1p = 0
    stable_2p = 0
    pairs_combined: dict[str, int] = {}
    for viol in d.get("violations", []):
        if viol.get("metric") != "static_color_flicker":
            continue
        ex = viol.get("extra", {})
        side = viol.get("side")
        tf = ex.get("total_flips", 0)
        sf = ex.get("stable_frames", 0)
        if side == "1P":
            total_flips_1p = tf
            stable_1p = sf
        else:
            total_flips_2p = tf
            stable_2p = sf
        for pk, pn in ex.get("pair_counts", {}).items():
            pairs_combined[pk] = pairs_combined.get(pk, 0) + pn
    print(f"  static_color_flicker: 1P={total_flips_1p}件/{stable_1p}f, 2P={total_flips_2p}件/{stable_2p}f")
    sorted_pairs = sorted(pairs_combined.items(), key=lambda kv: -kv[1])[:5]
    pair_names = {
        "1-2": "赤→青", "2-1": "青→赤", "1-3": "赤→緑", "3-1": "緑→赤",
        "1-4": "赤→黄", "4-1": "黄→赤", "1-5": "赤→紫", "5-1": "紫→赤",
        "2-3": "青→緑", "3-2": "緑→青", "2-4": "青→黄", "4-2": "黄→青",
        "2-5": "青→紫", "5-2": "紫→青", "3-4": "緑→黄", "4-3": "黄→緑",
        "3-5": "緑→紫", "5-3": "紫→緑", "4-5": "黄→紫", "5-4": "紫→黄",
    }
    pair_str = ", ".join(f"{pair_names.get(k, k)}={n}" for k, n in sorted_pairs)
    print(f"  主訴 top5: {pair_str}")

print()
print("=" * 70)
print("採用判定材料")
print("=" * 70)
print("- 3 動画 viz は cycle56_v3b_viz と同じ動画なので、 ユーザー目視で")
print("  「ぷよ誤認 c56_v3b より さらに 減ったか?」 を比較確認可能")
print("- 新 metric 数値で「赤↔青 系の誤認が中心」 と物理的に判明")
print("- 8 動画 eval 完走後に critical 増減で最終判定 (= baseline 1512 / c56_v3b 1551)")
