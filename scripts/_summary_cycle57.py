"""cycle 57 結果集計 (= baseline / c56_v3b / c57 比較).

出力:
- data/verify/cycle57_eval/_comparison.json (= 8 動画 critical + flicker 比較)
- data/verify/cycle57_viz/_user_summary.json (= 4 動画 + 主訴ペア + ojama 維持率)
"""
import json
from pathlib import Path

ROOT = Path("data/verify")

def load(p):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None

def calc_flicker(d):
    return sum(
        v.get("extra", {}).get("total_flips", 0)
        for v in d.get("violations", [])
        if v.get("metric") == "static_color_flicker"
    )

def calc_ojama(d):
    """ojama cell の総数 / 全 STABLE cell 数 = ojama 認識率."""
    ojama_cells = 0
    total_cells = 0
    stable_frames = 0
    for v in d.get("violations", []):
        if v.get("metric") == "ojama_global_scarcity":
            ex = v.get("extra", {})
            return ex.get("ratio", 0.0), ex.get("ojama_cells", 0), ex.get("total_cells", 0)
    # ojama_global_scarcity violation 不在 = 通常認識
    return None, None, None

# ============ Phase 1: 4 動画 user 比較 ============
user_videos = ["v89m7", "v30_match11", "v30_5min", "v97_match11"]
user_summary = {}
for v in user_videos:
    g = load(f"data/verify/cycle57_viz/{v}.json")
    if not g:
        user_summary[v] = {"status": "viz 未完"}
        continue
    s = g.get("summary", {})
    flicker_total = calc_flicker(g)
    pair_counts: dict[str, int] = {}
    for viol in g.get("violations", []):
        if viol.get("metric") == "static_color_flicker":
            for k, n in viol.get("extra", {}).get("pair_counts", {}).items():
                pair_counts[k] = pair_counts.get(k, 0) + n
    sorted_pairs = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:5]
    pair_names = {
        "1-2": "赤→青", "2-1": "青→赤", "1-3": "赤→緑", "3-1": "緑→赤",
        "1-4": "赤→黄", "4-1": "黄→赤", "1-5": "赤→紫", "5-1": "紫→赤",
        "2-3": "青→緑", "3-2": "緑→青", "2-4": "青→黄", "4-2": "黄→青",
        "2-5": "青→紫", "5-2": "紫→青", "3-4": "緑→黄", "4-3": "黄→緑",
        "3-5": "緑→紫", "5-3": "紫→緑", "4-5": "黄→紫", "5-4": "紫→黄",
    }
    user_summary[v] = {
        "critical": s.get("critical", 0),
        "warning": s.get("warning", 0),
        "flicker_total": flicker_total,
        "verdict": g.get("verdict"),
        "pair_top5": {pair_names.get(k, k): n for k, n in sorted_pairs},
    }

# ============ Phase 2: 8 動画 評価 比較 ============
videos8 = ["v29m2", "v40m7", "v51m2", "v57m2", "v70m2", "v89m3", "v95m15", "v97m11"]
c57_totals = {"critical": 0, "warning": 0, "flicker": 0}
c57_per_video = {}
for v in videos8:
    d = load(f"data/verify/cycle57_eval/{v}.json")
    if not d:
        continue
    s = d.get("summary", {})
    flicker_total = calc_flicker(d)
    c57_per_video[v] = {
        "critical": s.get("critical", 0),
        "warning": s.get("warning", 0),
        "flicker": flicker_total,
    }
    c57_totals["critical"] += s.get("critical", 0)
    c57_totals["warning"] += s.get("warning", 0)
    c57_totals["flicker"] += flicker_total

# baseline + c56_v3b 比較
baseline_summary = load("data/verify/baseline_v3_eval/_summary.json")
v3b_summary = load("data/verify/cycle56_v3b_eval/_summary.json")
baseline_critical = baseline_summary["totals"]["critical"] if baseline_summary else 1512
v3b_critical = v3b_summary["totals"]["critical"] if v3b_summary else 1551

cmp = {
    "baseline_critical": baseline_critical,
    "c56_v3b_critical": v3b_critical,
    "c57_critical": c57_totals["critical"],
    "c57_warning": c57_totals["warning"],
    "c57_flicker_total": c57_totals["flicker"],
    "diff_vs_baseline": c57_totals["critical"] - baseline_critical,
    "pct_vs_baseline": round((c57_totals["critical"] - baseline_critical) / max(1, baseline_critical) * 100, 1),
    "diff_vs_v3b": c57_totals["critical"] - v3b_critical,
    "per_video": c57_per_video,
}
Path("data/verify/cycle57_eval/_comparison.json").write_text(
    json.dumps(cmp, indent=2, ensure_ascii=False), encoding="utf-8",
)
Path("data/verify/cycle57_viz/_user_summary.json").write_text(
    json.dumps(user_summary, indent=2, ensure_ascii=False), encoding="utf-8",
)

print("=== cycle 57 vs baseline / c56_v3b ===")
print(json.dumps(cmp, indent=2, ensure_ascii=False))
print()
print("=== 4 動画 ユーザー目視用 ===")
print(json.dumps(user_summary, indent=2, ensure_ascii=False))
