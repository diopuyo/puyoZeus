"""cycle 33-36 + baseline 統合比較."""
from __future__ import annotations
import json
from pathlib import Path


def load(p):
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    videos = ["v89m3", "v97m11", "v70m2"]
    bl = load(Path("data/verify/retrospective_eval/baseline_v89m3.json"))
    cycles = ["cycle33", "cycle34", "cycle35", "cycle36"]
    boosts = {
        "cycle33": "tier1<20.0",
        "cycle34": "soft boost 0.4",
        "cycle35": "soft boost 1.0",
        "cycle36": "soft boost 0.6",
    }
    print("=" * 100)
    print("cycle 33-36 推論軸試行 vs baseline (v89m3 + 3 動画)")
    print("=" * 100)
    print(f"\n{'cycle':<30} {'verdict':<10} {'crit':>5} {'warn':>5}")
    print("-" * 55)
    bs = bl.get("summary", {})
    print(f"{'baseline (v89m3)':<30} {bl.get('verdict','?'):<10} "
          f"{bs.get('critical',0):>5} {bs.get('warning',0):>5}")
    for c in cycles:
        for v in videos:
            p = Path(f"data/verify/{c}_eval/{c}_{v}.json")
            d = load(p)
            s = d.get("summary", {})
            label = f"{c} ({v}, {boosts[c]})"
            print(f"{label:<30} {d.get('verdict','?'):<10} "
                  f"{s.get('critical',0):>5} {s.get('warning',0):>5}")
    print("\n=== v89m3 詳細 ===")
    metrics_data = {}
    metrics_data["baseline"] = bs
    for c in cycles:
        metrics_data[c] = load(
            Path(f"data/verify/{c}_eval/{c}_v89m3.json")
        ).get("summary", {})
    all_m = set()
    for s in metrics_data.values():
        all_m.update(s.get("by_metric", {}).keys())
    header = f"{'metric':<32} " + " ".join(
        f"{n:>9}" for n in ["base", *cycles]
    )
    print(header)
    print("-" * 100)
    for m in sorted(all_m):
        row = f"{m:<32} "
        for name in ["baseline", *cycles]:
            s = metrics_data[name]
            count = s.get("by_metric", {}).get(m, 0)
            crit = s.get("by_metric_critical", {}).get(m, 0)
            row += f"{count:>4}({crit:>3}) "
        print(row)


if __name__ == "__main__":
    main()
