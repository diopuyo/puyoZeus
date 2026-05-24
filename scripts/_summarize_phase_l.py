"""Phase L 全 7 動画 viz 評価サマリ."""
from __future__ import annotations
import json
from pathlib import Path


def main():
    videos = ["v29m22", "v30m23", "v40m10", "v50m26", "v75m14", "v89m7", "v95m3"]
    categories = {
        "v29m22": "existing_train",
        "v30m23": "holdout",
        "v40m10": "existing_train",
        "v50m26": "holdout",
        "v75m14": "holdout",
        "v89m7": "existing_train",
        "v95m3": "existing_train",
    }
    print("Phase L 全 7 動画 viz 評価サマリ")
    print("=" * 75)
    print(f"{'video':<10}{'category':<16}{'verdict':<10}{'crit':>6}{'warn':>6}")
    print("-" * 50)
    for v in videos:
        path = Path(f"data/verify/phase_l_eval/phase_l_{v}.json")
        if not path.is_file():
            continue
        d = json.load(open(path, encoding="utf-8"))
        s = d.get("summary", {})
        print(f"{v:<10}{categories.get(v, ''):<16}{d.get('verdict', '?'):<10}"
              f"{s.get('critical', 0):>6}{s.get('warning', 0):>6}")
    print()
    print("=== metric 別 全 7 動画合計 ===")
    by_metric = {}
    by_metric_crit = {}
    for v in videos:
        path = Path(f"data/verify/phase_l_eval/phase_l_{v}.json")
        if not path.is_file():
            continue
        d = json.load(open(path, encoding="utf-8"))
        bm = d.get("summary", {}).get("by_metric", {})
        bmc = d.get("summary", {}).get("by_metric_critical", {})
        for m, c in bm.items():
            by_metric[m] = by_metric.get(m, 0) + c
        for m, c in bmc.items():
            by_metric_crit[m] = by_metric_crit.get(m, 0) + c
    for m in sorted(by_metric.keys()):
        print(f"  {m:<35} total={by_metric[m]:>4} critical={by_metric_crit.get(m, 0):>4}")


if __name__ == "__main__":
    main()
