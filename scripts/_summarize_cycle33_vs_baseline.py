"""cycle 33 (3 動画) と baseline (v89m3) を比較。"""
from __future__ import annotations

import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    cycle33_videos = ["v89m3", "v97m11", "v70m2"]
    baseline_path = Path(
        "data/verify/retrospective_eval/baseline_v89m3.json"
    )
    baseline = load(baseline_path)

    print("=" * 100)
    print("cycle 33 (= bg_fp tier 1) vs baseline (cnn_phase_b_large_v2.pt) 比較")
    print("=" * 100)
    print(
        f"\n{'cycle':<25} {'verdict':<10} {'total':>6} {'crit':>6} "
        f"{'warn':>6}"
    )
    print("-" * 60)

    bs = baseline.get("summary", {})
    print(
        f"{'baseline (v89m3)':<25} {baseline.get('verdict', '?'):<10} "
        f"{bs.get('total_violations', 0):>6} "
        f"{bs.get('critical', 0):>6} "
        f"{bs.get('warning', 0):>6}"
    )

    cycle33_data = {}
    for vid in cycle33_videos:
        path = Path(f"data/verify/cycle33_eval/cycle33_{vid}.json")
        d = load(path)
        cycle33_data[vid] = d
        s = d.get("summary", {})
        print(
            f"{'cycle33 (' + vid + ')':<25} {d.get('verdict', '?'):<10} "
            f"{s.get('total_violations', 0):>6} "
            f"{s.get('critical', 0):>6} "
            f"{s.get('warning', 0):>6}"
        )

    # v89m3 で baseline と cycle 33 の差分
    print("\n=== v89m3 baseline vs cycle 33 ===")
    print(f"{'metric':<35} {'baseline':>10} {'cycle33':>10} {'diff':>8}")
    print("-" * 70)
    bs_metrics = bs.get("by_metric", {})
    bs_crit = bs.get("by_metric_critical", {})
    c33_v89 = cycle33_data.get("v89m3", {}).get("summary", {})
    c33_metrics = c33_v89.get("by_metric", {})
    c33_crit = c33_v89.get("by_metric_critical", {})
    all_metrics = set(bs_metrics.keys()) | set(c33_metrics.keys())
    for m in sorted(all_metrics):
        bs_count = bs_metrics.get(m, 0)
        c33_count = c33_metrics.get(m, 0)
        diff = c33_count - bs_count
        diff_str = f"{diff:+d}"
        print(
            f"{m:<35} {bs_count:>5}({bs_crit.get(m, 0):>3}) "
            f"{c33_count:>5}({c33_crit.get(m, 0):>3}) {diff_str:>8}"
        )

    print("\n=== 3 動画 cycle 33 サマリ ===")
    print(f"{'metric':<35} {'v89m3':>10} {'v97m11':>10} {'v70m2':>10}")
    print("-" * 75)
    all_metrics_c33 = set()
    for vid in cycle33_videos:
        s = cycle33_data.get(vid, {}).get("summary", {})
        all_metrics_c33.update(s.get("by_metric", {}).keys())
    for m in sorted(all_metrics_c33):
        row = f"{m:<35} "
        for vid in cycle33_videos:
            s = cycle33_data.get(vid, {}).get("summary", {})
            count = s.get("by_metric", {}).get(m, 0)
            crit = s.get("by_metric_critical", {}).get(m, 0)
            row += f"{count:>5}({crit:>3}) "
        print(row)


if __name__ == "__main__":
    main()
