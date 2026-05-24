"""cycle 33 (tier 1) vs cycle 34 (soft prior) vs baseline 比較."""
from __future__ import annotations

import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    videos = ["v89m3", "v97m11", "v70m2"]
    print("=" * 100)
    print("cycle 33 (tier 1 < 20) vs cycle 34 (soft prior) vs baseline 比較")
    print("=" * 100)

    # baseline は v89m3 のみ評価済
    baseline = load(Path("data/verify/retrospective_eval/baseline_v89m3.json"))

    # 全体サマリ
    print(
        f"\n{'cycle':<25} {'verdict':<10} {'total':>6} {'crit':>6} {'warn':>6}"
    )
    print("-" * 60)
    bs = baseline.get("summary", {})
    print(
        f"{'baseline (v89m3)':<25} {baseline.get('verdict', '?'):<10} "
        f"{bs.get('total_violations', 0):>6} {bs.get('critical', 0):>6} "
        f"{bs.get('warning', 0):>6}"
    )
    for cycle_name, dirname in [
        ("cycle33", "cycle33_eval"), ("cycle34", "cycle34_eval"),
    ]:
        for vid in videos:
            path = Path(f"data/verify/{dirname}/{cycle_name}_{vid}.json")
            d = load(path)
            s = d.get("summary", {})
            label = f"{cycle_name} ({vid})"
            print(
                f"{label:<25} {d.get('verdict', '?'):<10} "
                f"{s.get('total_violations', 0):>6} "
                f"{s.get('critical', 0):>6} {s.get('warning', 0):>6}"
            )

    # v89m3 で baseline / cycle33 / cycle34 を直接比較
    print("\n=== v89m3 詳細比較 ===")
    c33 = load(Path("data/verify/cycle33_eval/cycle33_v89m3.json"))
    c34 = load(Path("data/verify/cycle34_eval/cycle34_v89m3.json"))
    print(
        f"{'metric':<35} {'baseline':>12} {'cycle33':>12} {'cycle34':>12} "
        f"{'34 vs base':>12}"
    )
    print("-" * 90)
    bs_m = bs.get("by_metric", {})
    bs_c = bs.get("by_metric_critical", {})
    c33_s = c33.get("summary", {})
    c34_s = c34.get("summary", {})
    c33_m = c33_s.get("by_metric", {})
    c33_c = c33_s.get("by_metric_critical", {})
    c34_m = c34_s.get("by_metric", {})
    c34_c = c34_s.get("by_metric_critical", {})
    all_metrics = sorted(set(bs_m) | set(c33_m) | set(c34_m))
    for m in all_metrics:
        b = bs_m.get(m, 0)
        c33v = c33_m.get(m, 0)
        c34v = c34_m.get(m, 0)
        diff = c34v - b
        print(
            f"{m:<35} {b:>5}({bs_c.get(m, 0):>3}) "
            f"{c33v:>5}({c33_c.get(m, 0):>3}) "
            f"{c34v:>5}({c34_c.get(m, 0):>3}) "
            f"{diff:>+12d}"
        )

    # 3 動画 cycle 34 サマリ
    print("\n=== cycle 34 3 動画サマリ ===")
    print(f"{'metric':<35} {'v89m3':>12} {'v97m11':>12} {'v70m2':>12}")
    print("-" * 75)
    metrics_c34 = set()
    for vid in videos:
        s = load(
            Path(f"data/verify/cycle34_eval/cycle34_{vid}.json")
        ).get("summary", {})
        metrics_c34.update(s.get("by_metric", {}).keys())
    for m in sorted(metrics_c34):
        row = f"{m:<35} "
        for vid in videos:
            s = load(
                Path(f"data/verify/cycle34_eval/cycle34_{vid}.json")
            ).get("summary", {})
            count = s.get("by_metric", {}).get(m, 0)
            crit = s.get("by_metric_critical", {}).get(m, 0)
            row += f"{count:>5}({crit:>3}) "
        print(row)


if __name__ == "__main__":
    main()
