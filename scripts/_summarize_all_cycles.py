"""cycle 33-35 と baseline の v89m3 比較."""
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
    print("cycle 33 / 34 / 35 vs baseline 比較 (v89m3 メイン + 3 動画)")
    print("=" * 100)
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
        ("cycle33", "cycle33_eval"),
        ("cycle34", "cycle34_eval"),
        ("cycle35", "cycle35_eval"),
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

    # v89m3 で baseline / c33 / c34 / c35 直接比較
    print("\n=== v89m3 詳細比較 (baseline ↔ c33 ↔ c34 ↔ c35) ===")
    c33 = load(Path("data/verify/cycle33_eval/cycle33_v89m3.json"))
    c34 = load(Path("data/verify/cycle34_eval/cycle34_v89m3.json"))
    c35 = load(Path("data/verify/cycle35_eval/cycle35_v89m3.json"))
    print(
        f"{'metric':<35} {'base':>10} {'c33':>10} {'c34':>10} {'c35':>10} "
        f"{'35 vs base':>12}"
    )
    print("-" * 100)
    bs_m = bs.get("by_metric", {})
    bs_c = bs.get("by_metric_critical", {})
    def smc(x): return x.get("summary", {}).get("by_metric", {})
    def src(x): return x.get("summary", {}).get("by_metric_critical", {})
    metrics = sorted(
        set(bs_m) | set(smc(c33)) | set(smc(c34)) | set(smc(c35))
    )
    for m in metrics:
        b = bs_m.get(m, 0)
        v33 = smc(c33).get(m, 0)
        v34 = smc(c34).get(m, 0)
        v35 = smc(c35).get(m, 0)
        diff = v35 - b
        print(
            f"{m:<35} {b:>4}({bs_c.get(m, 0):>3}) "
            f"{v33:>4}({src(c33).get(m, 0):>3}) "
            f"{v34:>4}({src(c34).get(m, 0):>3}) "
            f"{v35:>4}({src(c35).get(m, 0):>3}) "
            f"{diff:>+12d}"
        )

    # cycle 35 3 動画サマリ
    print("\n=== cycle 35 3 動画サマリ ===")
    print(f"{'metric':<35} {'v89m3':>12} {'v97m11':>12} {'v70m2':>12}")
    print("-" * 75)
    metrics_c35 = set()
    for vid in videos:
        s = smc(load(Path(f"data/verify/cycle35_eval/cycle35_{vid}.json")))
        metrics_c35.update(s.keys())
    for m in sorted(metrics_c35):
        row = f"{m:<35} "
        for vid in videos:
            d = load(Path(f"data/verify/cycle35_eval/cycle35_{vid}.json"))
            count = smc(d).get(m, 0)
            crit = src(d).get(m, 0)
            row += f"{count:>5}({crit:>3}) "
        print(row)


if __name__ == "__main__":
    main()
