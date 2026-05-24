"""4 model 遡及評価結果の比較サマリ."""
from __future__ import annotations

import json
from pathlib import Path

CYCLES = ["baseline", "cycle32d", "cycle32e", "cycle32g"]
VIDEO_ID = "v89m3"
REPORT_DIR = Path("data/verify/retrospective_eval")


def main() -> None:
    print("=" * 100)
    print(f"4 model 遡及評価サマリ (動画: {VIDEO_ID})")
    print("=" * 100)
    print(
        f"\n{'cycle':<12} {'verdict':<10} {'total':>6} {'crit':>6} "
        f"{'warn':>6} {'info':>6}"
    )
    print("-" * 60)
    all_data = {}
    for cycle in CYCLES:
        path = REPORT_DIR / f"{cycle}_{VIDEO_ID}.json"
        if not path.is_file():
            print(f"{cycle:<12} (file not found)")
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        all_data[cycle] = d
        s = d["summary"]
        print(
            f"{cycle:<12} {d['verdict']:<10} "
            f"{s['total_violations']:>6} {s['critical']:>6} "
            f"{s['warning']:>6} {s['info']:>6}"
        )
    print()
    # メトリクス別比較
    metrics = set()
    for d in all_data.values():
        metrics.update(d["summary"]["by_metric"].keys())
    print(f"\n{'metric':<35} ", end="")
    for cycle in CYCLES:
        print(f"{cycle:>10}", end="")
    print()
    print("-" * 100)
    for metric in sorted(metrics):
        print(f"{metric:<35} ", end="")
        for cycle in CYCLES:
            if cycle in all_data:
                count = all_data[cycle]["summary"]["by_metric"].get(metric, 0)
                crit = all_data[cycle]["summary"]["by_metric_critical"].get(
                    metric, 0
                )
                print(f"{count:>5}({crit:>3})", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    print()
    # 注目すべき viz 上の問題
    print("\n=== Notable Findings ===")
    for cycle, d in all_data.items():
        crit = d["summary"]["critical"]
        verdict = d["verdict"]
        bg_dominant = d["summary"]["by_metric_critical"].get(
            "bg_color_dominant", 0
        )
        chain_missing = d["summary"]["by_metric_critical"].get(
            "retrospective_chain_missing", 0
        )
        chain_persist = d["summary"]["by_metric_critical"].get(
            "chain_no_disappear", 0
        )
        print(
            f"  {cycle:<12}: verdict={verdict}, critical={crit}, "
            f"bg_dominant={bg_dominant}, "
            f"chain_missing={chain_missing}, "
            f"chain_persist={chain_persist}"
        )


if __name__ == "__main__":
    main()
