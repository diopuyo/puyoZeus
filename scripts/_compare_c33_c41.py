"""cycle 33 (baseline 相当) vs cycle 41 (PuyoPresenceGate) 比較."""
from __future__ import annotations
import json
from pathlib import Path


def load(p):
    if not p.is_file(): return {}
    return json.load(open(p, encoding="utf-8"))


def main():
    videos = ["v89m3", "v97m11", "v70m2"]
    print("=" * 70)
    print("cycle 33 (= baseline 相当、 gate なし) vs cycle 41 (gate あり)")
    print("=" * 70)
    print(f"\n{'video':<10} {'c33 crit':>10} {'c41 crit':>10} {'diff':>10}")
    print("-" * 50)
    for v in videos:
        c33 = load(Path(f"data/verify/cycle33_eval/cycle33_{v}.json"))
        c41 = load(Path(f"data/verify/cycle41_eval/cycle41_{v}.json"))
        c33_crit = c33.get("summary", {}).get("critical", 0)
        c41_crit = c41.get("summary", {}).get("critical", 0)
        diff = c41_crit - c33_crit
        print(f"{v:<10} {c33_crit:>10} {c41_crit:>10} {diff:>+10}")
    print()
    # metric 別比較 (= v89m3)
    print("=== v89m3 metric 別 ===")
    c33 = load(Path("data/verify/cycle33_eval/cycle33_v89m3.json"))
    c41 = load(Path("data/verify/cycle41_eval/cycle41_v89m3.json"))
    bm33 = c33.get("summary", {}).get("by_metric_critical", {})
    bm41 = c41.get("summary", {}).get("by_metric_critical", {})
    metrics = sorted(set(bm33) | set(bm41))
    for m in metrics:
        v33 = bm33.get(m, 0)
        v41 = bm41.get(m, 0)
        d = v41 - v33
        print(f"  {m:<35} c33={v33:>4} c41={v41:>4} ({d:+d})")
    print()
    print("=== v70m2 metric 別 ===")
    c33 = load(Path("data/verify/cycle33_eval/cycle33_v70m2.json"))
    c41 = load(Path("data/verify/cycle41_eval/cycle41_v70m2.json"))
    bm33 = c33.get("summary", {}).get("by_metric_critical", {})
    bm41 = c41.get("summary", {}).get("by_metric_critical", {})
    metrics = sorted(set(bm33) | set(bm41))
    for m in metrics:
        v33 = bm33.get(m, 0)
        v41 = bm41.get(m, 0)
        d = v41 - v33
        print(f"  {m:<35} c33={v33:>4} c41={v41:>4} ({d:+d})")


if __name__ == "__main__":
    main()
