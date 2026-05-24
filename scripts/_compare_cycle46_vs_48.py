"""cycle 46 (= 偽 chain ガード前) vs cycle 48 (= ガード後) 比較."""
from __future__ import annotations
import json
from pathlib import Path


def main():
    videos = ["v29m2", "v40m7", "v51m2", "v57m2", "v70m2", "v89m3", "v95m15", "v97m11"]
    print(f"{'video':<10}{'c46 crit':>10}{'c48 crit':>10}{'diff':>8}{'c46 chain*':>12}{'c48 chain*':>12}")
    print("-" * 70)
    chain_metrics = ["chain_no_puyo_loss", "retrospective_chain_missing", "chain_state_too_short"]
    total_c46 = 0
    total_c48 = 0
    chain_c46_total = 0
    chain_c48_total = 0
    for v in videos:
        p46 = Path(f"data/verify/cycle46_eval/cycle46_{v}_buf15.json")
        p48 = Path(f"data/verify/cycle48_eval/cycle48_{v}.json")
        if not p46.is_file() or not p48.is_file():
            continue
        d46 = json.load(open(p46, encoding="utf-8"))
        d48 = json.load(open(p48, encoding="utf-8"))
        s46 = d46["summary"]
        s48 = d48["summary"]
        c46 = s46["critical"]
        c48 = s48["critical"]
        chain46 = sum(s46["by_metric_critical"].get(m, 0) for m in chain_metrics)
        chain48 = sum(s48["by_metric_critical"].get(m, 0) for m in chain_metrics)
        total_c46 += c46
        total_c48 += c48
        chain_c46_total += chain46
        chain_c48_total += chain48
        print(f"{v:<10}{c46:>10}{c48:>10}{c48-c46:>+8}{chain46:>12}{chain48:>12}")
    print("-" * 70)
    print(f"{'合計':<10}{total_c46:>10}{total_c48:>10}{total_c48-total_c46:>+8}{chain_c46_total:>12}{chain_c48_total:>12}")
    print()
    print(f"chain 系合計削減: {chain_c46_total} → {chain_c48_total} ({chain_c48_total-chain_c46_total:+d}, {100*(chain_c48_total-chain_c46_total)/max(1,chain_c46_total):.1f}%)")


if __name__ == "__main__":
    main()
