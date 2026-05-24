"""cycle 46 (= 4 連結 gate 前) vs cycle 49 (= ChainPhaseDetector に 4 連結 gate) 比較."""
from __future__ import annotations
import json
from pathlib import Path


def main():
    videos = ["v29m2", "v40m7", "v51m2", "v57m2", "v70m2", "v89m3", "v95m15", "v97m11"]
    print(f"{'video':<10}{'c46 crit':>10}{'c49 crit':>10}{'diff':>8}{'c46 chain*':>12}{'c49 chain*':>12}{'chain diff':>12}")
    print("-" * 80)
    chain_metrics = ["chain_no_puyo_loss", "retrospective_chain_missing", "chain_state_too_short", "chain_no_disappear"]
    total_c46 = 0; total_c49 = 0
    chain_c46_total = 0; chain_c49_total = 0
    for v in videos:
        p46 = Path(f"data/verify/cycle46_eval/cycle46_{v}_buf15.json")
        p49 = Path(f"data/verify/cycle49_eval/cycle49_{v}.json")
        if not p46.is_file() or not p49.is_file():
            continue
        d46 = json.load(open(p46, encoding="utf-8"))
        d49 = json.load(open(p49, encoding="utf-8"))
        s46 = d46["summary"]; s49 = d49["summary"]
        c46 = s46["critical"]; c49 = s49["critical"]
        chain46 = sum(s46["by_metric_critical"].get(m, 0) for m in chain_metrics)
        chain49 = sum(s49["by_metric_critical"].get(m, 0) for m in chain_metrics)
        total_c46 += c46; total_c49 += c49
        chain_c46_total += chain46; chain_c49_total += chain49
        print(f"{v:<10}{c46:>10}{c49:>10}{c49-c46:>+8}{chain46:>12}{chain49:>12}{chain49-chain46:>+12}")
    print("-" * 80)
    print(f"{'合計':<10}{total_c46:>10}{total_c49:>10}{total_c49-total_c46:>+8}{chain_c46_total:>12}{chain_c49_total:>12}{chain_c49_total-chain_c46_total:>+12}")
    print()
    print(f"chain 系合計削減: {chain_c46_total} → {chain_c49_total} ({chain_c49_total-chain_c46_total:+d}, {100*(chain_c49_total-chain_c46_total)/max(1,chain_c46_total):.1f}%)")

    print("\n=== v89m3 metric 別 ===")
    d46 = json.load(open(f"data/verify/cycle46_eval/cycle46_v89m3_buf15.json", encoding="utf-8"))
    d49 = json.load(open(f"data/verify/cycle49_eval/cycle49_v89m3.json", encoding="utf-8"))
    bm46 = d46["summary"]["by_metric_critical"]
    bm49 = d49["summary"]["by_metric_critical"]
    all_m = sorted(set(bm46) | set(bm49))
    for m in all_m:
        o = bm46.get(m, 0)
        n = bm49.get(m, 0)
        print(f"  {m:<35} c46={o:>4} c49={n:>4} ({n-o:+d})")


if __name__ == "__main__":
    main()
