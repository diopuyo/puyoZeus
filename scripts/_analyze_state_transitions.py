"""board log から state 推移を秒単位で集計し、 試合開始判定の精度を確認."""
from __future__ import annotations
import json
import sys
from pathlib import Path

def main(path):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    print(f"Loaded {len(entries)} frames from {path}")
    # 秒ごとに 1P/2P state の分布を集計
    by_sec = {}
    for e in entries:
        sec = int(e["t_sec"])
        if sec not in by_sec:
            by_sec[sec] = {"p1": {}, "p2": {}}
        by_sec[sec]["p1"][e["p1_state"]] = by_sec[sec]["p1"].get(e["p1_state"], 0) + 1
        by_sec[sec]["p2"][e["p2_state"]] = by_sec[sec]["p2"].get(e["p2_state"], 0) + 1
    print()
    print(f"{'sec':<5} {'p1 dominant':<15} {'p2 dominant':<15}")
    print("-" * 40)
    for sec in sorted(by_sec.keys()):
        p1 = max(by_sec[sec]["p1"].items(), key=lambda x: x[1])[0]
        p2 = max(by_sec[sec]["p2"].items(), key=lambda x: x[1])[0]
        print(f"{sec:<5} {p1:<15} {p2:<15}")

if __name__ == "__main__":
    main(sys.argv[1])
