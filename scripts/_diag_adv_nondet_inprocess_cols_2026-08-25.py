"""プロセス内不一致グループについて、base 指標のどのキーが動いたかを名指しする。

同一 (crc1, crc2, net, f1, f2) で adv が食い違った score レコード群に対し、
対応する base レコード (seq1=1P, seq2=2P) の row dict をキー単位で比較する。

使い方:
  python scripts/_diag_adv_nondet_inprocess_cols_2026-08-25.py trace_r1.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def hf(h: str) -> float:
    try:
        return float.fromhex(h)
    except ValueError:
        return float("nan")


def main() -> None:
    path = sys.argv[1]
    scores = {}
    bases = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] == "score":
                scores[rec["n"]] = rec
            elif rec["kind"] == "base":
                bases[rec["n"]][rec["seq"]] = rec
    groups = defaultdict(list)
    for n, rec in scores.items():
        key = (rec["crc1"], rec["crc2"], rec["snap"].get("net_balance_capped"),
               rec["snap"].get("forecast_p1"), rec["snap"].get("forecast_p2"))
        groups[key].append(n)
    for key, ns in groups.items():
        if len(ns) < 2:
            continue
        advs = {scores[n]["adv"] for n in ns}
        if len(advs) < 2:
            continue
        ns = sorted(ns)
        print(f"=== 不一致グループ crc1={key[0]} crc2={key[1]} 呼出n={ns} ===")
        ref = ns[0]
        for n in ns[1:]:
            for seq in (1, 2):
                ra = bases.get(ref, {}).get(seq)
                rb = bases.get(n, {}).get(seq)
                if not ra or not rb:
                    print(f"  n={ref} vs n={n} seq{seq}: baseレコード欠落 "
                          f"(ra={bool(ra)} rb={bool(rb)}) "
                          f"crc_ra={ra['crc'] if ra else '-'} "
                          f"crc_rb={rb['crc'] if rb else '-'}")
                    continue
                diffs = {k: (hf(ra["row"][k]), hf(rb["row"].get(k, "nan")))
                         for k in ra["row"] if ra["row"][k] != rb["row"].get(k)}
                if diffs:
                    print(f"  n={ref} vs n={n} seq{seq} (盤面crc {ra['crc']}=={rb['crc']}:"
                          f" {ra['crc'] == rb['crc']}):")
                    for k, (va, vb) in diffs.items():
                        print(f"      {k}: {va:.6f} -> {vb:.6f}")
                else:
                    print(f"  n={ref} vs n={n} seq{seq}: base全キー一致")


if __name__ == "__main__":
    main()
