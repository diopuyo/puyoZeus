"""トレース1本から指定時刻近傍の score/base レコードを人間可読で出す。

使い方:
  python scripts/_diag_adv_nondet_trace_probe_2026-08-25.py trace_r1.jsonl 161.8 164.2 ...
"""
from __future__ import annotations

import json
import sys


def hf(h: str) -> float:
    try:
        return float.fromhex(h)
    except ValueError:
        return float("nan")


def main() -> None:
    path = sys.argv[1]
    targets = [float(x) for x in sys.argv[2:]]
    scores = []
    bases = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] == "score":
                scores.append(rec)
            elif rec["kind"] == "base":
                bases.setdefault(rec["n"], {})[rec["seq"]] = rec
    for tt in targets:
        # snap.t_sec が tt に最も近い score レコード
        best = min(scores, key=lambda r: abs(hf(r["snap"].get("t_sec", "nan")) - tt))
        ts = hf(best["snap"].get("t_sec", "nan"))
        print(f"=== 目標t={tt} -> n={best['n']} snap.t_sec={ts:.3f} ===")
        print(f"  adv={hf(best['adv']):+.4f} p1={hf(best['p1']):.4f} "
              f"crc1={best['crc1']} crc2={best['crc2']}")
        print(f"  drivers={[(d[0], round(hf(d[1]), 4)) for d in best['drivers']]}")
        sn = {k: hf(v) for k, v in best["snap"].items()}
        print(f"  snap: pending=({sn.get('pending_p1')},{sn.get('pending_p2')}) "
              f"net_capped={sn.get('net_balance_capped')} "
              f"forecast=({sn.get('forecast_p1')},{sn.get('forecast_p2')})")
        for seq in (1, 2):
            br = bases.get(best["n"], {}).get(seq)
            if br:
                heavy = {k: round(hf(v), 4) for k, v in br["row"].items()
                         if k in ("current_max_chain", "dig_resistance", "ukeyasusa",
                                  "sub_chain_count", "saturation_chain_upper",
                                  "board_puyo_total", "board_ojama_count",
                                  "chain_articulation_point_count")}
                print(f"  side{seq} crc={br['crc']} {heavy}")


if __name__ == "__main__":
    main()
