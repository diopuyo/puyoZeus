"""計装トレース (trace_*.jsonl) 2本の突合 — 揺れた指標列を名指しする。

score レコード (リフレッシュ1回ごと) を呼出順 n で対応付け、
  - 入力 (crc1/crc2/snap) が同一か
  - 出力 (adv/p1/drivers) が同一か
  - base レコード (指標dict) のどのキーが違うか
を報告する。

使い方:
  python scripts/_diag_adv_nondet_trace_compare_2026-08-25.py trace_r1.jsonl trace_r2.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: str):
    env = None
    scores = {}
    bases = defaultdict(list)  # n -> [1P base rec, 2P base rec] (出現順)
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] == "env":
                env = rec
            elif rec["kind"] == "score":
                scores[rec["n"]] = rec
            elif rec["kind"] == "base":
                bases[rec["n"]].append(rec)
    return env, scores, bases


def hf(h: str) -> float:
    try:
        return float.fromhex(h)
    except ValueError:
        return float("nan")


def main() -> None:
    pa, pb = sys.argv[1], sys.argv[2]
    env_a, sc_a, ba_a = load(pa)
    env_b, sc_b, ba_b = load(pb)
    print(f"A={Path(pa).name} env={env_a}")
    print(f"B={Path(pb).name} env={env_b}")
    ns = sorted(set(sc_a) & set(sc_b))
    print(f"scoreレコード: A={len(sc_a)} B={len(sc_b)} 共通n={len(ns)}")
    n_out_diff = n_in_diff = 0
    diff_keys_count: dict[str, int] = defaultdict(int)
    max_key_diff: dict[str, float] = defaultdict(float)
    examples = []
    for n in ns:
        a, b = sc_a[n], sc_b[n]
        in_same = (a["crc1"] == b["crc1"] and a["crc2"] == b["crc2"]
                   and a["snap"] == b["snap"])
        out_same = (a["adv"] == b["adv"] and a["p1"] == b["p1"]
                    and a["drivers"] == b["drivers"])
        if not in_same:
            n_in_diff += 1
            snap_diff = {k: (a["snap"].get(k), b["snap"].get(k))
                         for k in set(a["snap"]) | set(b["snap"])
                         if a["snap"].get(k) != b["snap"].get(k)}
            if n_in_diff <= 5:
                print(f"[入力差] n={n} crc1 {a['crc1']}=={b['crc1']}: "
                      f"{a['crc1']==b['crc1']} crc2一致: {a['crc2']==b['crc2']} "
                      f"snap差={ {k: (hf(v[0]) if v[0] else None, hf(v[1]) if v[1] else None) for k, v in snap_diff.items()} }")
        if out_same:
            continue
        n_out_diff += 1
        adv_a, adv_b = hf(a["adv"]), hf(b["adv"])
        # base 側のどのキーが揺れたか (出現順: 1つ目=1P, 2つ目=2P)
        key_report = []
        for seq in (1, 2):
            la, lb = ba_a.get(n, []), ba_b.get(n, [])
            ra = la[seq - 1] if len(la) >= seq else None
            rb = lb[seq - 1] if len(lb) >= seq else None
            if not ra or not rb:
                key_report.append(f"seq{seq}:baseレコード欠落")
                continue
            if ra["crc"] != rb["crc"]:
                key_report.append(f"seq{seq}:盤面crc不一致!")
            for k in ra["row"]:
                va, vb = ra["row"][k], rb["row"].get(k)
                if va != vb:
                    # NaN==NaN は hex 文字列 'nan' 同士で一致扱いになる (良い)
                    d = abs(hf(va) - hf(vb))
                    diff_keys_count[f"{k}(side{seq})"] += 1
                    max_key_diff[f"{k}(side{seq})"] = max(
                        max_key_diff[f"{k}(side{seq})"], d)
                    key_report.append(f"seq{seq}.{k}: {hf(va):.6f}->{hf(vb):.6f}")
        examples.append((n, adv_a, adv_b, in_same, key_report))
    print(f"\n出力不一致 {n_out_diff}/{len(ns)} 回  入力不一致 {n_in_diff}/{len(ns)} 回")
    print("\n=== 揺れた指標キー (回数 / 最大差) ===")
    for k in sorted(diff_keys_count, key=lambda x: -diff_keys_count[x]):
        print(f"  {k}: {diff_keys_count[k]}回  最大差 {max_key_diff[k]:.6g}")
    print("\n=== 出力不一致の各リフレッシュ ===")
    for n, aa, ab, in_same, rep in examples[:30]:
        print(f"  n={n} adv {aa:+.4f} -> {ab:+.4f} (差 {abs(aa-ab):.4f}) "
              f"入力一致={in_same}")
        for r in rep[:8]:
            print(f"      {r}")


if __name__ == "__main__":
    main()
