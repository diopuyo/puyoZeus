"""同一プロセス内で「同一入力 → 同一出力」が成り立つかをトレース1本で検査する。

_score_advantage (full_row) の出力は (b1, b2, snap.net_balance_capped,
snap.forecast_p1, snap.forecast_p2) のみで決まるはず (コード上)。
トレースの score レコードをこの入力タプルでグループ化し、グループ内に
異なる adv が混在すれば「プロセス内非決定」(ハッシュシード説は棄却)。
全グループで一意なら「プロセスごとの定数」(シード/初期化順の説が残る)。

使い方:
  python scripts/_diag_adv_nondet_inprocess_check_2026-08-25.py trace_r1.jsonl
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
    groups: dict[tuple, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] != "score":
                continue
            key = (rec["crc1"], rec["crc2"],
                   rec["snap"].get("net_balance_capped"),
                   rec["snap"].get("forecast_p1"), rec["snap"].get("forecast_p2"))
            groups[key].append((rec["n"], hf(rec["snap"].get("t_sec", "nan")),
                                rec["adv"]))
    n_multi = 0
    n_incons = 0
    for key, lst in groups.items():
        if len(lst) < 2:
            continue
        n_multi += 1
        advs = {adv for _, _, adv in lst}
        if len(advs) > 1:
            n_incons += 1
            if n_incons <= 15:
                print(f"[プロセス内不一致] crc1={key[0]} crc2={key[1]} "
                      f"net={hf(key[2]) if key[2] else None} "
                      f"呼出{len(lst)}回 相異なるadv={len(advs)}種")
                for n, ts, adv in lst[:10]:
                    print(f"    n={n} t={ts:.3f} adv={hf(adv):+.6f}")
    print(f"\n複数回呼ばれた同一入力グループ: {n_multi}個")
    print(f"うち出力が食い違ったグループ: {n_incons}個")
    print("=> " + ("プロセス内非決定 (ハッシュシード説は棄却方向)" if n_incons
                   else "プロセス内は決定的 (プロセス定数: シード/初期化順が候補)"))


if __name__ == "__main__":
    main()
