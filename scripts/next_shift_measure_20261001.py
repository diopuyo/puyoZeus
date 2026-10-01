"""148動画の学習行で「次に置く組 P_k / その次 P_{k+1}」がどの表示枠に出ているかを数える (診断のみ)。

各行 r (同一側・同一試合、時刻順) について
- 真値 P_k   = 行 r→r+1 の盤面差分が「追加ちょうど2セル・色ぷよのみ」のときの色の組 (無順序)
- 真値 P_k+1 = 行 r+1→r+2 も同じ条件のときの組
- 候補枠     = 今の行の next/dnext、1つ前・2つ前の行の next/dnext
を突き合わせ、入口の遷移 (place/chain/ojama/multi/start) ごとに一致数と母数を出す。
使い方: PYTHONPATH=. python -m scripts.next_shift_measure_20261001 > logs/next_shift_train/measure.json
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import sys

import numpy as np

from scripts.next_shift_common_20261001 import LEAN_ROOT, read_side_sequences
from src.next_queue_alignment import (
    correct_sequence, entry_kinds, placed_pairs, pair_key, valid_pair,
)

WORKERS = 8
LOOKBACK = 2


def slots(queues: np.ndarray, r: int) -> dict[str, tuple | None]:
    """行 r から LOOKBACK 行前までの next/dnext 枠 (無順序の組)。"""
    out: dict[str, tuple | None] = {}
    for back in range(LOOKBACK + 1):
        i = r - back
        for name, cut in (("next", slice(0, 2)), ("dnext", slice(2, 4))):
            pair = tuple(int(v) for v in queues[i, cut]) if i >= 0 else None
            out[f"b{back}.{name}"] = pair_key(*pair) if pair and valid_pair(pair) else None
    return out


def count_sequence(grids: np.ndarray, queues: np.ndarray, c: Counter) -> None:
    """1側1試合の行列で一致を数える。"""
    entries, exits = entry_kinds(grids), placed_pairs(grids)
    fixed, methods = correct_sequence(grids, queues)
    for r in range(len(grids)):
        c[f"rows|{entries[r]}|method{methods[r]}"] += 1
        pk = exits[r]
        if pk is None:
            continue
        pk1 = exits[r + 1] if r + 1 < len(grids) else None
        s = slots(queues, r)
        tag = entries[r]
        rule = slots(fixed, r)
        for key in ("all", tag, f"{tag}|method{methods[r]}"):
            c[f"{key}|rule_n"] += 1
            c[f"{key}|rule_next=Pk"] += int(rule["b0.next"] == pk)
            c[f"{key}|orig_next=Pk"] += int(s["b0.next"] == pk)
            if pk1 is not None:
                c[f"{key}|rule_n1"] += 1
                c[f"{key}|rule_both"] += int(rule["b0.next"] == pk and rule["b0.dnext"] == pk1)
                c[f"{key}|orig_both"] += int(s["b0.next"] == pk and s["b0.dnext"] == pk1)
        for key in ("all", tag):
            c[f"{key}|n"] += 1
            for name, pair in s.items():
                c[f"{key}|Pk={name}"] += int(pair == pk)
            if pk1 is not None:
                c[f"{key}|n1"] += 1
                for name, pair in s.items():
                    c[f"{key}|Pk1={name}"] += int(pair == pk1)
            if pk1 is not None and s["b0.next"] == pk and s["b0.dnext"] == pk1:
                c[f"{key}|both_pre_slide"] += 1
            if pk1 is not None and s["b0.next"] == pk1:
                c[f"{key}|slid_next=Pk1"] += 1


def video_counts(path_text: str) -> dict:
    """1動画分の Counter。"""
    c: Counter = Counter()
    for grids, queues, _ in read_side_sequences(path_text):
        count_sequence(grids, queues, c)
    return dict(c)


def main() -> None:
    """全動画を並列集計して JSON を標準出力へ。"""
    paths = sorted(str(p) for p in LEAN_ROOT.glob("*.npz"))
    total: Counter = Counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        for part in pool.map(video_counts, paths):
            total.update(part)
    json.dump(dict(videos=len(paths), counts=dict(sorted(total.items()))), sys.stdout,
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
