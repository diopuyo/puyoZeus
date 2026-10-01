"""入力の正しさの測定: 安定化した (手に持つ組, NEXT, NEXT2) が実際に置かれた組と合う割合 (Phase 4、門の値ではない)。

置いた組 P_k = 連続する2つの STABLE 区間の盤面差分が「色ぷよちょうど2個の追加・他は不変」のときの2色 (無順序)。
exev logs/next_shift/RESULT.md と同じ定義。フレーム単位で、Phase 3 の並べ方 (known_pairs) と比べる。
使い方: PYTHONPATH=. python -m scripts.prefire_stable_queue_accuracy_20261001
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.board import Board
from src.exchange_event_record import read_records
from src.prefire_stable_queue import SideQueue
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS, RECORDS

OUT = Path('logs/prefire_prediction/stable_queue_accuracy.json')
SLOTS = ('in_hand', 'next', 'next2')


def frames(source: str) -> list[list[tuple]]:
    """側ごとの STABLE フレーム列 (t, 盤面, 生 queue)。overlay._remember と同じ条件。"""
    out = [[], []]
    for row in read_records(RECORDS / f'{source}.jsonl.gz'):
        if row['kind'] != 'update':
            continue
        for idx, side in enumerate((row['args'][0].p1, row['args'][0].p2)):
            board = side.confirmed_board
            if getattr(getattr(side, 'state', None), 'value', None) != 'stable' or board is None:
                continue
            grid = (board if isinstance(board, Board) else Board.from_list(np.asarray(board).tolist()))._grid
            queue = np.array([*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))])
            out[idx].append((row['args'][3], grid.astype(np.int8).tobytes(), queue))
    return out


def placed(before: bytes, after: bytes) -> tuple[int, int] | None:
    """2盤面の差分が色ぷよちょうど2個の追加なら、その2色 (昇順)。"""
    a, b = (np.frombuffer(x, np.int8) for x in (before, after))
    changed = np.flatnonzero(a != b)
    if len(changed) != 2 or np.any(a[changed] != 0) or not np.all((b[changed] >= 1) & (b[changed] <= 5)):
        return None
    return tuple(sorted(int(v) for v in b[changed]))


def phase3_known(previous_last: np.ndarray | None, raw: np.ndarray) -> tuple[int, ...]:
    """Phase 3 の並べ方 (直前の別盤面の最終 next + 今の生 next/dnext)。比較用。"""
    head = tuple(int(v) for v in previous_last[:2]) if previous_last is not None else (0, 0)
    return (*head, *(int(v) for v in raw))


def evaluate_side(items: list[tuple], rule: str = 'stable') -> dict:
    """フレームごとの known を区間の実際の組と比べる (母数は正解が分かり、known が既知の枠)。"""
    queue, rows, boards, last, previous_last = SideQueue(), [], [], None, None
    for t, board, raw in items:
        if not boards or boards[-1] != board:
            boards.append(board)
            previous_last = last
        last = raw
        queue.observe(t, board, raw)
        known = queue.known() if rule == 'stable' else phase3_known(previous_last, raw)
        rows.append((len(boards) - 1, tuple(0 if v not in (1, 2, 3, 4, 5) else v for v in known)))
    truth = [placed(boards[i], boards[i + 1]) for i in range(len(boards) - 1)]
    score = {s: [0, 0] for s in SLOTS}
    for interval, known in rows:
        for slot, name in enumerate(SLOTS):
            k = interval + slot
            pair = tuple(sorted(known[2 * slot:2 * slot + 2]))
            if k >= len(truth) or truth[k] is None or 0 in pair:
                continue
            score[name][0] += pair == truth[k]
            score[name][1] += 1
    return score


def main() -> None:
    sides = {source: frames(source) for source in BASELINE_DIRS}
    out = {}
    for rule in ('stable', 'phase3'):
        total = {s: [0, 0] for s in SLOTS}
        for items_by_side in sides.values():
            for items in items_by_side:
                for name, (hit, n) in evaluate_side(items, rule).items():
                    total[name][0] += hit
                    total[name][1] += n
        out[rule] = {k: dict(hits=h, denominator=n, rate=h / n if n else None) for k, (h, n) in total.items()}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
