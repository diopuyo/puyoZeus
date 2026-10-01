"""5A診断行を母数に、両側それぞれの読みと欠測条件を分解記録する。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.run_prefire_replay_20260930 import BASELINE_DIRS, RECORDS
from src.exchange_event_record import read_records
from src.prefire_stable_queue import SideQueue, PLAYABLE, STABLE_FRAMES

OUT = Path('logs/prefire_prediction/v5b/before')
OLD = Path('logs/prefire_prediction/replay/v5_L03')


def reasons(queue: SideQueue, board: bytes | None) -> list[str]:
    """理由は重複を許す。見えない画素と記録の未読は区別できないため断定しない。"""
    result = []
    if board is None:
        result.append('board_absent')
    elif 10 in board:
        result.append('board_unknown')
    interval = queue.interval
    if interval is None or interval.in_hand is None:
        result.append('in_hand_unassigned')
    if queue.run is None:
        return result + ['reading_absent']
    for name, pair in zip(('next', 'next2'), (queue.run[:2], queue.run[2:])):
        if not all(c in PLAYABLE for c in pair):
            result.append(name + '_unread_in_record')
    if queue.run_count < STABLE_FRAMES:
        result.append('less_than_three_identical_frames')
    if interval is not None and not interval.adopted:
        result.append('no_adopted_reading')
    elif interval is not None and not interval.promoted():
        result.append('before_promotion_next2_unavailable')
    return result


def measure(source: str) -> dict:
    """元の診断時刻へ合わせ、状態コードと観測条件の側別クロス集計を保存する。"""
    saved = np.load(OLD/source/'prefire_trace.npz')
    columns = list(saved['columns'])
    rows = saved['values']
    times = {round(float(r[0]), 6): r for r in rows}
    queues, boards, game = [SideQueue(), SideQueue()], [None, None], None
    counts, missing, matched = [Counter(), Counter()], [Counter(), Counter()], 0
    for row in read_records(RECORDS/f'{source}.jsonl.gz'):
        if row['kind'] != 'update':
            continue
        result, _, _, stamp, current = row['args'][:5]
        if game != current:
            queues, boards, game = [SideQueue(), SideQueue()], [None, None], current
        for side, data in enumerate((result.p1, result.p2)):
            if data.state.value == 'stable' and data.confirmed_board is not None:
                grid = getattr(data.confirmed_board, '_grid', data.confirmed_board)
                boards[side] = np.asarray(grid, dtype=np.int8).tobytes()
                queues[side].observe(stamp, boards[side], np.array([
                    *(data.next_pair or (0, 0)), *(data.dnext_pair or (0, 0))]))
        previous = times.get(round(float(stamp), 6))
        if previous is None:
            continue
        matched += 1
        for side in (0, 1):
            flags = reasons(queues[side], boards[side])
            counts[side].update(flags)
            code = int(previous[columns.index(f'status_{side+1}p')])
            if code == 1:
                missing[side].update(['denominator', *flags])
    return dict(source=source, denominator=len(rows), matched=matched,
                all_rows_by_side=counts, missing_rows_by_side=missing)


def main() -> None:
    """旧原票を変更せず、5記録の理由別母数を別ファイルへ保存する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    for source in BASELINE_DIRS:
        path = OUT/f'missing_{source}.json'
        if not path.exists():
            result = measure(source)
            path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(result, flush=True)


if __name__ == '__main__':
    main()
