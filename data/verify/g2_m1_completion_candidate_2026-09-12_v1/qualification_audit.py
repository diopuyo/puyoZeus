"""保存scheduleを同run原票へ接続。物理分布/元画像の合格を代用しない。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import qualification_saved as Q
import second_pending_replay as P
import schedule_saved as S


def rows(path: Path) -> Iterator[dict]:
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def indexed(path: Path, key: Any, include: Any) -> dict:
    result = {}
    for value in rows(path):
        if not include(value): continue
        identity = key(value)
        S.require(identity not in result, 'duplicate_source_row')
        result[identity] = value
    return result


def second(states: list[dict], steps: dict, end: int) -> dict[int, tuple[str, ...]]:
    result = {}
    for mode in states:
        S.require(mode['error'] is None and mode['closed'] is True, 'second_mode_failure')
        current = P.timeline(mode, steps, end)
        S.require(not set(result).intersection(current), 'overlapping_second_basis')
        result.update(current)
    return result


def verify(root: Path) -> dict:
    order = S.verify(root)
    scheduled = list(rows(root / 'M1_CAPTURE_SCHEDULE.jsonl'))
    status = json.loads((root / 'BELIEF_M1_SESSION.json').read_text(encoding='utf-8'))
    S.require(all(status[k] is None for k in ('error', 'session_error', 'observer_error', 'witness_error',
                                             'evaluation_flags_error')), 'session_failure')
    start = min([S.EARLIEST[0]] + [m['initial']['state']['frame'] for m in status['modes']])
    journal = indexed(root / 'atomic_journal.jsonl', lambda r: (r['frame_idx'], r['side']),
        lambda r: r.get('kind') == 'step' and start <= r['frame_idx'] <= S.END)
    pending = second(status['modes'], {f: r for (f, side), r in journal.items() if side == '2P'}, S.END)
    context = indexed(root / 'provisional_context.jsonl', lambda r: r['frame_idx'],
                      lambda r: S.EARLIEST[0] <= r['frame_idx'] <= S.END)
    first = indexed(root / 'PROBABILISTIC_TRACKING.jsonl', lambda r: r['scope']['frame_idx'],
                    lambda r: S.EARLIEST[0] <= r['scope']['frame_idx'] <= S.END)
    compared = 0
    for item in scheduled:
        frame = item['frame']
        if frame < S.EARLIEST[0]:
            S.require(item['reasons'] == ['before_window'], 'prewindow_reason')
            continue
        steps = tuple(journal[(frame, side)] for side in Q.SIDES)
        derived = Q.reasons(context[frame], item['flags'], steps, first[frame], pending.get(frame))
        S.require(tuple(item['reasons']) == derived, 'qualification_reasons_mismatch')
        compared += 1
    return order | dict(saved_reasons_reconstructed=True, compared_frames=compared,
        qualification_proven=False, physical_replay_proven=False, quality_gate_clear=False,
        remaining='effect/grace捕捉来歴・1P到来/2P物理分布replay・画像検収は別ゲート')
