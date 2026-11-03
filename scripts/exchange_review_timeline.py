"""計算完了時刻で値を公開するフレーム時系列。B3表示値は保存値のまま。"""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Iterator
from scripts.exchange_review_io import pickles

SIDE_FIELDS = ('event_id', 'input_sec', 'seed_status', 'seed', 'f_return', 'f_coverage',
               'quick_policy', 'quick_return', 'quick_assumption', 'fast_ms', 'fast_ready_sec',
               'recursive_event_id', 'recursive_input_sec', 'recursive_candidate', 'recursive_policy',
               'recursive_balance', 'recursive_truncated', 'recursive_ms', 'recursive_ready_sec',
               'actual_event_id', 'actual_returned', 'actual_amount', 'actual_policy',
               'actual_consumption', 'actual_input_sec', 'landing', 'landing_event_id', 'landing_input_sec', 'reason')
FIELDS = ('frame_index', 't_sec', 'game_idx', 'b3_probability', 'b3_input_sec', 'b3_source',
          *(f'{s}_{k}' for s in ('1P', '2P') for k in SIDE_FIELDS))


def display_rows(path: Path, game: int, fps: float) -> list[dict]:
    """b3_updatesのrawではなく、実際にpublishされた平滑化済みprobabilityを保持する。"""
    by_frame = {}
    for row in pickles(path):
        if row.get('public_game', row['game']) != game:
            continue
        by_frame[row['frame']] = dict(t_sec=row['frame'] / fps,
            probability=row.get('probability'), source=row.get('source'))
    return sorted(by_frame.values(), key=lambda r: r['t_sec'])


def fast_values(event: dict) -> dict:
    foreground = event.get('foreground')
    base = dict.fromkeys(SIDE_FIELDS[:11]) | dict(event_id=event['event_id'], input_sec=event['input_sec'],
                seed_status=event['seed']['status'], seed=event['seed'].get('seed'), reason=event.get('reason'))
    if foreground is None:
        return base
    f, quick = foreground['f'], foreground['quick']
    assumptions = quick.get('assumptions', [quick])
    return base | dict(f_return=f['cancelled'] + f['excess'] if f['coverage'] else None,
        f_coverage=f['coverage'], quick_policy='/'.join(q['selected_policy'] for q in assumptions),
        quick_return=foreground['two_hand_return'],
        quick_assumption='/'.join(q['assumption'] for q in assumptions), fast_ms=foreground['compute_ms'],
        fast_ready_sec=event['fast_ready_sec'])


def publications(events: list[dict]) -> list[tuple[float, str, dict]]:
    changes = []
    for event in events:
        side = event['receiver']
        changes.append((event.get('fast_ready_sec', event['input_sec']), side, fast_values(event)))
        if 'background' in event:
            b = event['background']
            values = dict(recursive_event_id=event['event_id'], recursive_input_sec=event['input_sec'],
                recursive_candidate=json.dumps(b.get('candidate'), ensure_ascii=False),
                recursive_policy=b.get('policy'), recursive_balance=b.get('balance'),
                recursive_truncated=b['truncated'], recursive_ms=b['compute_ms'],
                recursive_ready_sec=event['background_ready_sec'])
            changes.append((event['background_ready_sec'], side, values))
        actual = event.get('actual', {})
        if actual.get('input_sec') is not None:
            values = {f'actual_{key}': actual.get(key) for key in
                      ('returned', 'amount', 'policy', 'consumption', 'input_sec')}
            values['actual_event_id'] = event['event_id']
            changes.append((actual['input_sec'], side, values))
        if actual.get('landing'):
            landing = actual['landing']
            changes.append((landing['input_sec'], side, dict(landing=landing['amount'], landing_event_id=event['event_id'],
                                                           landing_input_sec=landing['input_sec'])))
    return sorted(changes, key=lambda item: item[0])


def frame_rows(events: list[dict], display: list[dict], match: dict, fps: float) -> Iterator[dict]:
    start, end = match['start_sec'], match['end_sec']
    first, stop = math.ceil(start * fps), math.ceil(end * fps)
    changes, cursor = publications(events), 0
    state = dict.fromkeys(FIELDS)
    times = [r['t_sec'] for r in display]
    for frame in range(first, stop):
        t = frame / fps
        while cursor < len(changes) and changes[cursor][0] <= t:
            _, side, values = changes[cursor]
            stamp = values.get('recursive_input_sec', values.get('input_sec'))
            key = 'recursive_input_sec' if 'recursive_input_sec' in values else 'input_sec'
            previous = state.get(f'{side}_{key}')
            if stamp is None or previous is None or stamp >= previous:
                state.update({f'{side}_{key}': value for key, value in values.items()})
            cursor += 1
        index = bisect_right(times, t) - 1
        b3 = display[index] if index >= 0 else {}
        yield state | dict(frame_index=frame - first, t_sec=t, game_idx=match['game_idx'],
                           b3_probability=b3.get('probability'), b3_input_sec=b3.get('t_sec'),
                           b3_source=b3.get('source'))


def write_csv(path: Path, rows: Iterator[dict]) -> int:
    count = 0
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if v is None else json.dumps(v, ensure_ascii=False) for k, v in row.items()})
            count += 1
    return count


def read_csv(path: Path) -> Iterator[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            yield {k: json.loads(v) if v else None for k, v in row.items()}
