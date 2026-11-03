"""入力記録を順方向に読み、開始証拠・外れ値・収集差分を調べる。"""
from __future__ import annotations

from collections import Counter
import json
from typing import Any

import numpy as np

from scripts._diag_dv2d_loss_inventory import OUT, BASE, read, save, resource
from scripts.falsedeath_replay_20261001 import records, input_record
from src.exchange_event_record import read_records

ZERO_FRAMES = 3
START_WINDOW = 5.0
FRAME_GAP_MAX = 0.1


def summary(result: Any, snapshot: Any, stamp: float) -> dict:
    """状態、得点、盤面、NEXTと落下累積をその時刻の値で保存する。"""
    sides = (result.p1, result.p2)
    return dict(t_sec=stamp, states=[s.state.name for s in sides],
        scores=[s.score for s in sides], cells=[int(np.count_nonzero(s.confirmed_board._grid))
        if s.confirmed_board is not None else None for s in sides],
        death_cells=[int(s.confirmed_board._grid[1, 2]) if s.confirmed_board is not None else None for s in sides],
        next=[s.next_pair for s in sides], motion=[s.next_slide_motion for s in sides],
        dropped=[snapshot.total_dropped_to_p1, snapshot.total_dropped_to_p2],
        confirmed_dead=list(getattr(result, 'confirmed_dead_sides', ())),
        chain_events=[None if s.chain_event is None else vars(s.chain_event) | {'before_board': None} for s in sides])


def scan(source: str, targets: list[dict], histories: dict) -> dict:
    """未来行を参照せず、境界の得点ゼロ継続と連続記録を確かめる。"""
    games, intervals, snapshots = {}, {}, {}
    previous, prior, death_seen = None, None, False
    for item in read_records(input_record('v2d', source)):
        if item['kind'] != 'update':
            continue
        result, snapshot, _, stamp, game, *_ = item['args']
        row = summary(result, snapshot, stamp)
        if game not in games:
            boundary = previous is not None and stamp-previous['t_sec'] <= FRAME_GAP_MAX
            games[game] = dict(game=game, first_sec=stamp, boundary_sec=stamp if boundary else None,
                boundary_scores=row['scores'], previous_scores=None if previous is None else previous['scores'],
                zero_run=0, zero_sec=None, zero_stable_sec=None, boundary_empty_sec=None,
                preceding_death=death_seen, keys=list(vars(result)))
            death_seen = False
        g = games[game]
        zeros = row['scores'] == [0, 0]
        g['zero_run'] = g['zero_run']+1 if zeros else 0
        in_window = stamp-g['first_sec'] <= START_WINDOW and g['boundary_sec'] is not None
        if in_window and g['zero_run'] >= ZERO_FRAMES and g['zero_sec'] is None:
            g['zero_sec'] = stamp
        if in_window and zeros and row['states'] == ['STABLE', 'STABLE'] and g['zero_stable_sec'] is None:
            g['zero_stable_sec'] = stamp
        if in_window and zeros and row['cells'] == [0, 0] and g['boundary_empty_sec'] is None:
            g['boundary_empty_sec'] = stamp
        for target in targets:
            key = f"{target['group']}_{target['game']}"
            start, end = target['first_sec'], target['candidate_observed_death_sec']
            if start-FRAME_GAP_MAX <= stamp <= end+FRAME_GAP_MAX:
                snapshots.setdefault(key, []).append(row)
                if prior is not None and start <= prior['t_sec'] < end:
                    duration = min(stamp, end)-max(prior['t_sec'], start)
                    counts = intervals.setdefault(key, Counter())
                    for idx, state in enumerate(prior['states']):
                        counts[f'{idx+1}P_{state}'] += duration
        death_seen = death_seen or bool(row['confirmed_dead'])
        previous, prior = row, row
    return dict(games=list(games.values()), intervals=intervals, observations=snapshots,
                history=histories[source]['games'])


def main() -> None:
    """176試合の同一母数を維持して、因果的な代替開始判定を保存する。"""
    save('RESOURCE_observations_v2.json', resource())
    targets = read(OUT/'EVENTS.json')
    histories = read(BASE/'v2d/HISTORY.json')['full']
    results = {}
    for source in records():
        relevant = [r for r in targets if r['record'] == source]
        results[source] = scan(source, relevant, histories)
        print(source, len(results[source]['games']), flush=True)
    save('OBSERVATIONS_v2.json', results)


if __name__ == '__main__':
    main()
