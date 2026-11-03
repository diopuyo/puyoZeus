"""B4の実際の9要求を再構成し、探索回数と片側世代を診断だけで比較する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from scripts.run_live_pipeline_20260928 import save_json
from src.exchange_event_record import read_records

BASE = Path('logs/live_b4_realtime_verified')
OUTPUT = Path('logs/live_b5_mc')
ROLLOUTS = (60, 30, 15)
PERCENTILES = (50, 95, 99)
MILLISECONDS = 1000.0


def distribution(values: list[float]) -> dict:
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
        **{f'P{p}': float(np.percentile(values, p)) if values else None for p in PERCENTILES})


def observations() -> list[dict]:
    """元ライブ経路と同じSTABLE時の凍結盤面を復元する。"""
    rows, boards, game = [], [None, None], None
    for record in read_records(BASE / 'inputs.jsonl.gz'):
        if record['kind'] != 'update':
            continue
        result, _, _, stamp, current_game, *_ = record['args']
        if current_game != game:
            boards, game = [None, None], current_game
        sides = (result.p1, result.p2)
        for index, side in enumerate(sides):
            if side.state.name == 'STABLE' and side.confirmed_board is not None:
                boards[index] = side.confirmed_board.copy()
        rows.append(dict(t_sec=stamp, game=game, result=result, boards=tuple(boards),
            context=tuple(None if side.confirmed_board is None else side.confirmed_board.grid_bytes()
                          for side in sides), next=tuple(side.next_pair for side in sides)))
    return rows


def request_indices(rows: list[dict]) -> list[int]:
    evaluations = json.loads((BASE / 'evaluations.json').read_text())
    stamps = {r['counter_search']['request_time'] for r in evaluations
              if r.get('counter_search', {}).get('request_time') is not None}
    lookup = {row['t_sec']: index for index, row in enumerate(rows)}
    return [lookup[stamp] for stamp in sorted(stamps)]


def benchmark(row: dict, rollouts: int) -> dict:
    """B4と同じ両側・既知NEXT・固定閾値のMCをキャッシュなしで計測する。"""
    from scripts.visualize_advantage_overlay import (
        _detect_chain_attacker, _resolve_counter_time_budget, COUNTER_THRESHOLD_OJAMA)
    from scripts.mc_counter_estimator import estimate_counter_distribution
    result = row['result']
    attack = _detect_chain_attacker(result.p1, result.p2, row['t_sec'])
    budget = _resolve_counter_time_budget(attack, row['t_sec'], False, {})
    probabilities, durations = [], []
    for board, pair in zip(row['boards'], row['next']):
        known = (pair,) if pair and all(color > 0 for color in pair) else ()
        start = time.perf_counter()
        answer = estimate_counter_distribution(board, budget, known_pairs=known,
                    thresholds_ojama=(COUNTER_THRESHOLD_OJAMA,), n_rollouts=rollouts)
        durations.append(time.perf_counter()-start)
        probabilities.append(float(answer.prob_at_least[COUNTER_THRESHOLD_OJAMA]))
    return dict(t_sec=row['t_sec'], budget_sec=budget, rollouts=rollouts,
                probabilities=probabilities, side_seconds=durations, seconds=sum(durations))


def admissible(rows: list[dict], index: int, measured: dict) -> dict:
    """同じ完了時刻を使い、盤面世代だけを両側→片側に変えた反実仮想。"""
    start = rows[index]
    done = start['t_sec'] + measured['seconds']
    end = next((i for i in range(index+1, len(rows)) if rows[i]['t_sec'] >= done), None)
    if end is None:
        return dict(both=False, sides=[False, False], reason='outside_window')
    path = rows[index+1:end+1]
    valid_time = rows[end]['t_sec'] < start['t_sec']+measured['budget_sec']
    valid_game = all(row['game'] == start['game'] for row in path)
    generations = [valid_game and all(
        row['context'][side] == start['context'][side]
        and row['next'][side] == start['next'][side] for row in path) for side in range(2)]
    sides = [valid_time and value for value in generations]
    return dict(both=all(sides), sides=sides, completion_t_sec=rows[end]['t_sec'],
                expired=not valid_time, side_generation_unchanged=generations)


def update_intervals(rows: list[dict]) -> dict:
    """確定盤面内容の更新間隔と、Noneへの遷移も含むB4世代寿命を区別する。"""
    values = {}
    for mode in ('confirmed', 'context'):
        for side in range(2):
            previous, last_time, last_game, intervals = None, None, None, []
            for row in rows:
                board = row['boards'][side]
                key = row['context'][side] if mode == 'context' else (
                    None if board is None else board.grid_bytes())
                if row['game'] != last_game:
                    previous, last_time, last_game = key, row['t_sec'], row['game']
                elif key != previous:
                    intervals.append((row['t_sec']-last_time)*MILLISECONDS)
                    previous, last_time = key, row['t_sec']
            values[f'{mode}_{side+1}p_ms'] = distribution(intervals)
    return values


def summarize(rows: list[dict], results: list[dict]) -> dict:
    summaries = {}
    reference = {row['t_sec']: np.array(row['probabilities']) for row in results if row['rollouts'] == ROLLOUTS[0]}
    for rollouts in ROLLOUTS:
        subset = [row for row in results if row['rollouts'] == rollouts]
        errors = [abs(p-reference[row['t_sec']][side]) for row in subset
                  for side, p in enumerate(row['probabilities'])]
        summaries[str(rollouts)] = dict(job_ms=distribution([r['seconds']*MILLISECONDS for r in subset]),
            side_ms=distribution([v*MILLISECONDS for r in subset for v in r['side_seconds']]),
            both_accepted=sum(r['admission']['both'] for r in subset), requests=len(subset),
            side_accepted=sum(sum(r['admission']['sides']) for r in subset), side_results=2*len(subset),
            expired_requests=sum(r['admission'].get('expired', False) for r in subset),
            probability_absolute_error=distribution(errors))
    return dict(summaries=summaries, update_intervals=update_intervals(rows), results=results,
        note='B4実要求のオフライン反実仮想。両側MCの合計所要で次の通知時点を判定。起動・IPC待ちは含まない。'
             '片側採用はその側の確率のみ有効で、両側が揃わない合成優位は採用できない。')


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, results = observations(), []
    load = os.getloadavg()
    for index in request_indices(rows):
        for rollouts in ROLLOUTS:
            measured = benchmark(rows[index], rollouts)
            measured['admission'] = admissible(rows, index, measured)
            results.append(measured)
            save_json(OUTPUT / 'progress.json', results)
    report = summarize(rows, results)
    report.update(loadavg_start=load, loadavg_end=os.getloadavg(), nice=os.nice(0))
    save_json(OUTPUT / 'report.json', report)


if __name__ == '__main__':
    main()
