"""新入力の起点残差を、D1の同じ画像窓・分類方法で比較する。"""
from __future__ import annotations

from collections import Counter
import gzip
import json
from typing import Any

from scripts._d1_count import cells, pair_candidates, UNKNOWN
from scripts._d1_summary import classify, SIMULATION
from scripts.report_e34b import directory, OUT
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_record import read_records


def observations(source: str) -> dict:
    """収集時に保存した多数決と原観測を読み、直前の確定盤面変更を添える。"""
    with gzip.open(OUT/'records'/f'{source}.windows.json.gz', 'rt', encoding='utf-8') as stream:
        windows = json.load(stream)
    game, changes, seen = None, [[], []], set()
    for record in read_records(OUT/'records'/f'{source}.jsonl.gz'):
        if record['kind'] != 'update':
            continue
        result, current = record['args'][0], record['args'][4]
        if current != game:
            game, changes = current, [[], []]
        for idx, side in enumerate((result.p1, result.p2)):
            board = side.confirmed_board
            if board is not None and (not changes[idx] or changes[idx][-1] != board._grid.tolist()):
                changes[idx].append(board._grid.tolist())
            event = side.chain_event
            if event is None:
                continue
            key = f'{game}:{idx}:{event.trigger_sec}'
            if key in seen:
                continue
            seen.add(key)
            windows[key]['changes'] = changes[idx][-2:]
    return windows


def difference(source: str, origin: dict, window: dict) -> dict:
    """最終組をD1の既存規則で除き、機構確定場面の時刻キーも維持する。"""
    board, snapshot = origin['board'], window['board']
    row = dict(source=source, game=origin['game'], side=origin['side'],
               trigger=origin['trigger_sec'], cells=[])
    if board is None or snapshot is None:
        row['unmeasurable'] = 'missing_origin' if board is None else window.get('reason')
        return row
    candidates = pair_candidates(window)
    pair = candidates[0] if len(candidates) == 1 else []
    if (source, row['trigger']) in SIMULATION:
        for previous in reversed(window['changes']):
            added = cells(previous, snapshot)
            if len(added) == 2 and all(previous[r][c] == 0 and 1 <= snapshot[r][c] <= 5 for r, c in added):
                pair = added
                break
    frames = [f for f in window['raw_frames'] if window['start_sec'] <= f['t_sec'] <= window['end_sec']]
    for r, c in cells(board, snapshot):
        cell = dict(row=r, col=c, origin=board[r][c], snapshot=snapshot[r][c],
            final_pair=(r, c) in pair, frames=len(frames),
            agreement=sum(f['cnn'][r][c] == f['hsv'][r][c] == snapshot[r][c] != UNKNOWN for f in frames))
        row['cells'].append(dict(cell, category=classify(row, cell)))
    return row


def residuals() -> dict:
    """両条件の全発火を保存し、起点欠測を残差ゼロとして数えない。"""
    result: dict[str, Any] = {v: [] for v in ('off', 'on')}
    for source in SOURCES:
        windows = observations(source)
        for variant in result:
            origins = json.loads((directory(variant, source)/'origin_audit.json').read_text())
            for origin in origins:
                key = f"{origin['game']}:{int(origin['side'] == '2P')}:{origin['trigger_sec']}"
                result[variant].append(difference(source, origin, windows[key]))
    summary = {}
    for variant, rows in result.items():
        counts = Counter(c['category'] for r in rows for c in r['cells'])
        summary[variant] = dict(fires=len(rows), measured=sum('unmeasurable' not in r for r in rows),
            missing=Counter(r['unmeasurable'] for r in rows if 'unmeasurable' in r), counts=counts,
            residual_cells=sum(n for k, n in counts.items() if k != '最終組（特定済み）'))
    common = paired_counts(result)
    save_json(OUT/'RESIDUAL_ROWS.json', result)
    summary['paired'] = common
    save_json(OUT/'RESIDUALS.json', summary)
    return summary


def paired_counts(result: dict) -> dict:
    """両条件で計数できる同じ発火の残差を併記し、欠測による減少を区別する。"""
    indexed = {v: {(r['source'], r['game'], r['side'], r['trigger']): r for r in rows}
               for v, rows in result.items()}
    assert indexed['off'].keys() == indexed['on'].keys()
    keys = [k for k in indexed['off'] if all('unmeasurable' not in indexed[v][k] for v in indexed)]
    counts = {v: Counter(c['category'] for k in keys for c in rows[k]['cells'])
              for v, rows in indexed.items()}
    return dict(fires=len(keys), total_fires=len(indexed['off']), counts=counts,
        residual_cells={v: sum(n for k, n in tally.items() if k != '最終組（特定済み）')
                        for v, tally in counts.items()})


if __name__ == '__main__':
    print(residuals())
