"""D1の可視差分と最後の着地候補を保存画像窓から列挙する。"""
from __future__ import annotations
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any
from scripts._d1_inventory import OUT, SOURCES

ROWS, COLS, UNKNOWN, MIN_FRAMES = 13, 6, 10, 3


def cells(a: list, b: list) -> list[tuple[int, int]]:
    """隠し段を除いた不一致座標を返す。"""
    return [(r, c) for r in range(1, ROWS) for c in range(COLS) if a[r][c] != b[r][c]]


def pair_candidates(window: dict) -> list[list]:
    """着地多数決盤面との差が2セル追加だけとなる先行画像を探す。"""
    board = window['board']
    if board is None:
        return []
    groups: dict[tuple, int] = Counter()
    for frame in window['raw_frames']:
        if frame['t_sec'] >= window['start_sec']:
            continue
        before = frame['cnn']
        diff = cells(before, board)
        if len(diff) == 2 and all(before[r][c] == 0 and 1 <= board[r][c] <= 5 for r, c in diff):
            groups[tuple(diff)] += 1
    return [list(k) for k, n in groups.items() if n >= MIN_FRAMES]


def analyze(source: str) -> list[dict]:
    """起点との差を、観測一致・最終組・計数不能に分けて保持する。"""
    inventory = json.loads((OUT/f'{source}_inventory.json').read_text())
    fires = {(f['game'], int(f['side'] == 'p2'), f['chain_event']['trigger_sec']): f for f in inventory['fires']}
    with gzip.open(f'logs/e31/records/{source}.jsonl.windows.json.gz', 'rt') as stream:
        windows = json.load(stream)
    result = []
    for window in windows:
        key = (window['game'], window['side'], window['trigger_sec'])
        fire = fires[key]
        origin, board = fire['origin'], window['board']
        row = dict(source=source, game=key[0], side=key[1], trigger=key[2], reason=window['reason'], cells=[])
        if origin is None or board is None:
            row['unmeasurable'] = 'missing_origin' if origin is None else window['reason']
            result.append(row)
            continue
        candidates = pair_candidates(window)
        pair = candidates[0] if len(candidates) == 1 else []
        row.update(pair_candidates=candidates, raw_difference=len(cells(origin, board)), pair_confirmed=bool(pair))
        prior = [x for x in inventory['changes'] if x['confirmed_board'] is not None and x['game'] == key[0] and x['side'] == fire['side'] and x['t'] <= key[2]]
        for r, c in cells(origin, board):
            frames = [f for f in window['raw_frames'] if window['start_sec'] <= f['t_sec'] <= window['end_sec']]
            agreement = sum(f['cnn'][r][c] == f['hsv'][r][c] == board[r][c] != UNKNOWN for f in frames)
            writes = [x for i, x in enumerate(prior) if i == 0 or x['confirmed_board'][r][c] != prior[i-1]['confirmed_board'][r][c]]
            last = writes[-1] if writes else None
            row['cells'].append(dict(row=r, col=c, origin=origin[r][c], snapshot=board[r][c], final_pair=(r,c) in pair,
                agreement=agreement, frames=len(frames), last_change_sec=last['t'] if last else None,
                last_change_state=last['state']['state'] if last else None))
        result.append(row)
    return result


def main() -> None:
    """発火単位の母数と差分セルを保存する。"""
    rows = [row for source in SOURCES for row in analyze(source)]
    (OUT/'differences.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    for source in SOURCES:
        group = [r for r in rows if r['source'] == source]
        measured = [r for r in group if 'unmeasurable' not in r]
        cs = [c for r in measured for c in r['cells']]
        print(source, dict(fires=len(group), measured=len(measured), diff=len(cs), pair=sum(c['final_pair'] for c in cs),
            pair_fires=sum(r['pair_confirmed'] for r in measured), consensus=sum(c['agreement']*2>c['frames'] for c in cs)), flush=True)
        for row in measured:
            if row['raw_difference'] > 4:
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
