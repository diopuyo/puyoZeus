"""R1の固定母数評価で共有する原票・独立画像の読取り。"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Iterator
import numpy as np
from scripts._d3_inventory import decode
from scripts._d3_measure import supported
from src.board import COLOR_UNKNOWN

OUT = Path('logs/r1')
VISIBLE_ROW = 1
UNKNOWN = -1
REPORT_FPS = 30
CONTACT_LOOKBACK_SEC = 3.


def lines(path: Path) -> Iterator[dict]:
    """圧縮の有無だけを吸収し、欠測・破損行は例外にする。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def records(source: str) -> list[dict]:
    """ONをD3と同じ行順へ展開する。評価器や画像認識は動かさない。"""
    result = []
    for row in lines(OUT/'records'/f'{source}.jsonl.gz'):
        if row['kind'] != 'update':
            continue
        args = decode(row)['args']
        result.append(dict(t=args[3], game=args[4], formula=args[7],
                           sides=[args[0][side] for side in ('p1', 'p2')]))
    return result


def observations(source: str) -> dict[tuple[int, int], np.ndarray]:
    """全原フレームの単独一致だけを可視セルの基準値にする。"""
    result = {}
    for row in lines(OUT/'followup'/f'{source}.observations.jsonl.gz'):
        for side, obs in enumerate(row['sides']):
            cnn, hsv = np.array(obs['cnn']), np.array(obs['hsv'])
            grid = np.where((cnn == hsv) & (cnn != COLOR_UNKNOWN), cnn, UNKNOWN).astype(np.int8)
            grid[:VISIBLE_ROW] = UNKNOWN
            result[row['frame'], side] = grid
    return result


def contact(event: dict, obs: dict) -> float | None:
    """旧D3で対応した追加位置が原画像で初めて接地して見えた時刻を固定する。"""
    fps, side = event['fps'], event['side']
    start = max(event.get('cycle_start') or 0, event['write_t']-CONTACT_LOOKBACK_SEC)
    cells, expected = event['write_additions'], event['write_board']
    for frame in range(round(start*fps), round(event['write_t']*fps)+1):
        grid = obs.get((frame, side))
        if grid is not None and cells and all(grid[r,c] == expected[r][c]
                for r, c in cells) and supported(grid, cells):
            return frame/fps
    return None


def first_reflection(event: dict, rows: list[dict]) -> float | None:
    """同じ置き周期内の最初のSTABLE反映を、同じ追加座標・色で対応付ける。"""
    cells, expected = event['write_additions'], event['write_board']
    for row in rows[event['lower']:event['upper']+1]:
        side = row['sides'][event['side']]
        board = side['confirmed_board']
        if row['game'] != event['game'] or side['state']['state'] != 'STABLE' or board is None:
            continue
        if cells and all(board[r][c] == expected[r][c] for r, c in cells):
            return row['t']
    return None


def quantiles(values: list[float]) -> dict:
    """遅延を30fps換算のフレーム数で報告する。"""
    return dict(n=len(values), p50=float(np.percentile(values, 50)) if values else None,
                p95=float(np.percentile(values, 95)) if values else None)
