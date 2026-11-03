"""固定母数の残存379セルを実際のR1採否と独立観測へ対応する。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scripts._d3_inventory import SOURCES
from scripts.r1_measure_helpers import records, lines
from scripts._d4_loss import save

EPS = 1e-6
RESIDUAL = 379


def observation_subset(source: str, frames: set[int]) -> dict:
    """必要原フレームだけを保持する。"""
    path = Path('logs/r1/followup')/f'{source}.observations.jsonl.gz'
    return {r['frame']: r for r in lines(path) if r['frame'] in frames}


def select_attempt(event: dict, attempts: list[dict]) -> dict | None:
    """未来の再試行を使わず、同周期の直前合図を対応する。"""
    candidates = [r for r in attempts if r.get('side') == event['side'] and
        (event.get('cycle_start') or 0)-EPS <= r['t_sec'] <= event['t']+EPS]
    exact = [r for r in candidates if abs(r['t_sec']-event['t']) <= EPS]
    return (exact or candidates)[-1] if candidates else None


def category(cell: dict, attempt: dict | None, obs: dict, side: int) -> tuple[str, dict]:
    """実際の上位ゲートを優先し、分類不能を六分類へ押し込まない。"""
    if attempt is None:
        return '対応合図なし', {}
    reason = attempt['reason']
    base = dict(reason=reason, signal=attempt['signal'], signal_t=attempt['t_sec'],
                observed_frame=attempt.get('observed_frame'))
    if reason == 'already_consumed':
        return '周期内消費済み', base
    if reason in ('nonstable', 'chain_animation', 'erasing'):
        return '連鎖中・非STABLE', base
    if reason == 'difference_limit':
        return '上限超', base
    if reason == 'floating':
        return '浮き', base
    if attempt.get('quality') or attempt.get('previous_quality'):
        return '品質不良', base
    if attempt.get('previous_erasing'):
        return '連鎖中・非STABLE', base
    frame = attempt.get('observed_frame')
    if frame not in obs:
        return '観測欠測', base
    current = obs[frame]['sides'][side]
    r, c = cell['row'], cell['col']
    cnn, hsv = current['cnn'][r][c], current['hsv'][r][c]
    base.update(cnn=cnn, hsv=hsv)
    if cnn != hsv or cnn == 10:
        return 'CNN≠HSV', base
    previous = obs.get(frame-1, {}).get('sides', [None, None])[side]
    if previous is None or previous['cnn'][r][c] != cnn or previous['hsv'][r][c] != cnn:
        return '2フレーム不一致', base
    if cnn != cell['reference']:
        return '選択画像が基準と異なる', base
    fixed = any(v['row']==r and v['col']==c for v in attempt['corrections'])
    return ('修正後に残存・再変化' if fixed else '差分候補外'), base


def run(source: str, events: list[dict]) -> list[dict]:
    """D3と同じ行・座標・基準色を守り、ON残差だけを分類する。"""
    rows = records(source)
    attempts = list(lines(Path('logs/r1/capture')/source/'placement_signal_reconcile.jsonl'))
    pending, frames = [], set()
    for event in events:
        if event['source'] != source or event['status'] != '対応候補' or not event['early']:
            continue
        board = rows[event['i']]['sides'][event['side']]['confirmed_board']
        for cell in event['cells']:
            if board is not None and board[cell['row']][cell['col']] == cell['reference']:
                continue
            attempt = select_attempt(event, attempts)
            pending.append((event, cell, attempt))
            if attempt and attempt.get('observed_frame') is not None:
                frames.update((attempt['observed_frame'], attempt['observed_frame']-1))
    obs, result = observation_subset(source, frames), []
    for event, cell, attempt in pending:
        name, detail = category(cell, attempt, obs, event['side'])
        result.append(dict(source=source, t=event['t'], i=event['i'], side=event['side'],
            row=cell['row'], col=cell['col'], reference=cell['reference'], category=name,
            age=None if attempt is None else event['t']-attempt['t_sec'], **detail))
    print(source, len(result), dict(Counter(r['category'] for r in result)), flush=True)
    return result


def main() -> None:
    """排他的分類の合計と既報379セルを照合する。"""
    events = json.loads(Path('logs/d3/measurements.json').read_text())
    result = [r for source in SOURCES for r in run(source, events)]
    assert len(result) == RESIDUAL, len(result)
    save('residual_cells.json', result)
    save('residual_summary.json', dict(denominator=len(result),
        categories=dict(Counter(r['category'] for r in result)),
        exact_signal=sum(r['age'] is not None and abs(r['age'])<EPS for r in result)))


if __name__ == '__main__':
    main()
