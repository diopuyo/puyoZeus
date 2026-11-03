"""原票を全フレーム時系列と切替表にまとめる。未来の評価値は参照しない。"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import csv
import json
from typing import Any
import numpy as np
from scripts._diag_review1003b_extract import ROOT, OUT, CASES, resources, save
from src.exchange_event_record import decode


def latest(values: list[dict], stamp: float) -> dict[str, Any]:
    """同時刻は記録末尾を優先し、それ以前の値だけを返す。"""
    idx = bisect_right([v['t_sec'] for v in values], stamp)-1
    return values[idx] if idx >= 0 else {}


def frame_input(row: dict) -> dict[str, Any]:
    """両者盤面を最上段から13行の色コードで記録する。"""
    result = []
    args = row['args']
    for side in (args[0].p1, args[0].p2):
        board = side.confirmed_board
        grid = None if board is None else board._grid
        result.append(dict(state=side.state.name, board=None if grid is None else grid.tolist(),
            colors=None if grid is None else int(np.count_nonzero((grid >= 1) & (grid <= 5))),
            queue=[*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))],
            next_pair=side.next_pair, dnext_pair=side.dnext_pair, score=side.score,
            chain_event=None if side.chain_event is None else {k: getattr(side.chain_event, k)
            for k in ('trigger_sec', 'mechanism', 'chain_count', 'total_score')}))
    return dict(t_sec=row['t_sec'], sides=result, formula_totals=list(args[5]),
                displayed_scores=list(args[6]), formula_visible=list(args[7]))


def build(case: tuple) -> dict[str, Any]:
    """全フレームへ直近評価と入力を付け、切替点を集計する。"""
    number, set_no, game, start, end = case
    frames = list(csv.DictReader((OUT / f'case{number}_frames.csv').open()))
    events = json.loads((OUT / f'case{number}_events.json').read_text())
    inputs = decode(json.loads((OUT / f'set{set_no}_game{game}_inputs.json').read_text()))
    inputs = [frame_input(r) for r in inputs if start <= r['t_sec'] <= end]
    lands = sorted([v for e in events for v in e['values'] if 'incoming' in v], key=lambda v: v['t_sec'])
    s3 = sorted([v for e in events for v in e['values'] if 'score_totals' in v], key=lambda v: v['t_sec'])
    full, transitions, previous = [], [], None
    for frame in frames:
        stamp = float(frame['t_sec'])
        land, score, inp = latest(lands, stamp), latest(s3, stamp), latest(inputs, stamp)
        reasons = [r['reason'] for r in land.get('post_counter_bound', [])]
        key = (frame['new_evaluator'], tuple(reasons))
        if key != previous:
            transitions.append(dict(t_sec=stamp, evaluator=key[0], reasons=reasons))
            previous = key
        full.append(dict(frame=frame, input=inp, landing=land, score_prediction=score))
    with (OUT / f'case{number}_timeline.jsonl').open('w') as stream:
        for row in full:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    reason_counts = Counter(tuple(r['reason'] for r in v['landing'].get('post_counter_bound', [])) for v in full)
    summary = dict(case=number, frames=len(frames), first=float(frames[0]['t_sec']),
        last=float(frames[-1]['t_sec']), evaluators=dict(Counter(r['new_evaluator'] for r in frames)),
        old_evaluators=dict(Counter(r['old_evaluator'] for r in frames)), transitions=transitions,
        reasons=[dict(reasons=list(k), frames=v) for k, v in reason_counts.items()],
        history_blocks=sum(v['landing'].get('history_integrity', {}).get('blocked', False) for v in full))
    save(f'case{number}_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    """全4件の時系列を出力する。"""
    resources()
    save('summaries.json', [build(case) for case in CASES])


if __name__ == '__main__':
    main()
