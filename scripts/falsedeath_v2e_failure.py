"""事前登録①を落とした既存原票だけを抽出し、条件を変更せず報告する。"""
from __future__ import annotations

import json

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT
from scripts._diag_set2_falsedeath_audit import events

DEST = OUT/'v2e'
SOURCE, GAME, SIDE = 'mia8KCjr52g', 7, '2P'
FIRST_SEC = 691.4333333333333
CERTAIN = ('unavoidable_death', 'confirmed_death')
TIME_TOLERANCE = 1e-6


def extract(mode: str) -> dict:
    """同じ試合・同じ初回時刻の値と、誤確定の表示時刻を保存する。"""
    directory = OUT/'replay'/mode/SOURCE
    records = [e for e in events(directory/'events.jsonl') if e['game_idx'] == GAME]
    values = [v for e in records for v in e['values'] if abs(v['t_sec']-FIRST_SEC) < TIME_TOLERANCE]
    history = json.loads((directory/'history_integrity_audit.json').read_text())
    game = next(r for r in history['games'] if r['game'] == GAME)
    with np.load(directory/'display.npz') as display:
        wrong = ((display['game_idx'] == GAME) & np.isin(display['source'], CERTAIN)
                 & (display['display_p1'] > .5))
        stamps = display['t_sec'][wrong]
    return dict(at_first_sec=values, game_history=game, wrong_frame_times=stamps.tolist(),
        false_frames=len(stamps), first_display_sec=float(stamps[0]) if len(stamps) else None,
        last_display_sec=float(stamps[-1]) if len(stamps) else None)


def main() -> None:
    """基準と候補の同時刻原票を切り出し、追記報告だけを作る。"""
    results = {mode: extract(mode) for mode in ('v2d', 'v2e')}
    terminals = json.loads((OUT/'TERMINALS.json').read_text())['all'][SOURCE]
    value = dict(source=SOURCE, game=GAME, side=SIDE, first_sec=FIRST_SEC,
        terminals=[r for r in terminals if r['game'] == GAME], results=results)
    with (DEST/'FALSE_CASE.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
    print(json.dumps({mode: {k: v for k, v in row.items()
        if k in ('false_frames', 'first_display_sec', 'last_display_sec')} for mode, row in results.items()}))


if __name__ == '__main__':
    main()
