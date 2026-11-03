"""到着済みSTABLEの経過を採取する。採点ラベルを生成・参照しない。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from scripts.audit_exchange_counts_v33 import OUT as V33
from scripts.audit_exchange_cells_v34 import OUT
from scripts.exchange_review_v32_data import BASE, VIDEOS, source, fire_rows
from scripts.exchange_review_io import save
from scripts.replay_exchange_projection_shadow import updates

HISTORY_SIZE, SIDES = 4, 2


def observe(histories: list[list], raw: object, side: int, stamp: float) -> None:
    if raw.state.name != 'STABLE' or raw.confirmed_board is None:
        return
    board = raw.confirmed_board._grid.ravel().tolist()
    history = histories[side]
    pair = list(raw.next_pair) if raw.next_pair else None
    if history and board == history[-1]['board'] and pair == history[-1]['next']:
        return
    history.append(dict(t=stamp, board=board, next=pair))


def collect(video: int) -> None:
    path = OUT/f'history_{video}.json'
    if path.exists():
        return
    if video == 80:
        items = json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())['origins']
        targets = {(7, r['fire']['t_sec']): dict(event=r['event_id'], side=r['side'],
            trigger=r['fire']['trigger_sec']) for r in items}
    else:
        items = json.loads((V33/f'calibration/input_{video}.json').read_text())
        fires = {r['event_id']: r for r in fire_rows(source(video))}
        targets = {(r['game'], r['input_sec']): dict(event=r['event'], side=r['side'],
            trigger=fires[r['event']]['trigger_sec']) for r in items}
    histories, game, result = [[], []], None, []
    for args in updates(source(video)/'inputs.jsonl.gz', {7} if video == 80 else None):
        if game != args[4]:
            histories, game = [[], []], args[4]
        for side, raw in enumerate((args[0].p1, args[0].p2)):
            observe(histories, raw, side, args[3])
        target = targets.get((game, args[3]))
        if target is None:
            continue
        history = [h for h in histories[target['side']] if h['t'] <= target['trigger']]
        hidden = [dict(Counter(h['board'][cell] for h in history)) for cell in range(6)]
        result.append(dict(event=target['event'], game=game, input_sec=args[3],
            history=deepcopy(history[-HISTORY_SIZE:]), hidden_history=hidden,
            history_latest=max((h['t'] for h in history), default=None)))
    save(path, result)
    print('履歴', video, len(result), flush=True)


if __name__ == '__main__':
    for selected in (*VIDEOS, 80):
        collect(selected)
