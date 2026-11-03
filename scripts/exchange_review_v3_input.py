"""v3専用。発火または連鎖終了時の観測だけから計算入力を作る。"""
from __future__ import annotations

import csv
import json
from bisect import bisect_right
from copy import deepcopy
from pathlib import Path
import numpy as np
from scripts.exchange_review_io import fires, grid, jsonl
from scripts.exchange_review_seed import PrefixSeeds, updates, extend_tapes
from src.b3_timing import TimingTable
from src.exchange_projection import hand_secs, hands_for
from src.board import COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN

CHAIN_STATES = ('CHAIN', 'GRAVITY_SETTLE')
SIDES = ('1P', '2P')
CELLS, COLS = 78, 6
DISPLAY_TTL_SEC = 3.


def board_text(board: list | str | None) -> str | None:
    if board is None:
        return None
    glyphs = {COLOR_EMPTY: '.', COLOR_OJAMA: 'o', COLOR_UNKNOWN: '?'}
    return ''.join(glyphs.get(c, str(c)) for c in grid(board))


def sanitize(data: dict) -> tuple[dict, dict]:
    """不使用色とUNKNOWNを探索用のおじゃまに置換し、元セルを保存する。"""
    data, audit = deepcopy(data), {}
    allowed = {COLOR_EMPTY, COLOR_OJAMA, *data['colors']}
    for key in ('receiver_board', 'attacker_board'):
        before = grid(data[key])
        mask = [c not in allowed for c in before]
        data[key] = [COLOR_OJAMA if m else c for c, m in zip(before, mask)]
        audit[key] = dict(original=board_text(before), mask=''.join('1' if m else '0' for m in mask),
                          replaced=sum(mask), sanitized=board_text(data[key]))
    return data, audit


class Inputs:
    """時刻付き原票。検索は指定時刻以前に限定する。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.shadow = list(jsonl(path / 'exchange_projection_shadow.jsonl'))
        self.fires = fires(self.shadow)
        self.updates = list(updates(path / 'inputs.jsonl.gz'))
        self.times = [r['t_sec'] for r in self.updates]
        with (path / 'review_data.csv').open(encoding='utf-8-sig') as stream:
            self.ledger = [dict(t_sec=float(r['t_sec']), game=int(r['game_idx']),
                pending=[int(float(r[f'{s}_incoming'] or 0)) for s in SIDES]) for r in csv.DictReader(stream)]
        # 半秒間隔のreviewより新しい台帳通知があれば、その通知まで反映する。
        self.ledger += [dict(t_sec=r['t_sec'], game=r['game_idx'], pending=[r['pending_p1'], r['pending_p2']])
                        for r in self.shadow if 'pending_p1' in r and 'game_idx' in r]
        self.ledger.sort(key=lambda r: r['t_sec'])
        self.ledger_times = [r['t_sec'] for r in self.ledger]

    def at(self, t: float, game: int) -> dict | None:
        index = bisect_right(self.times, t) - 1
        row = self.updates[index] if index >= 0 else None
        return row if row and row['game_idx'] == game else None

    def pending(self, t: float, game: int) -> dict:
        index = bisect_right(self.ledger_times, t) - 1
        row = self.ledger[index] if index >= 0 else None
        if row is None or row['game'] != game:
            raise ValueError(f'時刻{t}の得点台帳がありません')
        assert row['t_sec'] <= t
        return row

    def end_of_chain(self, fire: dict) -> dict | None:
        side = 1 - fire['exchange_input']['attacker']
        for row in self.updates[bisect_right(self.times, fire['t_sec']):]:
            if row['game_idx'] != fire['game_idx']:
                break
            current = row['sides'][side]
            if current['board'] is not None and (current['state'] == 'STABLE' or current['slide']):
                return row
        return None


def schedule(inputs: Inputs, game: int | None = None) -> list[dict]:
    result = []
    for fire in inputs.fires:
        if (game is not None and fire['game_idx'] != game) or not fire.get('exchange_input'):
            continue
        row = deepcopy(fire)
        busy = (row['exchange_input'].get('receiver_busy_sec', 0) > 0 or
                row.get('recompute', {}).get('receiver_state') in CHAIN_STATES)
        end = inputs.end_of_chain(row) if busy else None
        row['fire_sec'], row['deferred'] = row['t_sec'], busy
        row['end_observation'] = end
        if busy and end:
            row['t_sec'] = end['t_sec']
        result.append(row)
    return sorted(result, key=lambda r: (r['t_sec'], r['fire_sec']))


def resumed(row: dict, prefix: PrefixSeeds, seed: dict, inputs: Inputs) -> dict:
    """完走済み攻撃・相殺を足さず、台帳の残予告だけを次の探索へ渡す。"""
    data, end = row['exchange_input'], row['end_observation']
    attacker, receiver = data['attacker'], 1 - data['attacker']
    data['receiver_board'] = end['sides'][receiver]['board']
    pending = inputs.pending(row['t_sec'], row['game_idx'])
    data.update(receiver_busy_sec=0., receiver_credit=0, attack_counted=True,
        pending_receiver=pending['pending'][receiver], pending_attacker=pending['pending'][attacker],
        remaining_sec=max(0., data['remaining_sec'] - (row['t_sec'] - row['fire_sec'])))
    row['incoming'], row['pending_input_sec'] = data['pending_receiver'], pending['t_sec']
    row['evidence']['active_pair_included'] = False
    for side, key in ((receiver, 'known_receiver'), (attacker, 'known_attacker')):
        data[key] = [list(p) for p in prefix.tape.known_future(side)]
    if end['sides'][receiver]['slide']:
        data['known_receiver'] = [list(p) for p in prefix.tape.next_to_place(receiver)]
        row['evidence']['active_pair_included'] = len(data['known_receiver']) > len(prefix.tape.known_future(receiver))
    table = TimingTable.from_json(json.dumps(data['hand_table']))
    hand = hand_secs(table, np.array(grid(data['receiver_board'])).reshape(-1, COLS),
                     data.get('expected_hand_sec'))[0]
    row['hands_exp'] = hands_for(data['remaining_sec'], hand)
    return extend_tapes(seed, row)


def prepare_row(row: dict, prefix: PrefixSeeds, inputs: Inputs) -> tuple[dict, dict]:
    seed = prefix.at(row['t_sec'], row['game_idx'])
    assert seed.get('input_sec') is None or seed['input_sec'] <= row['t_sec']
    try:
        pending = inputs.pending(row['t_sec'], row['game_idx'])
    except ValueError:
        return row, seed | dict(reason='得点台帳未到達')
    if row['deferred']:
        if row['end_observation'] is None:
            return row, seed | dict(reason='連鎖終了未観測')
        assert row['end_observation']['t_sec'] == row['t_sec'] >= row['fire_sec']
        seed = resumed(row, prefix, seed, inputs)
    else:
        seed = extend_tapes(seed, row)
    if any(row['exchange_input'].get(k) is None for k in ('receiver_board', 'attacker_board')):
        return row, seed | dict(reason='保存入力盤面なし')
    data, audit = sanitize(row['exchange_input'])
    row['exchange_input'], row['board_audit'] = data, audit
    row.setdefault('pending_input_sec', pending['t_sec'])
    return row, seed
