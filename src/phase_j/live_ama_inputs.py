"""ama v7.1の因果的入力。連鎖前STABLE盤・引継ぎツモ番号を独立所有する。"""
from __future__ import annotations
from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any
from src.ama_evaluation_v7_projection import Projection
from src.ama_evaluation_v7_tape import CausalTape
from src.ama_tsumo_seed_v5 import load_table
from src.ama_plain import validate_request
from src.b3_score_ledger import ScoreLedger
from .live_ama_engine import request_from_saved

STABLE_INTERVAL_SEC = 1.
NOMINAL_FPS = 30


def queue_only(raw: dict) -> tuple:
    def pair(name: str) -> Any:
        value = raw.get(name)
        return value.get('tuple') if isinstance(value, dict) else value
    return pair('next_pair'), pair('dnext_pair'), bool(raw.get('next_slide_motion'))


class AmaInputs:
    def __init__(self, native: Any, table: Any) -> None:
        self.projection = Projection(native, stable_origin=True)
        self.tape = CausalTape(table, inherit_counter=True)
        if os.environ.get('PUYO_OBS_EXACT_SEED') == '1':
            from .live_ama_seed import identify_exact
            self.tape.identify = identify_exact
        self.ledger = ScoreLedger(newest_first_offset=True, enable_inflight=False, observed_landing_rule=False)
        self.colors: set[int] = set()
        self.index = 0
        self.previous = None
        self.last_requested = float('-inf')
        self.missing_reason: str | None = None

    def book(self, packet: dict) -> dict | None:
        stamp, game, raw = packet['t'], packet['game'], packet['raw']
        sides = [dict(side=('1P', '2P')[i], state=r['state']['state'], score=r.get('score'),
                      confirmed_board=r['confirmed_board']['board']['array'] if r.get('confirmed_board') else None,
                      next_slide_motion=r.get('next_slide_motion', False)) for i, r in enumerate(raw)]
        placed = tuple(s['side'] for s in sides if self.ledger.core.origin_sec is None and s['next_slide_motion'])
        snap = self.ledger.update(dict(t_sec=stamp, frame=round(stamp*NOMINAL_FPS), game_idx=game,
                                       match_active=True, sides=sides), placed)
        self.missing_reason = ('得点未確定・連続観測待ち' if snap is None or snap['history_waiting'] else
                               '得点履歴不足・次試合待ち' if snap['history_incomplete'] else None)
        if snap is None:
            return None
        values = {key: [snap[f'{key}_p{i+1}'] for i in (0, 1)]
                  for key in ('pending', 'generated', 'dropped', 'carry', 'score', 'pending_confirmed')}
        values = {k: [v if v is not None and math.isfinite(v) else None for v in vals] for k, vals in values.items()}
        return dict(t=stamp, game=game, **values, origin=snap['margin_origin_sec'], rate=snap['rate'])

    def observe_tape(self, raw: list, stamp: float) -> None:
        states = [r['state']['state'] for r in raw]
        triggers = [((r.get('chain_event') or {}).get('namespace') or {}).get('trigger_sec') for r in raw]
        for side, record in enumerate(self.projection.active):
            if record and record['kind'] == 'chain':
                states[side], triggers[side] = 'CHAIN', record['trigger']
        self.tape.observe(stamp, [queue_only(r) for r in raw], states, triggers)

    def observe(self, packet: dict) -> dict | None:
        stamp, raw = packet['t'], packet['raw']
        book = self.book(packet)
        if book is None or any(p is None for p in book['pending']):
            self.observe_tape(raw, stamp)
            return None
        before = deepcopy(self.projection.stable_boards)
        changed = self.projection.observe(raw, stamp, book)
        self.observe_tape(raw, stamp)
        self.colors.update(c for s in self.tape.tape.sides for pair in s.own for c in pair)
        states = [r['state']['state'] for r in raw]
        if states == ['CHAIN', 'CHAIN'] or 'MENU' in states:
            return None
        signature = repr((self.projection.boards, book['pending'], book['carry'], book['rate'],
                          [self.tape.tape.counter(s) for s in (0, 1)]))
        periodic = states == ['STABLE','STABLE'] and stamp-self.last_requested >= STABLE_INTERVAL_SEC
        if not changed and before == self.projection.stable_boards and signature == self.previous and not periodic:
            return None
        self.previous = signature
        self.last_requested = stamp
        item = self.request(packet['game'], stamp, book)
        self.index += 1
        return item

    def request(self, game: int, stamp: float, book: dict) -> dict:
        snap = self.projection.snapshot(book)
        origin = book['origin'] if book['origin'] is not None else stamp
        req = dict(game=game, generation=self.index, input_sec=stamp,
                   evidence_times=snap['evidence_times']+[self.tape.last], boards=snap['boards'], known=[],
                   colors=sorted(self.colors), pending=snap['pending'], elapsed_sec=stamp-origin,
                   stable=[True, True], seed=20261017+game)
        item = dict(index=self.index, request=req, carry=snap['carry'], projection=snap,
                    states=list(self.projection.states), source_kind='v71_causal', all_clear=snap['all_clear'])
        try:
            if any(b is None for b in snap['boards']) or len(self.colors) != 4:
                raise ValueError('盤面または4色未確定')
            bundle = self.tape.bundle(sorted(self.colors))
            req['known'] = bundle['input_known']
            item['bundle'] = bundle
            validate_request(request_from_saved(req))
            if len(item['carry']) != 2 or any(c is None or c < 0 for c in item['carry']):
                raise ValueError('保存端数が必要')
        except ValueError as error:
            item['missing'] = str(error)
        for state, kind in zip(item['states'], snap['kinds']):
            if state in ('CHAIN', 'OJAMA_FALL') and kind == 'held':
                item['missing'] = '発火直前STABLE盤なし・予測不可'
        assert max(req['evidence_times']) <= stamp
        return item
