"""画像が生存する認識側で、v4.9f専用の因果的入力を作る。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.exchange_count_ledger import ScorePrefix
from src.exchange_event_record import encode, fields, SIDE_FIELDS, CHAIN_FIELDS
from .live_v49f_images import LiveImages

HISTORY_SIZE = 4
IMAGE_STEP_SEC = .1
PREVIOUS_FRAME_SEC = 1/30
FORMULA_TOLERANCE = .1
POLICY = Path('models/v49f/policy.json')


def raw_side(side: Any) -> dict:
    result = fields(side, SIDE_FIELDS)
    event = side.chain_event
    result.chain_event = None if event is None else fields(event, (*CHAIN_FIELDS, 'before_board'))
    return encode(result)['namespace']


class CaptureV49F:
    def __init__(self) -> None:
        self.images = LiveImages()
        self.policy = json.loads(POLICY.read_text(encoding='utf-8'))
        self.game: int | None = None
        self.prefire = self.create_prefire()
        self.reset(None)

    def create_prefire(self) -> Any:
        """段3 (live_x_prefire / _shadow) のときだけ作る。既定OFFでは import もしない。"""
        from .live_v49f_prefire import flags
        if not flags()[0]:
            return None
        from .live_v49f_prefire import PrefireCapture
        return PrefireCapture(self.images)

    def reset(self, game: int | None) -> None:
        self.game = game
        self.ledger = ScorePrefix(self.policy['first_move_quiet_sec'])
        self.histories: list[list] = [[], []]
        self.colors: set[int] = set()
        self.active: list[dict | None] = [None, None]
        self.triggers: list[float | None] = [None, None]
        self.steps: list[list] = [[], []]
        self.next_id = 0

    def capture(self, result: Any, frame: Any, b3: dict, scores: Any) -> dict:
        from src.ledger_x_options import boundary_game, board_enabled
        game, stamp = boundary_game(b3), frame.media_sec
        if game != self.game:
            self.reset(game)
        self.images.append(stamp, frame.image)
        raw = (result.p1, result.p2)
        score_values = (scores.score_1p, scores.score_2p)
        for side, score in enumerate(score_values):
            self.ledger.observe(side, stamp, score)
        messages, reasons = [], []
        for side, current in enumerate(raw):
            self.remember(side, current, stamp, b3['sides'][side].get('formula_step'))
            try:
                messages.extend(self.side(side, current, stamp, score_values[side]))
            except ValueError as exc:
                reasons.append(str(exc))
        packet = dict(t=stamp, game=game, raw=[raw_side(s) for s in raw], origins=messages,
                      capture_reason=' / '.join(reasons) or None)
        if board_enabled():
            packet['adopted_game'] = b3.get('adopted_game')
            packet['firepower_ledgers'] = [self.ledger.snapshot(side, stamp) for side in range(len(raw))]
        if self.prefire is not None:
            ledgers = [self.ledger.snapshot(side, stamp) for side in range(len(raw))]
            packet['prefire'] = self.prefire.observe(game, stamp, raw, frame.image, ledgers)
        return packet

    def remember(self, side: int, raw: Any, stamp: float, step: dict | None) -> None:
        for pair in (raw.next_pair, raw.dnext_pair):
            self.colors.update(c for c in pair or () if 1 <= c <= 5)
        board = raw.confirmed_board
        if board is not None and raw.state.name == 'STABLE':
            values = board._grid.ravel().tolist()
            self.colors.update(c for c in values if 1 <= c <= 5)
            item = dict(t=stamp, board=values, next=list(raw.next_pair) if raw.next_pair else None)
            history = self.histories[side]
            if not history or any(item[k] != history[-1][k] for k in ('board', 'next')):
                history.append(item)
        if step is not None:
            self.steps[side].append(dict(step, side=side))
            self.steps[side] = self.steps[side][-HISTORY_SIZE:]

    def side(self, side: int, raw: Any, stamp: float, score: int | None) -> list[dict]:
        event = raw.chain_event
        if event is not None and event.trigger_sec != self.triggers[side]:
            self.active[side] = None
            if event.mechanism == 'formula_read' and event.chain_count == 1:
                initial = self.initial(side, raw, stamp)
                self.triggers[side] = event.trigger_sec
                return [initial]
            self.triggers[side] = event.trigger_sec
        current = self.active[side]
        if current is None:
            return []
        messages = []
        for step in self.steps[side]:
            if step['t_sec'] > current['last_formula']:
                messages.append(self.message(current, 'formula', step, step['t_sec']))
                current['last_formula'] = step['t_sec']
        messages.append(self.message(current, 'score', dict(t=stamp, score=score), stamp))
        messages.append(self.message(current, 'frame', self.frame(current, raw, stamp, score), stamp))
        return messages

    def initial(self, side: int, raw: Any, stamp: float) -> dict:
        from src.prefire_origin_stream import formula_step
        trigger = raw.chain_event.trigger_sec
        first = next((s for s in self.steps[side] if abs(s['t_sec']-trigger) <= FORMULA_TOLERANCE), None)
        ledger = self.ledger.snapshot(side, stamp)
        if first is None or ledger['carry'] is None or ledger['first_move'] is None:
            raise ValueError('初段式・試合開始得点待ち')
        history = [h for h in self.histories[side] if h['t'] <= trigger]
        inp = dict(video=0, game=self.game, event_id=self.next_id, input_sec=stamp,
            trigger_sec=trigger, side=side, sources=self.sources(raw, stamp, trigger, side),
            history=dict(history=deepcopy(history[-HISTORY_SIZE:]),
                history_latest=max((h['t'] for h in history), default=None),
                hidden_history=[dict(Counter(h['board'][c] for h in history)) for c in range(6)]),
            legacy=[], pair={}, colors=sorted(self.colors), first=formula_step(first, 1), ledger=ledger)
        current = dict(id=self.next_id, order=self.next_id, side=side,
                       last_image=float('-inf'), last_formula=first['t_sec'])
        self.active[side] = current
        self.next_id += 1
        return dict(self.message(current, 'initial', inp, stamp),
                    initial_next=list(raw.next_pair) if raw.next_pair else None)

    def frame(self, current: dict, raw: Any, stamp: float, score: int | None) -> dict:
        sources = []
        if score is not None and stamp-current['last_image'] >= IMAGE_STEP_SEC:
            sources = self.images.boards(stamp, stamp, current['side'], (-PREVIOUS_FRAME_SEC, 0.))
            for index, source in enumerate(sources):
                source['classifier'] = 'vote' if index % 2 else 'median'
            current['last_image'] = stamp
        event = raw.chain_event
        return dict(t=stamp, state=raw.state.name, score=raw.score,
            next=list(raw.next_pair) if raw.next_pair else None,
            board=raw.confirmed_board._grid.ravel().tolist() if raw.confirmed_board is not None else None,
            event={k: getattr(event, k) for k in CHAIN_FIELDS} if event else None,
            normal_score=score is not None, score_t=stamp, sources=sources)

    def sources(self, raw: Any, stamp: float, trigger: float, side: int) -> list[dict]:
        import numpy as np
        snapshot = getattr(raw, 'prefire_snapshot', None) or {}
        sources = []
        if snapshot.get('board') is not None:
            if snapshot['end_sec'] > stamp:
                raise ValueError('未到着の発火前盤')
            sources.append(dict(kind='cnn', t=snapshot['end_sec'],
                                board=np.asarray(snapshot['board']).ravel().tolist()))
        sources.extend(self.images.boards(trigger, stamp, side))
        return sources

    @staticmethod
    def message(current: dict, kind: str, data: dict, stamp: float) -> dict:
        return dict(id=current['id'], order=current['order'], t=stamp, kind=kind, data=data)
