"""採点専用。未来の実終了盤を予測器へ戻す経路を持たない。"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from scripts.exchange_review_io import save
from src.prefire_post_b3 import forecast_features


def finite_tree(value: object) -> object:
    if isinstance(value,float) and not np.isfinite(value):
        return None
    if isinstance(value,dict):
        return {k:finite_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_tree(v) for v in value]
    return value


class AuditOnly:
    def __init__(self, video: int) -> None:
        self.video = video
        self.events: dict[tuple,dict] = {}
        self.snapshots: dict[tuple,object] = {}
        self.raw_fires: set[tuple] = set()
        self.closures: list[dict] = []

    def observe(self, engine: object, value: dict | None, stamp: float, game: int, raw: list[dict]) -> None:
        for i, side in enumerate(raw):
            if side.get('chain_event') and side['state']['state'] == 'CHAIN':
                self.raw_fires.add((game,i,side['chain_event']['namespace']['trigger_sec']))
        if value is None:
            return
        if 'closed' in value:
            self.close(engine,value)
        if 'v_post' not in value:
            return
        for i, trigger in enumerate(value['chain_ids']):
            if trigger is None:
                continue
            key = (game,i,trigger)
            if key not in self.events:
                self.events[key] = dict(video=self.video,game=game,side=i,trigger=trigger,
                    fight=value['fight'],first_input=stamp,first_ready=value['ready_sec'],
                    a=value['v_post'],a_boards=deepcopy(value['boards']),a_pending=value['pending'],
                    confidence=deepcopy(value['confidence']),status='終了未観測')
                self.snapshots[key] = deepcopy(engine.state)
            self.events[key]['last_a'] = value['v_post']

    def infer(self, engine: object, state: object, boards: list, pending: list) -> float | None:
        if any(c not in (0,1,2,3,4,5,9) for b in boards for c in b) or not np.isfinite(pending).all():
            return None
        features = forecast_features(state,boards,pending,engine.table,engine.native)
        return engine.predictor.predict(features)

    def close(self, engine: object, value: dict) -> None:
        state = engine.state
        boards = [s.board.ravel().tolist() for s in state.sides]
        pending = [state.sl_snap[f'pending_p{i+1}'] for i in (0,1)]
        end_b = self.infer(engine,state,boards,pending)
        self.closures.append(dict(video=self.video,game=value['game'],fight=value['closed'],
            end=value['input_sec'],b_end=end_b,boards=boards,pending=pending,fired=value['fired']))
        for key,event in self.events.items():
            if event['game'] != value['game'] or event['fight'] != value['closed'] or event['status'] != '終了未観測':
                continue
            frozen = self.snapshots.pop(key)
            sender_only = deepcopy(event['a_boards'])
            sender_only[event['side']] = boards[event['side']]
            b = self.infer(engine,frozen,boards,pending)
            sender_b = self.infer(engine,frozen,sender_only,event['a_pending'])
            event.update(end=value['input_sec'],b=b,b_end=end_b,b_sender_only=sender_b,
                status='採点可能' if b is not None else '実終了盤欠測',
                sender_cell_errors=sum(x!=y for x,y in zip(event['a_boards'][event['side']],boards[event['side']])),
                actual_boards=boards,actual_pending=pending)

    def finish(self, engine: object) -> None:
        from scripts.prefire_v46_data import OUT
        save(OUT/f'oracle_{self.video}.json',finite_tree(dict(events=list(self.events.values()),
            closures=self.closures,raw_fires=sorted(self.raw_fires),scoring_only=True)))
