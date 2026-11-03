"""v4.6実験専用。発火側の終了盤を保持し、受け側の確定盤で再計算する。"""
from __future__ import annotations
from src import prefire_work_budget as work_budget
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter
from typing import Any
from scipy.special import expit
from src.prefire_choice_model import predict_pair
from src.prefire_post_b3 import PostB3, MS, LIMIT_MS, validate_publication
from src.prefire_confidence import candidate_summary
from src.prefire_branches import calculate
from src.prefire_growth import SideInput, ROLLOUT_COUNT
from src.prefire_worker import LOW_SPEC_RATIO
from src.prefire_trial_defaults import PREFIRE_V4_POST_ENABLED_DEFAULT

SEED = 20261015


def projected_advance(records: list[dict] | None, model: dict, stamp: float) -> float:
    if not records or any(not r['scores'] for r in records):
        return 0.
    value = predict_pair(records, model, stamp)
    display = model['display']
    return float(display['c']*(expit(display['attack_beta']*value['a']-display['return_beta']*value['b'])-.5))


class PostV46(PostB3):
    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 enabled: bool = PREFIRE_V4_POST_ENABLED_DEFAULT, pre_model: dict | None = None) -> None:
        super().__init__(predictor, table, policy, native, enabled)
        self.pool = ThreadPoolExecutor(ROLLOUT_COUNT)
        self.curve_cache: list[tuple | None] = [None, None]
        self.last_confidence: list[dict] = []
        self.receiver_changes: list[dict] = []
        self.stable_queue: list[tuple] = [(), ()]
        self.pre_model = pre_model

    def _boundary(self, stamp: float, game: int) -> None:
        if self.state.game_idx != game:
            self.curve_cache = [None, None]
            self.last_confidence, self.receiver_changes = [], []
            self.stable_queue = [(), ()]
        super()._boundary(stamp, game)

    def _current(self, stamp: float, game: int, raw: list[dict]) -> None:
        super()._current(stamp, game, raw)
        for i, side in enumerate(self.state.sides):
            if side.state == 'STABLE' and not self._resolving(i):
                self.stable_queue[i] = side.queue
                if side.board_changed and self._resolving(1-i):
                    self.receiver_changes.append(dict(t=stamp, side=i, board=side.board.ravel().tolist()))

    def _accept_forecasts(self, forecasts: list[dict], stamp: float, game: int) -> None:
        previous = {(i,c['signature'][0]):c.get('confidence') for i,c in enumerate(self.chains) if c}
        super()._accept_forecasts(forecasts, stamp, game)
        for i,chain in enumerate(self.chains):
            if chain is None:
                continue
            values = [f['confidence'] for f in forecasts if f['side']==i
                and f['trigger']==chain['signature'][0] and 'confidence' in f]
            chain['confidence'] = values[-1] if values else previous.get((i,chain['signature'][0]))

    def summaries(self) -> list[dict]:
        result = []
        for i, chain in enumerate(self.chains):
            if chain is None or not self._resolving(i):
                continue
            if chain.get('external'):
                value = chain.get('confidence') or candidate_summary([], 0)
            else:
                count = max((o['count'] for o in chain['observed']), default=0)
                value = candidate_summary(chain['pool'], count, chain['result']['conflict'])
            result.append(dict(value, side=i, trigger=chain['signature'][0]))
        return result

    def projection(self) -> dict:
        value = super().projection()
        value['confidence'] = self.summaries()
        value['board_times'] = [s.board_t for s in self.state.sides]
        return value

    def _curve(self, side: int, value: dict) -> dict:
        board = tuple(value['boards'][side])
        queue = self.stable_queue[side] or self.state.sides[side].queue
        known = tuple(tuple(queue[k:k+2]) for k in range(0, len(queue), 2))
        colors = tuple(self.state.colors.known()[1])
        key = (board, known, colors)
        request = SideInput(self.state.game_idx, side, self.state.t, board, known, colors,
            int(value['pending'][side]), self.state.t-self.state.game_start_t,
            tuple(value['evidence_times']), 0, SEED+self.state.game_idx)
        cached = self.curve_cache[side]
        if cached is None or cached[0] != key:
            record = calculate(request, self.native, self.pool, True)
            self.curve_cache[side] = (key, record)
        record = deepcopy(self.curve_cache[side][1])
        record['request'].update(pending=request.pending, elapsed_sec=request.elapsed_sec,
            input_sec=request.input_sec, evidence_times=list(request.evidence_times))
        record.update(input_sec=self.state.t, ready_sec=self.state.t, busy_until=self.state.t)
        return record

    def _evaluate(self, stamp: float, began: float, extra_ms: float = 0.) -> dict | None:
        summaries = self.summaries()
        if summaries != self.last_confidence:
            self.last_key = None
        value = super()._evaluate(stamp, began, extra_ms)
        if value is None or 'v_post' not in value:
            return value
        self.last_confidence = deepcopy(summaries)
        started = perf_counter()
        try:
            value['pre_records'] = [self._curve(i, value) for i in (0, 1)]
        except ValueError as error:
            value.update(pre_records=None, pre_missing=str(error))
        if self.pre_model is not None:
            value['pre_delta_full'] = projected_advance(value['pre_records'],self.pre_model,stamp)
        elapsed = (perf_counter()-started)*MS*LOW_SPEC_RATIO
        value['pre_low_spec_ms'] = elapsed
        value['low_spec_ms'] += elapsed
        value['ready_sec'] += elapsed/MS
        value['request_latency_ms'] += elapsed
        value['over_budget'] = value['request_latency_ms'] > LIMIT_MS
        self.busy = value['ready_sec']
        value['receiver_changes'] = self.receiver_changes
        self.receiver_changes = []
        value['raw_states'] = [s.state for s in self.state.sides]
        work_budget.publication(value)
        validate_publication(value)
        return value
