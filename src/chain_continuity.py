"""試作の連鎖継続: 同じ側の観測だけで境界と発火盤の裏付けを作る。"""
from __future__ import annotations

from typing import Any
import os

import numpy as np

from src.production_config import (LIVE_CHAIN_END_FORMULA_DEFAULT,
                                   LIVE_CHAIN_FORECAST_RECHECK_DEFAULT)

CONFIRM_FRAMES = 2
BOUNDARY_GAP_SEC = 0.5
FIRE_WINDOW_SEC = 0.35
MIN_CHAIN_SCORE = 40
MAX_READ_DIFFERENCES = 3
OJAMA = 9
PLACEMENT_CELLS = 2
PLACEMENT_SILENCE_SEC = 2.0


def end_enabled() -> bool:
    return os.environ.get('PUYO_LIVE_CHAIN_END_FORMULA', str(int(LIVE_CHAIN_END_FORMULA_DEFAULT))) == '1'


def recheck_enabled() -> bool:
    return os.environ.get('PUYO_LIVE_CHAIN_FORECAST_RECHECK', str(int(LIVE_CHAIN_FORECAST_RECHECK_DEFAULT))) == '1'


def pair(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict):
        value = value.get('tuple')
    return tuple(value) if value is not None and len(value) == 2 and all(1 <= c <= 5 for c in value) else ()


class ChainBoundary:
    """NEXT の繰上がりを連続観測で確認。状態名・スライド単独では終了しない。"""

    def __init__(self) -> None:
        self.queue: tuple[int, ...] = ()
        self.pending: tuple[int, ...] = ()
        self.run = 0
        self.slide_t = float('-inf')
        self.next_evidence_t = float('-inf')
        self.last_formula = float('-inf')
        self.last_score_change = float('-inf')
        self.score: float | None = None
        self.placement_t = float('-inf')
        self.placement_board: np.ndarray | None = None
        self.placement_pending: np.ndarray | None = None
        self.placement_run = 0
        self.count: int | None = None
        self.ojama_count: int | None = None
        self.ojama_pending: int | None = None
        self.ojama_run = 0
        self.t = float('-inf')
        self.kind = ''

    def observe(self, t: float, side: dict) -> None:
        from src.fire_board import clean
        formula = side.get('formula') or {}
        valid = bool(formula.get('valid')) and formula.get('product') is not None
        if valid:
            self.last_formula = t
        self._score(t, side.get('score'))
        if side.get('next_slide_motion'):
            self.slide_t = t
        self._placement(t, side)
        queue = pair(side.get('continuity_next', side.get('next_pair'))) + pair(side.get('continuity_dnext', side.get('dnext_pair')))
        self._next(t, queue, side, valid)
        board = clean(side.get('confirmed_board'))
        raw = clean(side.get('raw_cnn_board'))
        if board is not None and raw is not None:
            count = int(np.count_nonzero(board == OJAMA))
            supported = int(np.count_nonzero(raw == OJAMA)) >= count
            self.ojama_run = self.ojama_run + 1 if count == self.ojama_pending and supported else int(supported)
            self.ojama_pending = count
            if self.ojama_run >= CONFIRM_FRAMES:
                if self.ojama_count is not None and count > self.ojama_count and self._silent(t):
                    self.t, self.kind = t, 'ojama'
                self.ojama_count = count

    def _score(self, t: float, score: float | None) -> None:
        if score is None:
            return
        if self.score is not None and score - self.score >= MIN_CHAIN_SCORE:
            self.last_score_change = t
        self.score = score if self.score is None else max(score, self.score)

    def _silent(self, t: float) -> bool:
        return t - max(self.last_formula, self.last_score_change) >= BOUNDARY_GAP_SEC

    def _placement(self, t: float, side: dict) -> None:
        from src.fire_board import clean
        count = side.get('tsumo_count')
        if count is not None and self.count is not None and count > self.count and side.get('landing_diag'):
            self.placement_t = t
        board, raw = clean(side.get('confirmed_board')), clean(side.get('raw_cnn_board'))
        if board is None or raw is None:
            return
        self.placement_run = self.placement_run + 1 if np.array_equal(board, self.placement_pending) else 1
        self.placement_pending = board
        if self.placement_run < CONFIRM_FRAMES:
            return
        old = self.placement_board
        if old is not None:
            added = (old == 0) & (board >= 1) & (board <= 5)
            two_added = np.count_nonzero(added) == PLACEMENT_CELLS
            preserved = np.array_equal(board[~added], old[~added])
            supported = np.array_equal(raw[added], board[added])
            quiet = t - max(self.last_formula, self.last_score_change) >= PLACEMENT_SILENCE_SEC
            if two_added and preserved and supported and quiet and t - self.placement_t <= BOUNDARY_GAP_SEC:
                self.t, self.kind = t, 'placement'
        self.placement_board = board

    def _next(self, t: float, queue: tuple[int, ...], side: dict, valid: bool) -> None:
        count = side.get('tsumo_count')
        placed = count is not None and self.count is not None and int(count) > self.count
        if count is not None:
            self.count = int(count)
        if len(queue) != 4:
            return
        self.run = self.run + 1 if queue == self.pending else 1
        self.pending = queue
        if self.run < CONFIRM_FRAMES:
            return
        shifted = bool(self.queue and queue != self.queue and queue[:2] == self.queue[2:])
        same_placed = placed and queue == self.queue and bool(side.get('landing_diag'))
        if (shifted or same_placed) and t - self.slide_t <= BOUNDARY_GAP_SEC:
            self.next_evidence_t = t
        if not valid and self._silent(t):
            if self.t < self.next_evidence_t and t - self.next_evidence_t <= 2 * BOUNDARY_GAP_SEC:
                self.t, self.kind = t, 'next'
        self.queue = queue


def confirmed_fire(side: dict | None, chain_id: float, now: float) -> np.ndarray | None:
    from src.fire_board import clean
    event = (side or {}).get('chain') or (side or {}).get('chain_event') or {}
    event = event.get('namespace', event)
    trigger = event.get('trigger_sec')
    if trigger is None or float(trigger) > now or abs(float(trigger) - chain_id) > FIRE_WINDOW_SEC:
        return None
    return clean(event.get('before_board'))


def recheck_board(forecast: Any, t: float, raw: Any, side: dict | None) -> np.ndarray | None:
    """短い同一設置の履歴と確定発火盤が裏付ける盤だけ。長さは選択基準にしない。"""
    from src.fire_board import BACK, clean, step_scores
    fc = forecast.fc
    if fc is None or len(fc.observed) != 1 or t - fc.chain_id > FIRE_WINDOW_SEC:
        return None
    current = clean(raw)
    frames = [(tt, b) for tt, b in forecast.hist.frames if fc.chain_id - FIRE_WINDOW_SEC <= tt <= t][-BACK:]
    frames.append((t, current))
    confirmed = confirmed_fire(side, fc.chain_id, t)
    for _, board in reversed(frames):
        if board is None or np.any(board[0]):
            continue  # 隠し段の追加から火力を作らない。
        groups, _ = forecast.trace(board)
        scores = step_scores(groups)
        if not scores or scores[0] != fc.observed[0]:
            continue
        repeats = sum(np.array_equal(board, other) for _, other in frames if other is not None)
        if confirmed is None:
            continue
        differences = np.count_nonzero(board[1:] != confirmed[1:])
        confirmed_scores = step_scores(forecast.trace(confirmed)[0])
        if repeats >= CONFIRM_FRAMES and differences <= MAX_READ_DIFFERENCES and scores == confirmed_scores:
            return board
    return None
