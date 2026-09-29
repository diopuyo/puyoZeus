"""認識済み盤面の有界窓を発火前原票へ接続する。追加CNN推論は行わない。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any
import numpy as np

from src.board import BOARD_ROWS, BOARD_COLS, HIDDEN_ROWS, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.exchange_event_terminal import ObservedDeathDetector
from src.prefire_snapshot_reader import WINDOW_SEC, vote_window
from .live_source import RECOGNITION_HZ
from .live_snapshot_quality import SnapshotQuality

WINDOW_FRAMES = round(WINDOW_SEC * RECOGNITION_HZ)
TRIGGER_DELAY_SEC = WINDOW_SEC
HISTORY_FRAMES = WINDOW_FRAMES + round(TRIGGER_DELAY_SEC * RECOGNITION_HZ)
TIME_TOLERANCE = 1e-6
SLOT_TOLERANCE_SEC = 0.5 / RECOGNITION_HZ


@dataclass(slots=True)
class SnapshotObservation:
    """画像読取時点で独立済みの小さい整数盤面を、複製・リスト化せず保持する。"""
    t_sec: float
    cnn: np.ndarray
    hsv: np.ndarray | None
    quality: bool
    glow: bool

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class LiveSnapshotInputs:
    """取得窓と通知遅れの余裕を保持し、再通知には確定済み原票を返す。"""

    def __init__(self, reader: Any) -> None:
        self.reader = reader
        self.buffers = (deque(maxlen=HISTORY_FRAMES), deque(maxlen=HISTORY_FRAMES))
        self.filters = (SnapshotQuality(), SnapshotQuality())
        self.latest: list[tuple | None] = [None, None]
        self.terminal = ObservedDeathDetector()
        self.previous_scores: tuple = (None, None)
        self.latched, self.boundary_sec = False, float('-inf')
        self.game, self.start = 0, None
        self.cost: dict[str, float] = {}

    def boundary(self, result: Any, stamp: float) -> None:
        from scripts.visualize_advantage_overlay import (
            _detect_score_reset, accept_formal_boundary, update_score_reset_latch)
        scores = (result.p1.score, result.p2.score)
        reset = _detect_score_reset(*scores, *self.previous_scores)
        accepted = accept_formal_boundary(reset_now=reset, latched=self.latched,
                                          t_sec=stamp, last_formal_t=self.boundary_sec)
        if self.start is None or accepted:
            self.start = stamp
            if accepted:
                self.game += 1
                self.boundary_sec = stamp
            for buffer, quality in zip(self.buffers, self.filters):
                buffer.clear()
                quality.reset()
            self.latest = [None, None]
        self.latched = update_score_reset_latch(self.latched, reset, *scores)
        self.previous_scores = scores

    def snapshot(self, idx: int, trigger: float) -> dict:
        cached = self.latest[idx]
        if cached is not None and cached[0] == trigger:
            return cached[1]
        frames = []
        for offset in range(WINDOW_FRAMES, 0, -1):
            stamp = trigger-offset/RECOGNITION_HZ
            if stamp < self.start:
                continue
            row = self.frame_at(idx, stamp, trigger)
            if row is None:
                # 欠落を飛ばすと「連続着地」の意味が変わるため区間を分断する。
                frames.append(dict(t_sec=stamp, cnn=np.zeros((BOARD_ROWS, BOARD_COLS)), quality='missing_frame'))
            else:
                quality = row['glow'] if not frames else row['quality']
                frames.append(dict(t_sec=stamp, cnn=row['cnn'], hsv=row['hsv'], quality=quality))
        value = vote_window(frames, trigger)
        self.latest[idx] = (trigger, value)
        return value

    def frame_at(self, idx: int, stamp: float, trigger: float) -> SnapshotObservation | dict | None:
        """動画は時刻完全一致、機器の取得揺れは同じ30Hz区画内だけで対応する。"""
        exact = next((r for r in self.buffers[idx]
                      if abs(r['t_sec']-stamp) < TIME_TOLERANCE), None)
        if exact is not None:
            return exact
        # 区画の半幅未満なので同じ観測を別区画へ複製しない。窓外と未来は除外する。
        return min((r for r in self.buffers[idx]
                    if trigger-WINDOW_SEC <= r['t_sec'] < trigger and
                    abs(r['t_sec']-stamp) < SLOT_TOLERANCE_SEC),
                   key=lambda r: abs(r['t_sec']-stamp), default=None)

    def retain(self, frame: np.ndarray, side: Any, idx: int, stamp: float) -> None:
        started = perf_counter()
        region = (self.reader._p1_region, self.reader._p2_region)[idx]
        pixels = getattr(self.reader, '_live_hsv_pixels', {}).get(idx)
        board = self.observed_board(side, idx)
        # 浮遊盤面はvote_windowが必ず除外するので、採否に使われないグロー集計を省く。
        landed = board is not None and not np.any((board._grid[:-1] != 0) & (board._grid[1:] == 0))
        quality, glow = self.filters[idx].observe(frame, region, pixels, check_glow=landed)
        observed_at = perf_counter()
        self.cost['quality_sec'] = self.cost.get('quality_sec', 0.)+observed_at-started
        if board is None:
            return
        # E31 vote_windowはHSV列を参照しない。既存観測だけ残し、未使用の追加読取はしない。
        hsv = getattr(self.reader, '_live_hsv_boards', {}).get(idx)
        self.buffers[idx].append(SnapshotObservation(stamp, board._grid,
            hsv._grid if hsv is not None else None, quality or glow, glow))
        self.cost['buffer_sec'] = self.cost.get('buffer_sec', 0.)+perf_counter()-observed_at

    def update(self, frame: np.ndarray, result: Any, stamp: float) -> Any:
        started = perf_counter()
        copied = getattr(self.reader, '_live_raw_board_sec', 0.)
        self.cost = dict(hsv_sec=0.)
        self.boundary(result, stamp)
        sides = []
        for idx, side in enumerate((result.p1, result.p2)):
            event = side.chain_event
            snapshot = self.snapshot(idx, event.trigger_sec) if event is not None else None
            board = self.observed_board(side, idx)
            midchain = board.copy() if board is not None and side.state == BoardState.GRAVITY_SETTLE else None
            if midchain is not None:
                midchain._grid[:HIDDEN_ROWS] = np.where(midchain._grid[HIDDEN_ROWS] != 0, COLOR_UNKNOWN, 0)
            sides.append(replace(side, prefire_snapshot=snapshot, midchain_board=midchain))
            self.retain(frame, side, idx, stamp)
        self.cost['snapshot_sec'] = perf_counter()-started+copied
        dead = self.terminal.update(frame)
        self.cost['total_sec'] = perf_counter()-started+copied
        return replace(result, p1=sides[0], p2=sides[1], confirmed_dead_sides=dead,
                       terminal_evidence_available=True)

    def observed_board(self, side: Any, idx: int) -> Any:
        """後段の浮遊除去・確定履歴補完を予測用の画像観測へ混入させない。"""
        boards = getattr(self.reader, '_live_raw_boards', {})
        if idx in boards:
            return boards[idx]
        board = getattr(side, 'raw_cnn_board', None)
        return board if board is not None else side.cnn_board


def prepare_recognition(pipe: Any) -> None:
    """毎通知の既存画像読取を捕捉する。前フレームの観測は必ず捨てる。"""
    owner = getattr(pipe, 'pipe', pipe)
    if hasattr(owner, '_reader'):
        owner._reader._live_raw_boards = {}
        owner._reader._live_hsv_boards = {}
        owner._reader._live_hsv_pixels = {}
        owner._reader._live_raw_board_sec = 0.


def enrich_recognition(pipe: Any, frame: np.ndarray, result: Any, stamp: float) -> Any:
    """入力切替による認識器交換後は、前入力の窓を引き継がない。"""
    owner = getattr(pipe, 'pipe', pipe)
    if not hasattr(owner, '_reader'):
        return result
    epoch = (getattr(pipe, 'epoch', None), getattr(pipe, 'publishing_ready', True))
    if not hasattr(owner, '_live_snapshot_inputs') or getattr(owner, '_live_snapshot_epoch', None) != epoch:
        owner._live_snapshot_inputs = LiveSnapshotInputs(owner._reader)
        owner._live_snapshot_epoch = epoch
    return owner._live_snapshot_inputs.update(frame, result, stamp)
