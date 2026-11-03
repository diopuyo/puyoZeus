"""学習行と同じ状態・更新契機・特徴を使う、未来を持たないB3入力口。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from src import b3_rows
from src.b3_state import GameState, MATCH_TIME_EPS, SIDE_NAMES
from src.b3_timing import TimingTable
from src.b3_profile import profile, span

MS = 1000.0


@dataclass(frozen=True)
class QueueObservation:
    """収集器が採用した行だけを配送する。時刻はlean保存時と同じfloat32。"""
    index: int
    side: str
    frame: int
    t_sec: float
    queue: tuple[int, ...]
    game: int | None = None


class StreamingGameState(GameState):
    """NEXT行の全動画先読みを、当該フレームの入力と境界直前窓へ置換する。"""

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(rows_of=lambda side: ()))
        self.recent: deque[QueueObservation] = deque()
        self.pending: tuple[QueueObservation, ...] = ()
        self.new_game = False
        self.start_key: tuple[int, int] | None = None
        self.start_t = float('inf')
        self.started = False

    def accept(self, frame: dict, queues: tuple[QueueObservation, ...]) -> None:
        for q in queues:
            if q.frame > frame['frame'] or q.side not in SIDE_NAMES:
                raise ValueError('未来または不正なNEXT行')
        self.started = False
        self.new_game = frame['game_idx'] != self.game_idx
        if self.new_game:
            self.start_key = None
            self.recent.clear()
        for q in queues:
            key = (int(frame['game_idx']), q.game if q.game is not None else int(frame['game_idx']))
            if key != self.start_key:
                self.start_key = key
                self.start_t = q.t_sec
                self.recent.clear()
        self.pending = queues
        self.recent.extend(queues)
        cutoff = float(frame['t_sec']) - MATCH_TIME_EPS
        while self.recent and self.recent[0].t_sec < cutoff:
            self.recent.popleft()

    def _match_boundary(self, active: bool) -> tuple[bool, bool]:
        """採用行の試合開始通知と有効フレームが揃った時だけ初期化する。"""
        if self.start_key is None or self.t < self.start_t - MATCH_TIME_EPS:
            return False, False
        if self.start_key == self._match_key:
            return True, False
        if not active:
            return False, False
        self.reset_game(self.game_idx)
        self._match_key = self.start_key
        self.started = True
        self.new_game = True
        return True, True

    def apply_event(self, e: dict) -> int:
        """開始前に観測した遅延イベントも遮断する。"""
        if (not self._accept_match_observations or
                float(e['observed_at_sec']) < self.game_start_t - MATCH_TIME_EPS):
            return 0
        return super().apply_event(e)

    def _advance_lean(self) -> None:
        rows = self.recent if self.new_game else self.pending
        for side, st in zip(SIDE_NAMES, self.sides):
            for r in rows:
                if r.side != side or r.t_sec < self.game_start_t - MATCH_TIME_EPS:
                    continue
                q = tuple(int(v) if 1 <= int(v) <= 5 else 0 for v in r.queue)
                st.lean_row, st.queue, st.queue_t = r.index, q, min(r.t_sec, self.t)
                self.colors.add_queue(q)


class RealtimeRows:
    """探索・特徴の差替え口はrow_builder一つ。既定は学習生成器そのもの。"""

    def __init__(self, table: TimingTable, video_id: str = 'live',
                 row_builder: Callable | None = None, measure_timing: bool = True) -> None:
        self.state = StreamingGameState()
        self.table, self.video_id = table, video_id
        self.row_builder = row_builder or b3_rows._make_row
        self.boundary = False
        self.last_t = -float('inf')
        self.prev_icon: list = [None, None]
        self.prev_land = [0, 0]
        self.feature_ms = 0.0
        self.trigger = 0
        self.measure_timing = measure_timing
        self.timings: dict[str, float] = {}
        self.state_timings: dict[str, float] = {}

    def update(self, frame: dict, events: tuple[dict, ...] = (),
               queues: tuple[QueueObservation, ...] = (),
               emit: bool = True) -> dict | None:
        """全通知を処理する。emit=Falseは事後の試合選択のみで、状態は進める。"""
        started = perf_counter() if self.measure_timing else 0.0
        with profile(self.measure_timing) as self.timings:
            with span('state'):
                ready = self.advance(frame, events, queues, emit)
            self.state_timings = dict(self.timings)
            with span('feature_only'):
                row = self.build() if ready else None
        self.feature_ms = (perf_counter() - started) * MS if self.measure_timing else 0.0
        return row

    def advance(self, frame: dict, events: tuple[dict, ...] = (),
                queues: tuple[QueueObservation, ...] = (), emit: bool = True) -> bool:
        """計算の有無によらず、全観測と更新契機の時間軸を進める。"""
        gs = self.state
        if frame['frame'] <= gs.frame or frame['t_sec'] <= gs.t:
            raise ValueError('B3入力は厳密なフレーム・時刻順が必要')
        if frame['available_at_sec'] > frame['t_sec'] + b3_rows.AV_EPS:
            raise b3_rows.LeakError('未来の観測')
        if any(e['frame'] > frame['frame'] or e['observed_at_sec'] > frame['t_sec'] for e in events):
            raise b3_rows.LeakError('未来のイベント')
        gs.accept(frame, queues)
        placed = tuple(e['side'] for e in events if e['kind'] == 'placement_signal'
                       and e['frame'] == frame['frame'])
        self.boundary |= gs.update_frame(frame, placed)
        if gs.started:
            self.last_t = -float('inf')
            self.prev_icon, self.prev_land = [None, None], [0, 0]
        self.trigger = 0
        for event in events:
            self.trigger |= gs.apply_event(event)
        return self._ready(emit) if gs.in_match else False

    def _ready(self, emit: bool) -> bool:
        gs = self.state
        self.trigger |= b3_rows._frame_triggers(gs, self.boundary, self.last_t,
            b3_rows.HEARTBEAT_SEC, self.prev_icon, self.prev_land)
        self.boundary = False
        if not self.trigger or all(s.board is None for s in gs.sides):
            return False
        self.last_t = gs.t
        return emit

    def build(self) -> dict:
        """advance済みの状態から同期・非同期で同じ行を組み立てる。"""
        gs = self.state
        row = self.row_builder(gs, self.table, self.video_id, self.trigger, None)
        if row['av_max'] > gs.t + b3_rows.AV_EPS:
            raise b3_rows.LeakError('特徴に未来の情報が混入')
        return row

    def _row(self, emit: bool) -> dict | None:
        """従来の内部呼出し口も維持する。"""
        return self.build() if self._ready(emit) else None
