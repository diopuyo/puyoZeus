"""保存入力の時刻順再生と完了時計の結合 (dl1k 研究専用・既定 OFF)。

観測用 engine は毎コマ進め、要求点で「撃ち合い中か・鍵」を読む。計算用 engine は実行する要求の
入力位置まで遅れて進める (v4.15 OBS 再生と同じ)。要求の計算は、その要求の入力位置の状態だけを使う。
各要求点の行には、同じ撃ち合いで完了済み・500ms 以内の EX だけを結合する。自分の要求は自分の行に間に合わない。
要求の到着 (学習=入力時刻、OBS=保存受信時刻) が入力時刻より遅い場合は、到着時刻の順に時計へ渡す。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
from itertools import count
from time import perf_counter
from typing import Any, Callable

from src.phase_j._live_ex_frozen.src.dl1k_clock import REASON_OK, Request, WorkerClock
from src.phase_j._live_ex_frozen.src.dl1k_columns import AVAILABLE_COLUMNS, MASK_NOT_APPLICABLE, MASK_NOT_YET, blank
from src.phase_j._live_ex_frozen.src.dl1k_records import Frame
from src.phase_j._live_ex_frozen.src.dl1k_values import branch_values

MS = 1000.
STATUS_OUTSIDE = 'outside_exchange'


@dataclass
class Joined:
    """1 要求点の結合結果。extra は 70 列。"""
    extra: dict
    status: str
    fight: bool
    key: tuple
    request_index: int | None = None
    input_t: float | None = None
    available_at: float | None = None


def extra_from(result: dict) -> dict:
    """計算結果を 70 列へ。計算自体の失敗は不成立 (証明だけは計算済みなら残す)。
    retR を計算した結果 (PUYO_EX_RETR=1) なら retR の 8 列を足す (EX の探索の成否とは独立)。"""
    if result.get('failure'):
        extra = branch_values(None, result.get('proof'))
    else:
        extra = branch_values(result['output'], result['proof'], result['observed_scores'])
    if 'retr' in result:
        extra.update(result['retr'])
    return extra


@dataclass
class Replay:
    """observer は毎コマ、worker は実行時だけ進める。compute は engine -> 結果辞書。"""
    observer: Any
    worker: Any
    compute: Callable[[Any], dict]
    audit: Callable[[Request, float, dict, float], None] | None = None
    buffer: deque = field(default_factory=deque)
    events: list = field(default_factory=list)
    order: Any = field(default_factory=count)
    counts: dict = field(default_factory=lambda: dict(key_mismatch=0, catch_up_frames=0, points=0,
                                                       fight_points=0, joins=0))
    clock: WorkerClock | None = None
    recording: Any = None

    def __post_init__(self) -> None:
        self.clock = WorkerClock(self._run)

    def _catch_up(self, index: int) -> float:
        """worker を入力位置 index まで進め、その実時間 (ms) を返す。"""
        started = perf_counter()
        while self.buffer and self.buffer[0].index <= index:
            frame = self.buffer.popleft()
            self.worker.step(frame.t, frame.game, frame.raw)
            self.counts['catch_up_frames'] += 1
        return (perf_counter()-started)*MS

    def _run(self, request: Request, started_at: float) -> tuple[dict, float]:
        sync_ms = self._catch_up(request.index)
        probe = self.worker.probe()
        if probe.key != request.key or not probe.fight:
            self.counts['key_mismatch'] += 1
        if self.recording is not None:
            self.recording.before_compute(request, self.worker)
        result = self.compute(self.worker)
        result['sync_ms'] = sync_ms
        actual_ms = sync_ms+result['compute_ms']
        if self.recording is not None:
            self.recording.computed(request, started_at, result, actual_ms, self.worker)
        if self.audit is not None:
            self.audit(request, started_at, result, actual_ms)
        return extra_from(result), actual_ms

    def advance(self, t: float) -> None:
        """到着時刻が t 以下の要求・取消を到着順に時計へ渡す。"""
        while self.events and self.events[0][0] <= t:
            arrival, _, request = heapq.heappop(self.events)
            if request is None:
                self.clock.cancel_waiting(arrival)
            else:
                self.clock.submit(request)

    def join(self, t: float, fight: bool, key: tuple) -> Joined:
        """行の利用時刻 t に読める EX。撃ち合い外は NOT_APPLICABLE、未完了・鮮度超過は NOT_YET。"""
        self.counts['joins'] += 1
        if not fight:
            return Joined(blank(MASK_NOT_APPLICABLE), STATUS_OUTSIDE, False, key)
        pub, reason = self.clock.at(t, key)
        if reason != REASON_OK:
            return Joined(blank(MASK_NOT_YET), reason, True, key)
        extra = dict(pub.value)
        for column in AVAILABLE_COLUMNS:
            extra[column] = pub.available_at
        return Joined(extra, reason, True, key, pub.request.index, pub.request.input_t, pub.available_at)

    def _schedule(self, frame: Frame, fight: bool, key: tuple, arrival: float) -> None:
        if arrival < frame.t:
            raise ValueError('到着が入力より前')
        request = Request(arrival, key, frame.index, frame.t) if fight else None
        heapq.heappush(self.events, (arrival, next(self.order), request))
        self.counts['points'] += 1
        self.counts['fight_points'] += fight

    def _settle(self, frame: Frame) -> None:
        """待ちも未到着の要求も無ければ、worker を現在位置まで進める (費用に数えない)。"""
        waiting = [e[2].index for e in self.events if e[2] is not None]
        if self.clock.idle(frame.t) and not waiting:
            self._catch_up(frame.index)

    def step(self, frame: Frame, join: bool, request: bool, arrival: float | None = None) -> Joined | None:
        """1 入力を進める。join=True なら結合してから、request=True なら自分の要求を出す。"""
        self.advance(frame.t)
        self.buffer.append(frame)
        self.observer.step(frame.t, frame.game, frame.raw)
        probe = self.observer.probe() if (join or request) else None
        if self.recording is not None:
            self.recording.observed(frame, self.observer, request)
        joined = self.join(frame.t, probe.fight, probe.key) if join else None
        if request:
            self._schedule(frame, probe.fight, probe.key, frame.t if arrival is None else arrival)
            self.advance(frame.t)
        self._settle(frame)
        return joined

    def finish(self) -> dict:
        """入力の終わり。到着済みの要求だけ時計へ渡し、未実行の待ちは件数に残す。"""
        self.advance(float('inf'))
        return dict(self.counts, **self.clock.summary())
