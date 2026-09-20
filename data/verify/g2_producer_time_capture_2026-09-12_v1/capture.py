"""原採録の完了時刻を別票へ捕捉する。試合開始の真値は認定しない。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any

FPS, STRIDE, SIDES = 60, 2, 2
LISTS = ('advance_times', 'visual_rise_times', 'rejected_rise_times', 'anomalies')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('producer_capture:' + reason)


class Capture:
    def __init__(self, loop: Any, tail: Any, identity: dict[str, str]) -> None:
        self.loop, self.tail = loop, tail
        self.identity = dict(identity)
        require(bool(identity) and all(type(v) is str and v for v in identity.values()), 'identity')
        self.recorder = loop.runtime_state['accounting_recorder']
        self.shared = loop.runtime_state['shared_game']
        require(self.recorder is not None and self.recorder._observed_frame_count == 0, 'late_recorder')
        require(tail.count == 0 and tail.stream.count == 0, 'late_metadata')
        require(loop.error is None and not hasattr(loop, '_producer_capture'), 'late_or_reentry')
        require(all(not getattr(self.shared, n) for n in LISTS), 'preexisting_boundary')
        self.original = loop.collect_lean
        self.first: int | None = None
        self.last: int | None = None
        self.count, self.busy, self.closed = 0, False, False
        self.error: BaseException | None = None
        self.history: dict[str, list[Any]] = {n: [] for n in LISTS}
        self.events: list[dict[str, Any]] = []
        self.hook = self.collect
        loop._producer_capture = self
        loop.collect_lean = self.hook

    def ready(self) -> None:
        if self.error is not None:
            raise self.error
        if self.loop.error is not None:
            raise self.loop.error
        require(not self.closed and self.loop._producer_capture is self, 'closed_or_replaced')

    def collect(self, cap: Any, pipeline: Any, start_frame: int, n_frames: int,
                effective_interval_frames: int, fps: float) -> Any:
        self.ready()
        try:
            require(not self.busy, 'recursive_call')
            require(type(start_frame) is int and start_frame >= 0, 'frame')
            require(type(n_frames) is int and n_frames == 1 and type(effective_interval_frames) is int
                    and effective_interval_frames == STRIDE and type(fps) in (int, float) and fps == FPS,
                    'clock_contract')
            require(self.last is None or start_frame == self.last + STRIDE, 'noncontiguous_frame')
            require(self.recorder._observed_frame_count == self.count and self.tail.count == self.count * SIDES,
                    'precall_count')
            self.busy = True
            result = self.original(cap, pipeline, start_frame, n_frames, effective_interval_frames, fps)
            self.completed(start_frame)
            return result
        except BaseException as exc:
            self.error = self.loop.error = exc
            raise
        finally:
            self.busy = False

    def completed(self, frame: int) -> None:
        state = self.loop.runtime_state
        require(state['accounting_recorder'] is self.recorder and state['shared_game'] is self.shared, 'producer_replaced')
        require(state['fi'] == frame and state['t_sec'] == frame / FPS, 'producer_clock')
        require(self.recorder._observed_frame_count == self.count + 1, 'unprocessed_or_count')
        self.tail.check(frame)
        require(self.tail.count == (self.count + 1) * SIDES, 'metadata_count')
        updated = {n: deepcopy(getattr(self.shared, n)) for n in LISTS}
        additions: list[dict[str, Any]] = []
        for name in LISTS:
            previous, current = self.history[name], updated[name]
            require(current[:len(previous)] == previous, 'boundary_prefix_mutated')
            for value in current[len(previous):]:
                additions.append({'list': name, 'raw_value': value,
                                  'available_frame': frame, 'available_sec': frame / FPS})
        self.history, self.events = updated, self.events + additions
        self.first = frame if self.first is None else self.first
        self.last, self.count = frame, self.count + 1

    def snapshot(self) -> dict[str, Any]:
        self.ready()
        try:
            require(not self.busy and self.last is not None, 'snapshot_not_ready')
            self.tail.check(self.last)
            require(self.recorder._observed_frame_count == self.count, 'snapshot_count')
            require(self.loop.runtime_state['accounting_recorder'] is self.recorder
                    and self.loop.runtime_state['shared_game'] is self.shared, 'snapshot_producer')
            require(all(getattr(self.shared, n) == self.history[n] for n in LISTS), 'snapshot_boundary')
            return deepcopy({'identity': self.identity, 'first_frame': self.first, 'last_frame': self.last,
                'observed_count': self.count, 'metadata': list(self.tail.rows), 'boundary_events': self.events,
                'accounting': self.recorder.sidecar_value(self.first, self.last + 1),
                'game_anchor_qualified': False, 'live_qualified': False, 'quality_gate_clear': False})
        except BaseException as exc:
            self.error = self.loop.error = exc
            raise

    def close(self) -> None:
        require(not self.closed and not self.busy, 'close_state')
        require(self.loop.collect_lean is self.hook and self.loop._producer_capture is self, 'foreign_hook')
        self.loop.collect_lean = self.original
        del self.loop._producer_capture
        self.closed = True


def install(stack: Any, loop: Any, tail: Any, identity: dict[str, str]) -> Capture:
    capture = Capture(loop, tail, identity)
    stack.callback(capture.close)
    return capture
