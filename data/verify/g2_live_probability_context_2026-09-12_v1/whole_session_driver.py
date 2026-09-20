"""会計完了後にのみ既存Sessionを駆動する。update差替えと開始資格の緩和はしない。"""
from __future__ import annotations
from contextlib import ExitStack
from typing import Any, Callable
from whole_collector_capture import forbidden_collect, require

KEY = 'whole_session_driver'
READY = ('probabilistic_basis_connection', 'probabilistic_tracking_mode')


class Driver:
    def __init__(self, state: dict, create: Callable[..., Any], frames: tuple[int, ...]) -> None:
        require(KEY not in state and 'belief_m1_session' not in state
                and 'belief_collector_boundary' not in state, 'duplicate_session_driver')
        require(bool(frames) and tuple(sorted(set(frames))) == frames, 'evaluation_frames')
        self.state, self.create, self.frames = state, create, frames
        self.scope = ExitStack()
        self.session: Any = None
        self.bridge: Any = None
        self.start_frame: int | None = None
        self.last: int | None = None
        self.closed, self.stopped = False, False
        self.error: BaseException | None = None
        state[KEY] = self

    def __call__(self, bridge: Any, frame: int) -> None:
        try:
            require(not self.closed and not self.stopped and self.error is None, 'session_driver_closed')
            require(bridge.state is self.state and self.state[KEY] is self, 'session_state_owner')
            require(self.bridge is None or self.bridge is bridge, 'session_bridge_replaced')
            require(bridge.capture.last == frame and bridge.completed_frames[-1] == frame, 'session_before_accounting')
            require(self.last is None or self.last < frame, 'session_duplicate_frame')
            self.last = frame
            if self.bridge is None:
                self.bridge = bridge
                self.entry = bridge.consumer
                bridge.lifetime.push(self.close)
            if self.session is None:
                if all(key in self.state for key in READY):
                    self.start(bridge, frame)
                else:
                    require(frame < self.frames[0], 'session_not_ready_at_selected_frame')
            else:
                self.session.completed(frame)
            if frame == bridge.frames[-1]:
                self.close(None, None, None)
        except BaseException as error:
            self.error = error
            if self.session is not None:
                self.session.error = error
            raise

    def start(self, bridge: Any, frame: int) -> None:
        # Jの証拠observerは生成後のemitだけを見る。生成frameを評価に流用しない。
        require(frame < self.frames[0], 'session_started_too_late')
        require('belief_m1_session' not in self.state, 'foreign_session')
        rec = self.state['provisional_context_observer']
        require(rec.active is None and not rec.errors and rec.rows
                and rec.rows[-1]['frame_idx'] == frame, 'session_after_observer')
        context = dict(state=self.state, pipe=bridge.initial['pipeline'],
                       factory=self.state['private_suffix_factory'], stack=self.scope)
        self.state['joint_capture_stack'] = self.scope
        self.session = self.create(self.scope, context)
        require(tuple(self.session.frames) == self.frames, 'session_frame_contract')
        self.state['belief_m1_session'] = self.session
        self.start_frame = frame
        self.scope.callback(self.stop)

    def stop(self) -> None:
        require(self.state[KEY] is self and self.state['belief_m1_session'] is self.session,
                'session_restore_owner')
        require(self.bridge.consumer is self.entry, 'session_consumer_replaced')
        self.bridge.consumer = forbidden_collect
        self.stopped = True
        # このdriverが唯一の駆動入口。入口停止後に原Sessionの解放検査へ渡す。
        self.session.restored = self.bridge.consumer is forbidden_collect

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        if self.closed:
            return False
        failure = body if body is not None else self.error
        try:
            if body is None and self.error is None:
                require(self.session is not None and self.last == self.bridge.frames[-1], 'session_incomplete')
        except BaseException as error:
            self.error = failure = error
        finally:
            try:
                self.scope.__exit__(type(failure) if failure is not None else None, failure, trace)
            except BaseException as error:
                if failure is None:
                    self.error = failure = error
            finally:
                self.closed = True
        if body is None and failure is not None:
            raise failure
        return False
