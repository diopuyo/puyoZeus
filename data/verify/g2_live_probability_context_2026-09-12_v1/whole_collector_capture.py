"""原whole collectの会計observe返却後だけ、既存producer観測コアを駆動する。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
from types import CodeType, SimpleNamespace as N
from typing import Any, Callable

SOURCE_SHA = '672963055a9fffa66531411be0a51742ee57c35de481652a77d02158ce24bee8'
FPS, STRIDE = 60, 2
KEY = 'whole_collector_capture'
LOCAL_KEYS = ('shared_game', 'pipeline', 'ojama_tracker')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('whole_collector_capture:' + reason)


def forbidden_collect(*args: Any, **kwargs: Any) -> Any:
    """transportは実collectorではない。旧one-update入口を呼ばせない。"""
    raise RuntimeError('whole_transport_is_not_a_collector')


def authentic_code(collector: Any) -> CodeType:
    source = Path(collector.__file__).resolve()
    require(source.parts[-3:] == ('event_first30_observed_context_v5_2026-08-30', 'scripts',
                                  'collect_boards_lean.py'), 'source_path')
    raw = source.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, 'source_sha')
    module = compile(raw, str(source), 'exec', dont_inherit=True)
    code = next(c for c in module.co_consts if isinstance(c, CodeType) and c.co_name == 'collect_lean')
    require(collector.collect_lean.__code__ == code, 'original_whole_code')
    require(collector.collect_lean.__globals__ is vars(collector), 'original_whole_globals')
    return collector.collect_lean.__code__


class Bridge:
    def __init__(self, collector: Any, state: dict, stack: Any, recorder_type: type,
                 capture_module: Any, frames: tuple[int, ...],
                 completed: Callable[[Any, int], None]) -> None:
        require(KEY not in state, 'duplicate_install')
        require(bool(frames) and tuple(range(frames[0], frames[-1] + STRIDE, STRIDE)) == frames, 'frames')
        self.code = authentic_code(collector)
        self.collector, self.state, self.stack = collector, state, stack
        self.recorder_type, self.capture_module = recorder_type, capture_module
        self.frames, self.consumer = frames, completed
        self.original_factory = collector.EventAccountingRecorder
        self.original_physical_factory = collector.EventPhysicalRecorder
        self.physical: Any = None
        self.pending_frame: int | None = None
        self.attempted_frame: int | None = None
        self.stage = 'constructor'
        self.failure_save_error: str | None = None
        self.capture: Any = None
        self.error: BaseException | None = None
        self.closed, self.completed_frames = False, []
        self.lifetime = ExitStack()
        self.hook = self.construct
        self.physical_hook = self.construct_physical
        self.restore_error: str | None = None
        state[KEY] = self
        collector.EventAccountingRecorder = self.hook
        collector.EventPhysicalRecorder = self.physical_hook
        stack.push(self.restore_factory)

    def caller(self, frame: Any) -> dict:
        require(frame is not None and frame.f_code is self.code, 'caller_code')
        require(frame.f_globals is vars(self.collector), 'caller_globals')
        return dict(frame.f_locals)

    def construct(self) -> Any:
        frame = inspect.currentframe()
        try:
            values = self.caller(frame.f_back)
        finally:
            del frame
        require(self.capture is None and not self.closed, 'duplicate_constructor')
        require(values['enable_event_accounting_sidecar'] is True, 'accounting_disabled')
        self.physical_enabled = values['enable_event_physical_sidecar']
        require(type(self.physical_enabled) is bool, 'physical_flag')
        require(values['fps'] == FPS and values['effective_interval_frames'] == STRIDE, 'clock')
        require(values['start_frame'] == self.frames[0], 'start_frame')
        tail = self.state['reset_metadata_tail']
        recorder = self.recorder_type()
        require(type(recorder) is self.recorder_type and 'observe' not in vars(recorder), 'recorder_factory_type')
        self.original_observe = recorder.observe
        self.transport = N(runtime_state=dict(values, accounting_recorder=recorder),
                           error=None, collect_lean=forbidden_collect)
        self.initial = {key: values[key] for key in LOCAL_KEYS}
        journal = self.state['private_suffix_factory'].provider.journal
        identity = dict(source_id=journal.source_id, run_id=journal.run_id)
        self.capture = self.capture_module.install(self.lifetime, self.transport, tail, identity)
        self.observe_hook = self.observe
        recorder.observe = self.observe_hook
        self.state['joint_producer_capture'] = self.capture
        self.stack.push(self.finish)
        return recorder

    def construct_physical(self) -> Any:
        frame = inspect.currentframe()
        try:
            self.caller(frame.f_back)
        finally:
            del frame
        require(self.capture is not None and self.physical_enabled and self.physical is None, 'physical_constructor')
        value = self.original_physical_factory()
        require(type(value) is self.original_physical_factory and value._observed_frame_count == 0
                and 'observe' not in vars(value), 'physical_initial_state')
        self.physical, self.original_physical_observe = value, value.observe
        self.physical_observe_hook = self.observe_physical
        value.observe = self.physical_observe_hook
        return value

    def observe(self, frame_idx: int, time_sec: float, tracker: Any, **kwargs: Any) -> Any:
        self.attempted_frame, self.stage = frame_idx, 'accounting'
        frame = inspect.currentframe()
        try:
            values = self.caller(frame.f_back)
        finally:
            del frame
        try:
            require(not self.closed and self.error is None, 'closed_or_failed')
            count = len(self.completed_frames)
            require(count < len(self.frames) and frame_idx == self.frames[count], 'frame_coverage')
            require(time_sec == frame_idx / FPS and values['fi'] == frame_idx, 'frame_clock')
            require(values['accounting_recorder'] is self.capture.recorder, 'recorder_replaced')
            require(all(values[key] is value for key, value in self.initial.items()), 'producer_replaced')
            require(tracker is self.initial['ojama_tracker'], 'tracker_replaced')
            require(self.capture.recorder._observed_frame_count == count, 'pre_observe_count')
            require(self.pending_frame is None, 'previous_physical_incomplete')
            result = self.original_observe(frame_idx, time_sec, tracker, **kwargs)
            self.transport.runtime_state = values
            self.capture.completed(frame_idx)
            self.pending_frame = frame_idx
            if not self.physical_enabled:
                self.dispatch(frame_idx)
            return result
        except BaseException as error:
            self.error = self.transport.error = error
            raise

    def observe_physical(self, frame_idx: int, time_sec: float, game_idx: int,
                         sides: Any, **kwargs: Any) -> Any:
        self.attempted_frame, self.stage = frame_idx, 'physical'
        frame = inspect.currentframe()
        try:
            values = self.caller(frame.f_back)
        finally:
            del frame
        try:
            require(self.pending_frame == frame_idx and values['fi'] == frame_idx
                    and time_sec == frame_idx / FPS, 'physical_clock')
            require(values['physical_recorder'] is self.physical, 'physical_replaced')
            require(self.physical._observed_frame_count == len(self.completed_frames), 'physical_pre_count')
            result = self.original_physical_observe(frame_idx, time_sec, game_idx, sides, **kwargs)
            require(self.physical._observed_frame_count == len(self.completed_frames) + 1, 'physical_post_count')
            self.dispatch(frame_idx)
            return result
        except BaseException as error:
            self.error = self.transport.error = error
            raise

    def dispatch(self, frame_idx: int) -> None:
        require(self.pending_frame == frame_idx, 'dispatch_pending')
        self.pending_frame = None
        self.completed_frames.append(frame_idx)
        if frame_idx == self.frames[-1]:
            self.save_terminal()
        self.stage = 'consumer'
        self.consumer(self, frame_idx)

    def save_terminal(self) -> None:
        """事後reconcileより前の原票。終了時に再採録して過去を書き換えない。"""
        value = self.capture.snapshot()
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
        path = self.state['output'] / 'JOINT_PRODUCER_CAPTURE.json'
        with path.open('xb') as stream:
            stream.write(raw)
        require(path.read_bytes() == raw, 'saved_bytes')
        self.state['joint_producer_capture_snapshot'] = json.loads(raw)

    def finish(self, kind: Any, body: Any, trace: Any) -> bool:
        if body is not None:
            self.save_failure(body)
        try:
            require(self.capture.recorder.observe is self.observe_hook, 'observe_foreign_hook')
            if self.physical is not None:
                require(self.physical.observe is self.physical_observe_hook, 'physical_foreign_hook')
            if body is None:
                require(self.error is None and tuple(self.completed_frames) == self.frames, 'incomplete')
                require('joint_producer_capture_snapshot' in self.state, 'missing_terminal_snapshot')
        except BaseException as error:
            self.restore_error = repr(error)
            if body is None:
                raise
        finally:
            if self.capture.recorder.observe is self.observe_hook:
                del self.capture.recorder.observe
            if self.physical is not None and self.physical.observe is self.physical_observe_hook:
                del self.physical.observe
            # 内側の失敗で外側の元例外を消さない。
            try:
                self.lifetime.__exit__(kind, body, trace)
            except BaseException as error:
                self.restore_error = repr(error)
                if body is None:
                    raise
            finally:
                self.closed = True
        return False

    def save_failure(self, error: BaseException) -> None:
        """最後の成功値と途中値を分けた失敗原票。正常snapshotとして消費させない。"""
        try:
            value = dict(error_type=type(error).__name__, error=repr(error), stage=self.stage,
                attempted_frame=self.attempted_frame, completed_frames=list(self.completed_frames),
                pending_frame=self.pending_frame, capture_last=self.capture.last,
                transport_frame=self.transport.runtime_state.get('fi'),
                metadata=list(self.capture.tail.rows), metadata_count=self.capture.tail.count,
                accounting_count=self.capture.recorder._observed_frame_count,
                accounting_pending=getattr(self.capture.recorder, '_previous_pending', None),
                physical_count=None if self.physical is None else self.physical._observed_frame_count,
                identity=dict(self.capture.identity), boundary_events=list(self.capture.events),
                start_counter_trace=list(getattr(self.capture, 'start_counter_trace', [])),
                full_snapshot=False, quality_gate_clear=False)
            path = self.state['output'] / 'JOINT_PRODUCER_CAPTURE.failure.json'
            with path.open('x', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        except BaseException as save_error:
            self.failure_save_error = repr(save_error)

    def restore_factory(self, kind: Any, body: Any, trace: Any) -> bool:
        try:
            require(self.collector.EventAccountingRecorder is self.hook, 'factory_foreign_hook')
            self.collector.EventAccountingRecorder = self.original_factory
            require(self.collector.EventPhysicalRecorder is self.physical_hook, 'physical_factory_foreign_hook')
            self.collector.EventPhysicalRecorder = self.original_physical_factory
            if body is None:
                require(self.capture is not None and self.closed, 'never_constructed_or_unclosed')
        except BaseException as error:
            self.restore_error = repr(error)
            if body is None:
                raise
        return False
