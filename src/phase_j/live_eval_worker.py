"""イベント評価の子プロセス。認識・配信の所有権を持たない。"""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from pathlib import Path
import os
import pickle
import traceback
from types import SimpleNamespace
from typing import Any

from src.exchange_event_tracker import ExchangeEventTracker
from .live_notification_eval import NotificationExchangeOverlay
from . import live_stage_timing as timing

AUDIT_ENGINES = ('_origin_guard', '_midchain', '_hidden_death', '_prefire', '_history_integrity')


def boundary_tracker(models: Any, original: Any = None) -> Any:
    class Tracker(ExchangeEventTracker):
        def __init__(self, models: Any) -> None:
            super().__init__(models)
            self.sealed: list = []
            self.sealed_diagnostics: list = []

        def boundary(self, game_idx: int, t_sec: float) -> None:
            previous = self._game_idx
            super().boundary(game_idx, t_sec)
            if previous is not None and previous != game_idx:
                self.sealed.extend(self.record_mapping(r) for r in self.records)
                self.sealed_diagnostics.extend(self.diagnostics)
                self.records.clear()
                self._contexts.clear()
                self.diagnostics.clear()
                self._diagnostic_keys.clear()
    tracker = Tracker(models)
    if original is not None:
        tracker.__dict__.update(original.__dict__)
    return tracker


def make_overlay(connection: Any, config: tuple, pending: deque | None = None) -> NotificationExchangeOverlay:
    models, signals, m0, settled = config[:4]
    options = config[4] if len(config) > 4 else {}
    pending = pending if pending is not None else deque()

    def build(*args: Any) -> Any:
        connection.send(dict(kind='static', args=args, request_id=overlay.request_id))
        reply = static_reply(connection, overlay.request_id, pending)
        if reply['kind'] == 'error':
            raise RuntimeError(reply['message'])
        return reply['value']

    overlay = NotificationExchangeOverlay(models, build, signals, m0, per_side_settled=settled, **options)
    overlay.tracker = boundary_tracker(models, overlay.tracker)
    return overlay


def static_reply(connection: Any, request_id: int, pending: deque) -> dict:
    """タイムアウト後の次要求を、static応答と取り違えず主ループへ戻す。"""
    while True:
        reply = connection.recv()
        if 'op' in reply:
            pending.append(reply)
        elif reply.get('request_id') == request_id:
            return reply


def state(overlay: Any) -> dict:
    tracker = overlay.tracker
    result = dict(kind='ok', probability=tracker.probability, source=tracker.source, display=overlay.display,
        smoothing=overlay.smoothing_state(),
        static_probability=tracker._static_probability,
        prediction_discard_reason=getattr(tracker, 'prediction_discard_reason', None),
        sealed=tracker.sealed, diagnostics=tracker.sealed_diagnostics,
        tracker_view={key: getattr(tracker, key, None) for key in
            ('current', 'firing', 'resolver', '_score_elapsed', 'count_sides')},
        landing_view=SimpleNamespace(**{key: getattr(overlay._landing_projection, key) for key in
            ('death', 'death_record', 'identity', 'last')}),
        history=[history[-1:] for history in overlay._history])
    tracker.sealed, tracker.sealed_diagnostics = [], []
    return result


def execute(overlay: Any, request: dict) -> dict:
    checkpoint = None
    for index, blob in enumerate(request['commands']):
        operation, args = timing.measure('event_decode', pickle.loads, blob)
        if operation == 'update':
            overlay.failure_sec = args[3]
            if overlay.tracker._game_idx != args[4]:
                checkpoint = dict(index=index, next_id=overlay.tracker._next_exchange_id,
                    smoothing=overlay.smoothing_state())
        overlay.failure_stage = operation
        timing.measure('event_'+operation, getattr(overlay, operation), *args)
    overlay.failure_stage = 'calculate'
    if request.get('fault'):
        raise RuntimeError('B16故障注入: '+request['fault'])
    if request['op'] == 'advance':
        reply = dict(kind='ok', smoothing=overlay.smoothing_state(), checkpoint=checkpoint,
            sealed=overlay.tracker.sealed, diagnostics=overlay.tracker.sealed_diagnostics)
        if os.environ.get('PUYO_B3_JUDGMENT') == '1':
            from .live_b3 import reference_state
            reply['b3_reference'] = timing.measure('event_reference', reference_state, overlay.tracker)
        overlay.tracker.sealed, overlay.tracker.sealed_diagnostics = [], []
        return reply
    if request['op'] != 'updates':
        overlay.calculate()
    return dict(state(overlay), checkpoint=checkpoint)


def worker(connection: Any, config: tuple, parent_pid: int, directory: Path | None = None) -> None:
    from .live_cpu import apply_runtime
    from .live_cache import bounded_chain_caches
    from .live_lifetime import protect_parent
    protect_parent(parent_pid)
    runtime = apply_runtime('event-evaluation')
    pending: deque = deque()
    overlay = make_overlay(connection, config, pending)
    if os.environ.get('PUYO_OBS_PREPARE_EXACT') == '1':
        from .live_obs_prepare import prepare_imports
        prepare_imports('event-evaluation')
    connection.send(dict(kind='ready', pid=os.getpid(), runtime=runtime))
    with bounded_chain_caches(future_search=True), ExitStack() as stack:
        timing.install(stack, directory, 'event_evaluation')
        serve(connection, config, overlay, pending)


def serve(connection: Any, config: tuple, overlay: Any, pending: deque | None = None) -> None:
    from .live_memory import MemoryProbe, retained_counts
    from src.exchange_integrity_memory import integrity_memory_roots
    memory = MemoryProbe('event-evaluation')
    pending = pending if pending is not None else deque()
    try:
        while True:
            request = pending.popleft() if pending else connection.recv()
            if request['op'] == 'close':
                return
            context = request.get('timing_context', {})
            timing.begin('event_evaluation', context.get('captured_at', 0.),
                         context.get('frame'), context.get('t_sec'))
            try:
                if request['op'] == 'records':
                    tracker = overlay.tracker
                    reply = dict(kind='ok', records=tracker.sealed+[tracker.record_mapping(r) for r in tracker.records],
                                 diagnostics=tracker.sealed_diagnostics+tracker.diagnostics,
                                 audits={name: engine.summary() if engine is not None else None
                                         for name in AUDIT_ENGINES
                                         for engine in (getattr(overlay, name),)})
                    reply['audit_retention'] = retained_counts(integrity_memory_roots(overlay))
                else:
                    if request.get('reset'):
                        overlay = make_overlay(connection, config, pending)
                        overlay.tracker._next_exchange_id = request.get('next_id', 1)
                        if request.get('smoothing') is not None:
                            overlay.restore_smoothing(request['smoothing'])
                    overlay.request_id = request.get('request_id')
                    reply = execute(overlay, request)
                if memory.path:
                    memory.sample(getattr(overlay, 'failure_sec', 0.),
                        dict(overlay=overlay, tracker=overlay.tracker, **integrity_memory_roots(overlay)))
            except Exception as error:
                reply = dict(kind='error', exception_type=type(error).__name__,
                    message=str(error), stack=traceback.format_exc(), pid=os.getpid(),
                    t_sec=getattr(overlay, 'failure_sec', None),
                    stage=getattr(overlay, 'failure_stage', None))
            timing.measure('event_ipc_reply', connection.send,
                           dict(reply, request_id=request.get('request_id'), op=request['op']))
            timing.finish(request_id=request.get('request_id'), operation=request['op'])
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()
