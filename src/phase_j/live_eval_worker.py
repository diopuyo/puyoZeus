"""イベント評価の子プロセス。認識・配信の所有権を持たない。"""
from __future__ import annotations

from dataclasses import asdict
import os
import pickle
import traceback
from types import SimpleNamespace
from typing import Any

from .live_evaluation import DeferredTracker, SplitExchangeOverlay


def boundary_tracker(models: Any) -> Any:
    class Tracker(DeferredTracker):
        def __init__(self, models: Any) -> None:
            super().__init__(models)
            self.sealed: list = []
            self.sealed_diagnostics: list = []

        def boundary(self, game_idx: int, t_sec: float) -> None:
            previous = self._game_idx
            super().boundary(game_idx, t_sec)
            if previous is not None and previous != game_idx:
                self.sealed.extend(asdict(r) for r in self.records)
                self.sealed_diagnostics.extend(self.diagnostics)
                self.records.clear()
                self._contexts.clear()
                self.diagnostics.clear()
                self._diagnostic_keys.clear()
    return Tracker(models)


def make_overlay(connection: Any, config: tuple) -> SplitExchangeOverlay:
    models, signals, m0, settled = config

    def build(*args: Any) -> Any:
        connection.send(dict(kind='static', args=args))
        reply = connection.recv()
        if reply['kind'] == 'error':
            raise RuntimeError(reply['message'])
        return reply['value']

    overlay = SplitExchangeOverlay(models, build, signals, m0, per_side_settled=settled)
    overlay.tracker = boundary_tracker(models)
    return overlay


def state(overlay: Any) -> dict:
    tracker = overlay.tracker
    result = dict(kind='ok', probability=tracker.probability, source=tracker.source,
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
    for blob in request['commands']:
        operation, args = pickle.loads(blob)
        if operation == 'update':
            overlay.failure_sec = args[3]
        overlay.failure_stage = operation
        getattr(overlay, operation)(*args)
    overlay.failure_stage = 'calculate'
    if request.get('fault'):
        raise RuntimeError('B16故障注入: '+request['fault'])
    if request['op'] == 'advance':
        return dict(kind='ok')  # 境界の退避記録は公開応答まで子が保持する。
    if request['op'] != 'updates':
        overlay.calculate()
    return state(overlay)


def worker(connection: Any, config: tuple, parent_pid: int) -> None:
    from .live_cpu import apply_runtime
    from .live_cache import bounded_chain_caches
    from .live_lifetime import protect_parent
    protect_parent(parent_pid)
    runtime = apply_runtime('event-evaluation')
    overlay = make_overlay(connection, config)
    connection.send(dict(kind='ready', pid=os.getpid(), runtime=runtime))
    with bounded_chain_caches():
        serve(connection, config, overlay)


def serve(connection: Any, config: tuple, overlay: Any) -> None:
    try:
        while True:
            request = connection.recv()
            if request['op'] == 'close':
                return
            try:
                if request['op'] == 'records':
                    tracker = overlay.tracker
                    reply = dict(kind='ok', records=tracker.sealed+[asdict(r) for r in tracker.records],
                                 diagnostics=tracker.sealed_diagnostics+tracker.diagnostics)
                else:
                    if request.get('reset'):
                        overlay = make_overlay(connection, config)
                    reply = execute(overlay, request)
            except Exception as error:
                reply = dict(kind='error', exception_type=type(error).__name__,
                    message=str(error), stack=traceback.format_exc(), pid=os.getpid(),
                    t_sec=getattr(overlay, 'failure_sec', None),
                    stage=getattr(overlay, 'failure_stage', None))
            connection.send(reply)
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()
