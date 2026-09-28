"""一回の評価失敗をHOLDへ隔離し、連続失敗時だけ子プロセスを交換する。"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time
import traceback
from typing import Any

from .live_eval_worker import worker
from .live_spool import DiskRows

FAILURE_LIMIT = 3
REPLY_TIMEOUT_SEC = 30.0
STARTUP_TIMEOUT_SEC = 120.0
JOIN_TIMEOUT_SEC = 2.0
NOTICE_PREFIX_SIZE = 4


class EvaluationError(RuntimeError):
    """親のフレーム境界で処理する、記録済みの子プロセス例外。"""


class TrackerView:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.probability: float | None = None
        self.source = 'waiting_confirmed'
        self._static_probability: float | None = None
        self.prediction_discard_reason: str | None = None
        self.records: list = []

    def save(self, path: Path) -> None:
        self.owner.save(path)


class SupervisedOverlay:
    def __init__(self, models: Any, build_static: Any, signals: Any, m0: Any = None,
                 per_side_settled: bool = False, *, directory: Path) -> None:
        self.config = (models, signals, m0, per_side_settled)
        self.build_static, self.directory = build_static, directory
        directory.mkdir(parents=True, exist_ok=True)
        self.tracker = TrackerView(self)
        self.archive = DiskRows(directory/'supervised-exchanges.pickle')
        self.diagnostics = DiskRows(directory/'supervised-diagnostics.pickle')
        self.journal = DiskRows(directory/'evaluation-journal.pickle')
        self.pending: list[bytes] = []
        self.game: int | None = None
        self.t_sec = 0.0
        self.failures, self.restarts, self.error_count = 0, 0, 0
        self.dirty, self.closed = False, False
        self.fault: str | None = None
        self.auto_acknowledge = True
        self.on_error: Any = None
        self.start()

    def start(self) -> None:
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker,
            args=(child, self.config, os.getpid()), name='event-evaluation-worker')
        self.process.start()
        child.close()
        try:
            self.runtime = self.receive(STARTUP_TIMEOUT_SEC)
        except Exception:
            self.stop_worker()
            raise

    def receive(self, timeout: float = REPLY_TIMEOUT_SEC) -> dict:
        deadline = time.monotonic()+timeout
        while True:
            if not self.connection.poll(max(0.0, deadline-time.monotonic())):
                raise TimeoutError('イベント評価応答なし')
            reply = self.connection.recv()
            if reply['kind'] != 'static':
                return reply
            try:
                value = self.build_static(*reply['args'])
                self.connection.send(dict(kind='ok', value=value))
            except Exception as error:
                self.connection.send(dict(kind='error', message=str(error)))
                self.static_error = dict(exception_type=type(error).__name__,
                                        message=str(error), stack=traceback.format_exc())

    def update(self, *args: Any) -> None:
        self.t_sec, game = args[3:5]
        if game != self.game:
            self.journal.close()
            self.journal = DiskRows(self.directory/'evaluation-journal.pickle')
            self.game = game
        blob = pickle.dumps(('update', args), protocol=pickle.HIGHEST_PROTOCOL)
        self.journal.append(blob)
        self.pending.append(blob)

    def calculate(self) -> None:
        commands = list(self.journal) if self.dirty else self.pending
        request = dict(op='batch', commands=commands, reset=self.dirty, fault=self.fault)
        self.fault, self.static_error = None, None
        try:
            self.connection.send(request)
            reply = self.receive()
        except (OSError, EOFError, TimeoutError) as error:
            reply = dict(kind='error', exception_type=type(error).__name__, message=str(error),
                         stack=traceback.format_exc(), pid=self.process.pid)
        if reply['kind'] == 'error':
            self.failed(dict(reply, **(self.static_error or {})))
            raise EvaluationError(reply['message'])
        self.pending.clear()
        self.journal.append(pickle.dumps(('calculate', ())))
        self.dirty = False
        if self.auto_acknowledge:
            self.succeeded()
        self.tracker.probability, self.tracker.source = reply['probability'], reply['source']
        self.tracker._static_probability = reply['static_probability']
        self.tracker.prediction_discard_reason = reply['prediction_discard_reason']
        self.archive.extend(reply['sealed'])
        self.diagnostics.extend(reply['diagnostics'])
        for key, value in reply['tracker_view'].items():
            setattr(self.tracker, key, value)
        self._landing_projection, self._history = reply['landing_view'], reply['history']

    def succeeded(self) -> None:
        """イベント計算だけでなく、公開候補の組立まで成功した時点で連続数を戻す。"""
        self.failures = 0

    def failed(self, error: dict) -> None:
        self.dirty = True
        self.pending.clear()  # 再構築用の全通知はディスクの試合journalに残っている。
        self.failures += 1
        self.error_count += 1
        log_event(self.directory, dict(error, kind='evaluation_error',
            t_sec=error.get('t_sec') if error.get('t_sec') is not None else self.t_sec,
            observed_sec=self.t_sec, game=self.game, consecutive=self.failures))
        if self.on_error is not None:
            self.on_error()  # 再起動のモデル読込を待たず、まずHOLDを公開する。
        if self.failures >= FAILURE_LIMIT or not self.process.is_alive():
            previous = self.process.pid
            self.stop_worker()
            self.start()
            self.restarts += 1
            self.failures = 0
            log_event(self.directory, dict(kind='evaluation_restart', t_sec=self.t_sec,
                game=self.game, previous_pid=previous, pid=self.process.pid,
                replay_from='match_boundary', notifications=len(self.journal)))

    def save(self, path: Path) -> None:
        if self.pending and not self.dirty:
            self.connection.send(dict(op='updates', commands=self.pending))
            flushed = self.receive()
            if flushed['kind'] == 'error':
                self.failed(flushed)
            else:
                self.archive.extend(flushed['sealed'])
                self.diagnostics.extend(flushed['diagnostics'])
                self.pending.clear()
        self.connection.send(dict(op='records'))
        current = self.receive()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream:
            for rows in (self.archive, current['records']):
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        events = list(self.diagnostics)+current['diagnostics']
        path.with_suffix('.diagnostics.json').write_text(json.dumps(
            dict(events=events, counts=dict(Counter(r['reason'] for r in events))),
            ensure_ascii=False), encoding='utf-8')

    def stop_worker(self) -> None:
        if self.process.is_alive():
            try:
                self.connection.send(dict(op='close'))
            except (OSError, EOFError):
                pass
            self.process.join(JOIN_TIMEOUT_SEC)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(JOIN_TIMEOUT_SEC)
        self.connection.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_worker()
        for rows in (self.journal, self.archive, self.diagnostics):
            rows.close()


def log_event(directory: Path, row: dict) -> None:
    row.update(utc=datetime.now(timezone.utc).isoformat(), monotonic=time.perf_counter())
    with (directory/'evaluation_errors.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')


def factory(bridge: Any, directory: Path) -> Any:
    def create(*args: Any, **kwargs: Any) -> SupervisedOverlay:
        overlay = SupervisedOverlay(*args, **kwargs, directory=directory)
        overlay.auto_acknowledge = False
        overlay.on_error = lambda: bridge.evaluation_hold(bridge.current_notice, overlay.game or 0)
        bridge.event_evaluator = overlay
        return overlay
    return create


def guard_loop(loop: Any) -> None:
    """認識通知の取得後に限って例外を隔離し、次の通知を継続する。"""
    import ast
    handler = ast.parse('try:\n    pass\nexcept Exception as error:\n'
        '    _evaluation_failure(_live_bridge, packet, locals(), error)\n    continue').body[0]
    handler.body = loop.body[NOTICE_PREFIX_SIZE:]
    loop.body = loop.body[:NOTICE_PREFIX_SIZE]+[handler]


def evaluation_failure(bridge: Any, notice: Any, local: dict, error: Exception) -> None:
    evaluator = getattr(bridge, 'event_evaluator', None)
    if not isinstance(error, EvaluationError):
        row = dict(exception_type=type(error).__name__, message=str(error),
                   stack=traceback.format_exc(), pid=os.getpid())
        if evaluator is not None:
            evaluator.failed(row)
        else:
            log_event(bridge.evaluation_directory, dict(row, kind='evaluation_error', t_sec=notice.t_sec))
    if evaluator is None or evaluator.on_error is None:
        bridge.evaluation_hold(notice, local.get('game_idx', 0))
