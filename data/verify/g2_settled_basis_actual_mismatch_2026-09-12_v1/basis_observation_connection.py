"""原Observer.captureの前に同callの診断票を保存し、元例外と所有解除を維持する。"""
from __future__ import annotations
import json
from pathlib import Path
from types import MethodType
from typing import Any
import basis_local_observation as LOCAL


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('basis_observation:' + reason)


def capture_row(observer: Any, item: dict, result: Any, error: Any) -> dict:
    recovery, frame = observer.recovery, item['frame']
    saved = recovery.waits.get(id(frame))
    require(saved is not None and saved['item'] is item and saved['caller'] is frame, 'saved_call_identity')
    require(frame.f_code in recovery.journal.codes and recovery.journal.active is None, 'original_J_complete')
    require(item['pipe'] is recovery.pipe and saved['frame'] == item['scope']['frame_idx'], 'pipe_and_clock')
    require(saved['scope'] == recovery.evidence.scope(recovery.factory, recovery.pipe), 'scope_changed')
    row = dict(frame=saved['frame'], call_token=item['token'], scope=saved['scope'],
               native_error=None if error is None else repr(error), quality_gate_clear=False)
    if error is not None:
        return row | dict(diagnostic_fields_missing=True)
    snapshot = recovery.journal.snapshot.__func__.__globals__['board']
    typed = recovery.state['postcommit_current_receiver'].rec.side_value(result)
    return row | LOCAL.capture(frame.f_locals, result, typed, snapshot)


class Connection:
    def __init__(self, observer: Any, output: Path) -> None:
        require('capture' not in vars(observer), 'foreign_instance_capture')
        self.observer, self.original = observer, observer.capture
        self.first, self.last = observer.gate.reset_frame, observer.gate.deadline
        require(type(self.first) is int and type(self.last) is int and self.first < self.last, 'bounds')
        self.max_rows = self.last - self.first + 1
        self.path = output / 'BASIS_LOCAL_OBSERVATIONS.jsonl'
        self.stream = self.path.open('x', encoding='utf-8')
        self.rows = 0
        self.error: str | None = None
        self.closed = False
        self.wrapper = MethodType(self.forward, observer)
        observer.capture = self.wrapper

    def forward(self, observer: Any, item: dict, result: Any, error: Any) -> Any:
        failure = None
        try:
            require(observer is self.observer and observer.capture is self.wrapper and not self.closed, 'capture_owner')
            # 原Observerの範囲外call判定をそのまま残す。
            if id(item['frame']) in observer.recovery.waits:
                saved = observer.recovery.waits[id(item['frame'])]
                require(self.first <= saved['frame'] <= self.last and self.rows < self.max_rows, 'observation_bounds')
                value = capture_row(observer, item, result, error)
                self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                self.stream.flush()
                self.rows += 1
        except BaseException as caught:
            self.error, failure = repr(caught), caught
        returned = self.original(item, result, error)  # 元例外が診断例外より優先。
        if failure is not None:
            raise failure
        return returned

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        failure = None
        owned = vars(self.observer).get('capture') is self.wrapper
        try:
            if owned:
                del self.observer.capture
            require(owned and self.observer.capture == self.original, 'capture_restore')
        except BaseException as caught:
            failure = caught
        finally:
            try:
                self.stream.close()
            except BaseException as caught:
                failure = failure or caught
            self.closed = True
        status = dict(rows=self.rows, bounds=[self.first, self.last], max_rows=self.max_rows,
                      restored=owned and self.observer.capture == self.original,
                      error=self.error, close_error=None if failure is None else repr(failure),
                      closed=self.stream.closed, quality_gate_clear=False)
        try:
            with self.path.with_name('BASIS_LOCAL_OBSERVATION_STATUS.json').open('x', encoding='utf-8') as stream:
                json.dump(status, stream, ensure_ascii=False, indent=2)
        except BaseException as caught:
            failure = failure or caught
        if body is None and failure is not None:
            raise failure
        return False


def install(stack: Any, observer: Any, output: Path) -> Connection:
    value = Connection(observer, output)
    stack.push(value.close)
    return value
