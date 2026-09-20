"""固定原Recorder.complete_stepからの発行を直接捕捉し、直近両sideだけ保持する。"""
from __future__ import annotations

import hashlib
import json
import inspect
from pathlib import Path
import sys
from typing import Any
import journal_context as C

SOURCE = Path(__file__).resolve().parent.parent/'g2_atomic_journal_capture_2026-09-09_v1/observer.py'
SOURCE_SHA = 'b0d956a4fef51846a7baa2cee95a3679887bd2e3e9e620462e8375dc238d830c'


class Witness:
    def __init__(self, journal: Any) -> None:
        C.require(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA, 'J_witness_source')
        complete = type(journal).complete_step
        C.require(Path(complete.__code__.co_filename).resolve() == SOURCE, 'J_complete_source')
        C.require(inspect.ismethod(journal.emit) and journal.emit.__self__ is journal
            and journal.emit.__func__ is type(journal).emit, 'J_original_emit')
        self.journal, self.code, self.original = journal, complete.__code__, journal.emit
        self.rows: dict[str, str] = {}
        self.error: Any = None
        self.closed = False

    def emit(self, value: Any, caller: Any) -> Any:
        try:
            if self.closed or self.error is not None:
                return self.original(value)
            if value.get('kind') != 'step':
                return self.original(value)
            C.require(caller.f_code is self.code and caller.f_locals['self'] is self.journal, 'J_original_caller')
            item = caller.f_locals['item']
            C.require(value['token'] == item['token'] and value['side'] == item['scope']['side'], 'J_original_item')
            encoded = json.dumps(value, allow_nan=False, sort_keys=True)
            count = self.journal.count
            result = self.original(value)
            C.require(self.journal.count == count+1 and encoded == json.dumps(value, allow_nan=False, sort_keys=True), 'J_emit_changed')
            self.rows[value['side']] = encoded
            return result
        except BaseException as error:
            self.error = error
            raise

    def pair(self, frame: int) -> list[dict[str, Any]]:
        C.require(not self.closed and self.error is None and set(self.rows) == set(C.SIDES), 'J_witness_pair')
        rows = [json.loads(self.rows[side]) for side in C.SIDES]
        C.require(all(row['frame_idx'] == frame for row in rows), 'J_witness_frame')
        return rows


def install(stack: Any, journal: Any) -> Witness:
    value = Witness(journal)
    existed, previous = 'emit' in vars(journal), vars(journal).get('emit')
    def emit(row: Any) -> Any:
        return value.emit(row, sys._getframe(1))
    def close(kind: Any, body: Any, trace: Any) -> bool:
        try:
            C.require(vars(journal).get('emit') is emit, 'J_witness_foreign_patch')
            setattr(journal, 'emit', previous) if existed else delattr(journal, 'emit')
            C.require(journal.emit == value.original, 'J_witness_restore')
        except BaseException as error:
            value.error = error
            if body is None:
                raise
        finally:
            value.closed = True
        return False
    stack.push(close)
    journal.emit = emit
    return value
