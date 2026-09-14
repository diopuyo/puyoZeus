"""原enqueue保存成功後の2P複写。FIFO hook・消費・公開権限には触れない。"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

KEY = '_g2_second_enqueue_tap'
SIDE = '2P'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('second_enqueue_tap:' + reason)


class Tap:
    def __init__(self, journal: Any, path: Path, stack: Any) -> None:
        require(not hasattr(journal, KEY), 'duplicate_install')
        require('emit' not in vars(journal), 'foreign_emit')
        self.journal, self.original = journal, journal.emit
        self.status_path = path.with_name(path.stem + '_STATUS.json')
        require(not self.status_path.exists(), 'existing_status')
        self.closed, self.failure = False, None
        self.rows: list[dict] = []
        self.tokens: set[str] = set()
        self.stream = path.open('x', encoding='utf-8')
        self.hook = self.emit
        setattr(journal, KEY, self)
        journal.emit = self.hook
        stack.push(self.close)

    def emit(self, row: dict) -> None:
        try:
            self.original(row)  # 原保存が失敗したら成功票を作らない。
        except BaseException as error:
            self.failure = error
            raise
        try:
            require(not self.closed and self.journal.emit is self.hook, 'emit_owner')
            if self.failure is not None:
                return
            if row.get('kind') != 'enqueue' or row.get('side') != SIDE:
                return
            if row.get('status') != 'returned':
                return
            require(row['token'] not in self.tokens, 'duplicate_token')
            copied = deepcopy(row)
            packet = dict(source_fields=copied, native_step_verified=False,
                          physical_certified=False, quality_gate_clear=False)
            self.stream.write(json.dumps(packet, allow_nan=False) + '\n')
            self.stream.flush()
            self.tokens.add(row['token'])
            self.rows.append(copied)
        except BaseException as error:
            # 原enqueueのexceptへ戻さず、Native後の資格検査/終了で拒否する。
            if self.failure is None:
                self.failure = error

    def between(self, frame: int, cutoff: int, scope: tuple) -> tuple[dict, ...]:
        """検索のみ。Native同call確認・到来/ACK照合は呼び手が別に行う。"""
        require(not self.closed and self.failure is None, 'unusable')
        require(type(frame) is int and type(cutoff) is int and frame <= cutoff, 'clock')
        return tuple(deepcopy(row) for row in self.rows
            if row['source_id'] == scope[0] and row['run_id'] == scope[1]
            and row['software_reset'] == scope[2] and row['pipe_object_id'] == scope[3]
            and row['generation']['reset_epoch'] == scope[5] and row['side'] == scope[-1]
            and frame < row['frame_idx'] <= cutoff)

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        changed = self.journal.emit is not self.hook or getattr(self.journal, KEY, None) is not self
        failure = self.failure
        try:
            if self.journal.emit is self.hook:
                delattr(self.journal, 'emit')
            if getattr(self.journal, KEY, None) is self:
                delattr(self.journal, KEY)
        except BaseException as error:
            failure = failure or error
        try:
            self.stream.close()
        except BaseException as error:
            failure = failure or error
        self.closed = True
        if changed and failure is None:
            failure = ValueError('second_enqueue_tap:close_owner')
        try:
            with self.status_path.open('x', encoding='utf-8') as stream:
                json.dump(dict(rows=len(self.rows), closed=True,
                    stream_closed=getattr(self.stream, 'closed', None), restored=not changed,
                    error=None if failure is None else repr(failure),
                    physical_certified=False, quality_gate_clear=False), stream)
        except BaseException as error:
            failure = failure or error
            print('SECOND_ENQUEUE_STATUS_ERROR=' + repr(error), file=sys.stderr)
        if failure is not None and body is None:
            raise failure
        return False
