"""既存writer Witnessの成功通知から2P到来を複写。元emit/stream/FIFOは変更しない。"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from types import CodeType
from typing import Any
import second_enqueue_tap as BASE

KEY = '_g2_second_witness_tap'


class Tap(BASE.Tap):
    def __init__(self, witness: Any, path: Path, stack: Any) -> None:
        BASE.require(not hasattr(witness, KEY) and 'written' not in vars(witness), 'witness_owner')
        BASE.require(not witness.closed and witness.error is None, 'witness_unusable')
        self.witness, self.journal, self.original = witness, witness.journal, witness.written
        self.status_path = path.with_name(path.stem + '_STATUS.json')
        BASE.require(not self.status_path.exists(), 'existing_status')
        self.closed, self.failure, self.rows, self.tokens = False, None, [], set()
        codes = [value for value in type(self.journal).wrap_enqueue.__code__.co_consts
                 if isinstance(value, CodeType) and value.co_name == 'enqueue']
        BASE.require(len(codes) == 1, 'enqueue_original_code')
        self.enqueue_code = codes[0]
        self.stream = path.open('x', encoding='utf-8')
        self.hook = self.written
        setattr(witness, KEY, self)
        witness.written = self.hook
        stack.push(self.close)

    def source(self, row: dict, caller: Any) -> None:
        parent = caller.f_back
        for wrapper, inner in reversed(self.witness.contract.wrappers):
            BASE.require(parent.f_code is wrapper.__code__
                         and parent.f_locals.get('emit') is inner, 'enqueue_wrapper_origin')
            parent = parent.f_back
        BASE.require(parent.f_code is self.enqueue_code
            and parent.f_locals.get('self') is self.journal
            and parent.f_locals.get('token') == row['token'], 'enqueue_original_origin')
        BASE.require(row['pipe_object_id'] == id(parent.f_locals['pipe'])
            and row['added_occurrence_tokens'] == parent.f_locals['added'], 'enqueue_original_payload')

    def written(self, text: str, caller: Any) -> None:
        self.original(text, caller)
        try:
            BASE.require(not self.closed and self.witness.written is self.hook, 'witness_hook_owner')
            BASE.require(self.witness.error is None, 'original_witness_failed')
            if self.failure is not None:
                return
            row = json.loads(text)
            if row.get('kind') != 'enqueue' or row.get('side') != '2P' or row.get('status') != 'returned':
                return
            BASE.require(row['row_index'] == self.witness.last_index, 'witness_row_unverified')
            self.source(row, caller)
            BASE.require(row['token'] not in self.tokens, 'duplicate_token')
            self.stream.write(json.dumps(dict(source_fields=row, native_step_verified=False,
                physical_certified=False, quality_gate_clear=False), allow_nan=False) + '\n')
            self.stream.flush()
            self.rows.append(deepcopy(row))
            self.tokens.add(row['token'])
        except BaseException as error:
            self.failure = self.failure or error

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        changed = self.witness.written is not self.hook or getattr(self.witness, KEY, None) is not self
        if self.witness.written is self.hook:
            delattr(self.witness, 'written')
        if getattr(self.witness, KEY, None) is self:
            delattr(self.witness, KEY)
        failure = self.failure or self.witness.error
        try:
            self.stream.close()
        except BaseException as error:
            failure = failure or error
        self.closed = True
        if changed:
            failure = failure or ValueError('second_witness_tap:close_owner')
        try:
            with self.status_path.open('x', encoding='utf-8') as stream:
                json.dump(dict(rows=len(self.rows), closed=True, restored=not changed,
                    error=None if failure is None else repr(failure), quality_gate_clear=False), stream)
        except BaseException as error:
            failure = failure or error
        if failure is not None and body is None:
            raise failure
        return False
