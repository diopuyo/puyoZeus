"""実RowStreamの保存成功後の最終二票だけを保持する。票の生成はしない。"""
from __future__ import annotations
from collections import deque
from types import MethodType
from typing import Any
import entry_boundary as E

KEY = 'reset_metadata_tail'


class Tail:
    def __init__(self, sink: Any) -> None:
        self.sink, self.stream = sink, sink.rows
        self.rows: Any = deque(maxlen=len(E.SIDES))
        self.count = self.stream.count

    def check(self, frame: int) -> None:
        sink = self.sink
        E.require(sink.rows is self.stream and not sink.busy and not sink.closed and not sink.errors,
                  'metadata_not_ready')
        E.require(self.count == self.stream.count and len(self.rows) == len(E.SIDES), 'metadata_missing')
        E.require([(r['frame_idx'],r['side']) for r in self.rows] == [(frame,s) for s in E.SIDES],
                  'metadata_previous_frame_sides')
        E.require(all(r['time_sec']==frame/E.FPS and 'returned' in r for r in self.rows), 'metadata_clock_or_return')
        self.stream.stream.flush()


def install(stack: Any, state: Any) -> Tail:
    E.require(KEY not in state, 'metadata_tail_reentry')
    sink = state['collector_metadata_sink']
    row_stream = sink.rows
    E.require(type(row_stream).__name__=='RowStream' and row_stream.count==0, 'metadata_stream_start')
    E.require(not sink.busy and not sink.closed and not sink.errors, 'metadata_not_ready')
    tail, original = Tail(sink), row_stream.append
    E.require('append' not in vars(row_stream), 'metadata_stream_already_wrapped')
    def append(self: Any, row: Any) -> None:
        original(row)
        E.require(self is row_stream and self.count==tail.count+1, 'metadata_stream_sequence')
        tail.rows.append(row)
        tail.count = self.count
    stack.callback(vars(row_stream).pop, 'append', None)
    row_stream.append = MethodType(append, row_stream)
    state[KEY] = tail
    return tail


def completed(state: Any, frame: int) -> None:
    E.require(KEY in state, 'metadata_tail_absent')
    tail = state[KEY]
    E.require(tail.sink is state['collector_metadata_sink'], 'metadata_sink_replaced')
    tail.check(frame)
