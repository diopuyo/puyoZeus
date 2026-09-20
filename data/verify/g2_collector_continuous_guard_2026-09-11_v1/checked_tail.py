"""元Sink/RowStreamの型・source・時計・保存wrapperの存続を検査する。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from types import FunctionType
from typing import Any
import continuous_guard as G
import metadata_tail as M

BOUNDED_SHA = 'ae789cbd9e2337c622ac9ead4dcea9741d7d76a5a57e91db74d1f20bfc418691'


class Tail(M.Tail):
    def __init__(self, sink: Any) -> None:
        module = sys.modules[type(sink).__module__]
        G.require(type(sink) is module.Sink and type(sink.rows) is module.RowStream,'metadata_actual_types')
        G.require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==BOUNDED_SHA,'metadata_actual_source')
        self.original = module.O
        G.require((self.original.FPS,self.original.STRIDE,self.original.SIDES)==(M.E.FPS,M.E.STRIDE,M.E.SIDES),
                  'metadata_actual_clock')
        super().__init__(sink)
        self.callback: Any = None
    def check(self, frame: int) -> None:
        G.require(self.stream.append is self.callback,'metadata_append_wrapper_changed')
        G.require((self.original.FPS,self.original.STRIDE,self.original.SIDES)==(M.E.FPS,M.E.STRIDE,M.E.SIDES),
                  'metadata_actual_clock')
        super().check(frame)


def install(stack: Any, state: Any) -> Tail:
    original = FunctionType(M.install.__code__,dict(vars(M),Tail=Tail))
    tail = original(stack,state)
    tail.callback = tail.stream.append
    return tail
