"""元47保存raw＋人工次NEXT一手。未来rawを実動画観測と呼ばない。"""
from __future__ import annotations
from types import FunctionType
from typing import Any
import prefix_inputs as OLD

FPS,STRIDE,FIRST = OLD.FPS,OLD.STRIDE,OLD.FIRST
NEXT,END,REPEAT_SOURCE = 34866,34876,34862
FRAMES = tuple(range(OLD.FRAMES[0],END,STRIDE))
LIVE = OLD.LIVE
initial_fixture = OLD.initial_fixture


def saved() -> tuple[Any,Any]:
    base,raw = OLD.saved()
    raw.update({f:raw[REPEAT_SOURCE] for f in FRAMES if f>OLD.FRAMES[-1]})
    return base,raw


def geometry(types: Any, invocation: Any, side: str) -> Any:
    old = OLD.geometry(types,invocation,side)
    if invocation.frame!=NEXT: return old
    candidate = types.MotionCandidate(3,NEXT,NEXT,f'artificial-hidden-next:{old.software_epoch}:3')
    return types.MotionObservation(invocation,side,old.software_epoch,
        f'artificial-hidden-segment:{old.software_epoch}',NEXT,invocation.time_sec,None,candidate)


def install(q: Any, stack: Any, pipe: Any, pixels: Any, types: Any, clock: Any) -> Any:
    original = FunctionType(OLD.install.__code__,dict(vars(OLD),saved=saved,geometry=geometry))
    rows = original(q,stack,pipe,pixels,types,clock)
    helper = q.fixture_helpers(type(pipe._next_detector).detect_both)
    previous = helper.actual_next
    def actual_next(pair: Any, other: Any = None) -> Any:
        result = previous(pair,other)
        if clock['frame']<NEXT: return result
        return type(result)(type(result.p1)(*OLD.BB,*OLD.BP),result.p2)
    stack.callback(setattr,helper,'actual_next',previous)
    helper.actual_next = actual_next
    return rows
