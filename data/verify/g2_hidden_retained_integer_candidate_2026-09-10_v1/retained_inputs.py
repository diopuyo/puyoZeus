"""原PP配置で整数current61を作ってから、既存隠し二手系列へ入る人工連続入力。"""
from __future__ import annotations
from functools import lru_cache
import json
from types import FunctionType
from typing import Any
import prefix_inputs as P0
import continuation_inputs as OLD

FPS,STRIDE,FIRST,SHIFT = P0.FPS,P0.STRIDE,P0.FIRST,16
PP = (5,5)
PY_FRAME,BB_FRAME,BP_FRAME = P0.FIRST+SHIFT,P0.SECOND+SHIFT,OLD.NEXT+SHIFT
LAND,END = OLD.LAND+SHIFT,OLD.END+SHIFT
FRAMES = tuple(range(P0.FRAMES[0],END,STRIDE))
LIVE = P0.LIVE
EVENTS = {FIRST:1,PY_FRAME:2,BB_FRAME:3,BP_FRAME:4}
QUIET = (P0.QUIET[0],FIRST+4,PY_FRAME+4,BB_FRAME+4,BP_FRAME+4)


@lru_cache(maxsize=1)
def before_integer() -> Any:
    with (LIVE/'directional_history.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    return next(r['grid_after'] for r in rows if r['scope']['frame_idx']==34940)


def saved() -> tuple[Any,Any]:
    initial,raw = before_integer(),{}
    base,original = OLD.saved()
    for frame in FRAMES:
        raw[frame] = initial if frame<P0.RAW_FIRST else base if frame<=PY_FRAME else original[frame-SHIFT]
    return initial,raw


def initial_fixture(pipe: Any, factory: Any) -> dict[str,Any]:
    from src.board import Board
    sm = pipe._sm_1p
    assert not factory.controller.history and sm.context.state.value=='menu' and sm.context.confirmed_board is None
    base = before_integer()
    sm.context.confirmed_board = Board.from_dict({'grid':[list(r) for r in base]})
    return dict(artificial_initial_condition=True,initial_confirmed=base,initial_state='menu',
        state_value_unchanged=True,stable_transition_forced=False,original_video_checkpoint=False,
        integer_S_slot_not_injected=True,source_frame_is_fixture_content_only=34940)


def geometry(types: Any, invocation: Any, side: str) -> Any:
    frame,epoch = invocation.frame,invocation.runtime.histories[side].epoch
    number = EVENTS.get(frame)
    event = None if number is None else types.MotionCandidate(number,frame,frame,
        f'artificial-hidden-retained:{epoch}:{number}')
    quiet = (frame-STRIDE,frame) if frame in QUIET else None
    return types.MotionObservation(invocation,side,epoch,f'artificial-hidden-segment:{epoch}',
        frame,invocation.time_sec,quiet,event)


def pairs(frame: int) -> tuple[Any,Any]:
    if frame<FIRST: return PP,P0.PY
    if frame<PY_FRAME: return P0.PY,P0.BB
    if frame<BB_FRAME: return P0.BB,P0.BP
    if frame<BP_FRAME: return P0.BP,P0.BB
    return P0.BB,P0.BP


def install(q: Any, stack: Any, pipe: Any, pixels: Any, types: Any, clock: Any) -> Any:
    original = FunctionType(P0.install.__code__,dict(vars(P0),saved=saved,geometry=geometry))
    rows = original(q,stack,pipe,pixels,types,clock)
    helper = q.fixture_helpers(type(pipe._next_detector).detect_both)
    previous = helper.actual_next
    def actual_next(pair: Any, other: Any = None) -> Any:
        result = previous(pair,other)
        first,second = pairs(clock['frame'])
        return type(result)(type(result.p1)(*first,*second),result.p2)
    stack.callback(setattr,helper,'actual_next',previous)
    helper.actual_next = actual_next
    return rows
