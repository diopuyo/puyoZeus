"""実raw列を再用し人工NEXT時計へ束縛する47更新。実video再生ではない。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LIVE = ROOT.parent/'video38_history_publication_probe_live_2026-09-10_v7'
FRAMES = tuple(range(34772,34866,2))
FPS, STRIDE, FIRST, SECOND, OFFSET, RAW_FIRST = 60, 2, 34800, 34836, 202, 34802
PY, BB, BP = (5,4),(2,2),(2,5)
QUIET = (34776,34804,34840)


def saved() -> tuple[Any,dict[int,Any]]:
    with (LIVE/'directional_history.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    base = next(r['grid_after'] for r in rows if r['scope']['frame_idx']==35002)
    raw = {r['scope']['frame_idx']-OFFSET:r['raw_capture']['raw']['grid'] for r in rows
           if 35004 <= r['scope']['frame_idx'] <= 35066}
    assert all(f in raw for f in FRAMES if f >= RAW_FIRST)
    return base,raw


def geometry(types: Any, invocation: Any, side: str) -> Any:
    frame,epoch = invocation.frame,invocation.runtime.histories[side].epoch
    number = {FIRST:1,SECOND:2}.get(frame)
    event = None if number is None else types.MotionCandidate(number,frame,frame,
        f'artificial-hidden-prefix:{epoch}:{number}')
    quiet = (frame-STRIDE,frame) if frame in QUIET else None
    return types.MotionObservation(invocation,side,epoch,f'artificial-hidden-segment:{epoch}',
        frame,invocation.time_sec,quiet,event)


def initial_fixture(pipe: Any, factory: Any) -> dict[str,Any]:
    """最初のupdateより前の人工初期条件。実run checkpoint復元ではない。"""
    from src.board import Board
    sm = pipe._sm_1p
    assert not factory.controller.history and sm.context.state.value=='menu'
    assert sm.context.confirmed_board is None
    base,_ = saved()
    sm.context.confirmed_board = Board.from_dict({'grid':[list(r) for r in base]})
    return dict(artificial_initial_condition=True,before_first_original_update=True,
        initial_confirmed=base,state_value_unchanged=sm.context.state.value=='menu',
        initial_state='menu',stable_transition_forced=False,
        original_video_checkpoint=False,source_frame_is_fixture_content_only=35002)


def install(q: Any, stack: Any, pipe: Any, pixels: Any, types: Any, clock: Any) -> list[Any]:
    base,raw = saved()
    helper = q.fixture_helpers(type(pipe._next_detector).detect_both)
    original,call,read = helper.actual_next,type(pixels).__call__,pipe._reader.read_both_boards
    rows: list[Any] = []
    def next_value(pair: Any, other: Any = None) -> Any:
        result,frame = original(pair,other),clock['frame']
        first,second = (PY,BB) if frame<FIRST else (BB,BP) if frame<SECOND else (BP,BB)
        return type(result)(type(result.p1)(*first,*second),result.p2)
    def observed(self: Any, invocation: Any, side: str) -> Any:
        result = call(self,invocation,side)
        if self is not pixels or side!='1P': return result
        value = geometry(types,invocation,side)
        rows.append(dict(frame=value.frame_idx,epoch=value.software_epoch,quiet=value.quiet_frames,
            candidate=None if value.candidate is None else vars(value.candidate),artificial=True))
        return value
    def boards(image: Any, **kwargs: Any) -> Any:
        first,second = read(image,**kwargs)
        value = raw[clock['frame']] if clock['frame']>=RAW_FIRST else base
        first = type(first).from_dict({'grid':[list(row) for row in value]})
        return first,second
    for obj,name,old in ((helper,'actual_next',original),(type(pixels),'__call__',call),
                         (pipe._reader,'read_both_boards',read)):
        stack.callback(setattr,obj,name,old)
    helper.actual_next,type(pixels).__call__,pipe._reader.read_both_boards = next_value,observed,boards
    return rows
