"""原65入力の後だけ、同NEXT BBの合法な発火配置と人工式入力を追加する。"""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
from types import FunctionType
from typing import Any
import sys

ROOT=Path(__file__).resolve().parent
VERIFY=ROOT.parent
sys.path.insert(0,str(VERIFY/'g2_hidden_repeat_integration_2026-09-10_v1'))
import run_hidden as H
import path_support as P
OLD=H.I
FPS,STRIDE,FIRST,LIVE=OLD.FPS,OLD.STRIDE,OLD.FIRST,OLD.LIVE
NEXT,FIRE,END=34902,34908,34916
FRAMES=tuple(range(OLD.FRAMES[0],END,STRIDE))
BB,BP,PY=(2,2),(2,5),(5,4)
initial_fixture=OLD.initial_fixture
SAVED=VERIFY/'g2_hidden_repeat_integration_2026-09-10_v1/prefix_cpu_v3/COMBINED_HIDDEN.json'


@lru_cache(maxsize=1)
def generated() -> dict[str,Any]:
    from src import puyo_core_bridge as core
    events=json.loads(SAVED.read_bytes())['events']
    cert=[e['current'] for e in events if e['kind']=='conditional_current_after_original_J'][-1]
    before=P.grid(cert['inferred_grid'])
    assert cert['frame']==OLD.FRAMES[-1] and core.simulate_chain(P.board(core,before)).chain_count==0
    options=P.options(core,before,BB)
    for index,final in enumerate(options):
        prediction=core.simulate_chain(P.board(core,final))
        if not prediction.chain_count or final[0]!=before[0]: continue
        raw=tuple(tuple(P.UNKNOWN if r==0 and c==0 else v for c,v in enumerate(row)) for r,row in enumerate(final))
        if tuple(g for g in options if P.compatible(raw,g))!=(final,): continue
        return dict(before=before,placed=final,raw=raw,chains=prediction.chain_count,
            final=prediction.final_board.to_dict()['grid'],option_index=index,option_count=len(options),
            artificial=True,physical_certified=False,original_conditional_frame=cert['frame'])
    raise ValueError('unique_hidden_firing_fixture_missing')


def saved() -> Any:
    base,raw=OLD.saved()
    prior=raw[OLD.FRAMES[-1]]
    raw.update({f:generated()['raw'] if f>=FIRE else prior for f in FRAMES if f>OLD.FRAMES[-1]})
    return base,raw


def geometry(types: Any, invocation: Any, side: str) -> Any:
    if invocation.frame<NEXT: return OLD.geometry(types,invocation,side)
    frame,epoch=invocation.frame,invocation.runtime.histories[side].epoch
    event=types.MotionCandidate(5,frame,frame,f'artificial-hidden-firing:{epoch}:5') if frame==NEXT else None
    quiet=(frame-STRIDE,frame) if frame==NEXT+2*STRIDE else None
    return types.MotionObservation(invocation,side,epoch,f'artificial-hidden-segment:{epoch}',
        frame,invocation.time_sec,quiet,event)


def pairs(frame: int) -> Any:
    return OLD.pairs(frame) if frame<NEXT else (BP,PY)


def install(*args: Any) -> Any:
    return FunctionType(OLD.install.__code__,dict(vars(OLD),saved=saved,geometry=geometry,pairs=pairs))(*args)
