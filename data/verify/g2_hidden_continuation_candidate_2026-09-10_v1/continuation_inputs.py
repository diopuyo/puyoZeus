"""次BPの合法配置から作る人工raw。元47保存rawは一切変更しない。"""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
from types import FunctionType
from typing import Any
import lifetime_inputs as OLD
import path_support as P

FPS,STRIDE,FIRST,NEXT = OLD.FPS,OLD.STRIDE,OLD.FIRST,OLD.NEXT
LAND,END = 34870,34886
FRAMES = tuple(range(OLD.FRAMES[0],END,STRIDE))
LIVE,initial_fixture = OLD.LIVE,OLD.initial_fixture
SAVED = Path(__file__).resolve().parent.parent/'g2_hidden_current_lifetime_candidate_2026-09-10_v1/prefix_cpu_v2/CONDITIONAL_LIFETIME.json'


@lru_cache(maxsize=1)
def generated() -> tuple[Any,Any,Any]:
    from src import puyo_core_bridge as core
    events = json.loads(SAVED.read_bytes())['events']
    source = [e['current'] for e in events if e['kind']=='conditional_current_after_original_J'][-1]
    before = P.grid(source['inferred_grid'])
    options = P.options(core,before,OLD.OLD.BP)
    for final in options:
        if final[0]!=before[0] or core.simulate_chain(P.board(core,final)).chain_count: continue
        raw = tuple(tuple(P.UNKNOWN if r==0 and c==0 else color for c,color in enumerate(row))
            for r,row in enumerate(final))
        if tuple(g for g in options if P.compatible(raw,g))==(final,):
            return before,final,raw
    raise ValueError('artificial_continuation_unique_nonfiring_missing')


def saved() -> tuple[Any,Any]:
    base,raw = OLD.saved()
    _,_,final_raw = generated()
    raw.update({f:final_raw for f in FRAMES if f>=LAND})
    return base,raw


def geometry(types: Any, invocation: Any, side: str) -> Any:
    old = OLD.geometry(types,invocation,side)
    if invocation.frame!=LAND: return old
    return types.MotionObservation(invocation,side,old.software_epoch,
        f'artificial-hidden-segment:{old.software_epoch}',LAND,invocation.time_sec,(LAND-STRIDE,LAND),None)


def install(q: Any, stack: Any, pipe: Any, pixels: Any, types: Any, clock: Any) -> Any:
    # 寿命版のNEXT注入は再用し、盤面/quiet入力だけをこの人工着地へ差し替える。
    return FunctionType(OLD.install.__code__,dict(vars(OLD),saved=saved,geometry=geometry))(
        q,stack,pipe,pixels,types,clock)
