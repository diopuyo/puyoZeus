"""原抽出loopのsource認証・実callee寿命・元例外を入口で守る。"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import sys
from types import FunctionType
from typing import Any

OLD = Path(__file__).resolve().parent.parent/'g2_empty_tail_desync_trigger_2026-09-11_v1'
sys.path.insert(0,str(OLD))
import collector_continuous as C


def require(ok: bool, reason: str) -> None:
    if not ok: raise ValueError('collector_continuous:'+reason)


def extracted(source: Any) -> Any:
    data = source.read_bytes()
    require(hashlib.sha256(data).hexdigest()==C.SOURCE_SHA,'collector_source_changed')
    original = next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='collect_lean')
    first = next(i for i,n in enumerate(original.body) if ast.unparse(n).startswith('acc = _LeanNpzAccumulator('))
    last = next(i for i,n in enumerate(original.body) if isinstance(n,ast.For) and ast.unparse(n.target)=='local_i')
    setup,loop = original.body[first:last],original.body[last]
    calls = [n for n in ast.walk(loop) if isinstance(n,ast.Call) and ast.unparse(n.func)=='_process_side_lean']
    require(len(calls)==2 and [ast.literal_eval(n.args[2]) for n in calls]==['1P','2P'],'collector_side_calls')
    function = ast.parse('def continuous(start_frame: int = 0) -> object:\n    pass').body[0]
    wait = ast.parse('while True:\n    '+C.INPUTS+' = yield locals()\n').body[0]
    wait.body.append(loop)
    function.body = setup+[wait]
    return ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))


def build(collector: Any, source: Any, overrides: dict[str,Any] | None = None) -> Any:
    original = FunctionType(C.build.__code__,dict(vars(C),extracted=extracted),argdefs=C.build.__defaults__)
    loop = original(collector,source,overrides)
    callee = collector._process_side_lean
    values = loop.generator.gi_frame.f_globals
    loop.source_collector = collector
    loop.error = None
    def collect_lean(cap: Any, pipeline: Any, start_frame: int, n_frames: int,
                     effective_interval_frames: int, fps: float) -> None:
        if loop.error is not None: raise loop.error
        try:
            require(collector._process_side_lean is callee and values['_process_side_lean'] is callee,
                    'collector_callee_changed')
            require(loop.generator.gi_frame is not None,'collector_generator_closed')
            loop.runtime_state = loop.generator.send((cap,pipeline,start_frame,n_frames,effective_interval_frames,fps))
        except BaseException as exc:
            loop.error = exc
            raise
    loop.collect_lean = collect_lean
    return loop
