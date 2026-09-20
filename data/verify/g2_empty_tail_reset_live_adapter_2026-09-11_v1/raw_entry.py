"""実collectorのraw guardへ、検収した次update入口二frameだけを接続。"""
from __future__ import annotations
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import CodeType
from typing import Any

def proof_module() -> Any:
    alias = '_live_empty_reset_proof'
    path = Path(__file__).resolve().parent/'proof.py'
    if alias in sys.modules:
        value = sys.modules[alias]
        if Path(value.__file__).resolve()!=path: raise RuntimeError('live_proof_alias_collision')
        return value
    spec = importlib.util.spec_from_file_location(alias,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    spec.loader.exec_module(value)
    return value


P = proof_module()

RAW_SHA = 'cb4d17b5a9787fde4bb656686b6988999b1cdc2bd4801992408bb4e378a84151'


def authenticate(bridge: Any) -> None:
    function = type(bridge).update
    path = Path(function.__code__.co_filename)
    raw = path.read_bytes()
    P.require(hashlib.sha256(raw).hexdigest()==RAW_SHA,'raw_original_source')
    module_code = compile(raw,str(path),'exec',dont_inherit=True)
    cls = next(c for c in module_code.co_consts if isinstance(c,CodeType) and c.co_name=='Bridge')
    actual = next(c for c in cls.co_consts if isinstance(c,CodeType) and c.co_name=='update')
    P.require(function.__code__==actual and function.__globals__['Bridge'] is type(bridge),'raw_original_code')
    P.require(bridge.collector.collect_lean.__code__ is bridge.code
              and bridge.collector.collect_lean.__globals__ is vars(bridge.collector),'raw_original_collector')


def actual_caller(bridge: Any, entry_module: Any, caller: Any, pipe: Any, state: Any) -> Any:
    if caller.f_code is bridge.code: return caller
    P.require(caller.f_code is entry_module.invoke.__code__,'raw_unknown_outer_caller')
    wrapper = caller.f_back
    method = pipe.update
    P.require(inspect.ismethod(method) and method.__self__ is pipe
              and wrapper.f_code is method.__func__.__code__,'raw_actual_entry_wrapper')
    actual = wrapper.f_back
    P.require(actual.f_code is bridge.code and actual.f_globals is vars(bridge.collector),'raw_original_outer_frame')
    entry = caller.f_locals['entry']
    P.require(entry is wrapper.f_locals['entry'] and entry.pipe is pipe and entry.state is state
              and entry.done and entry.error is None,'raw_actual_entry_state')
    P.require(caller.f_locals['original'] is wrapper.f_locals['original']
              and caller.f_locals['original'].__self__ is pipe,'raw_original_bound_update')
    return actual


def install(stack: Any, state: Any, pipe: Any, entry_module: Any) -> Any:
    bridge = state['raw_input_bridge']
    authenticate(bridge)
    P.require('update' not in vars(bridge) and bridge.active is None and bridge.last_frame is None,'raw_attach_boundary')
    original,class_update = bridge.update,type(pipe).update
    counter = dict(completed_updates=0,error=None)
    def update(actual: Any, caller: Any, subject: Any, frame: int, clock: float, pixels: Any) -> Any:
        try:
            P.require(subject is pipe and type(pipe).update is class_update,'raw_actual_pipeline')
            caller = actual_caller(bridge,entry_module,caller,pipe,state)
            value = original(actual,caller,subject,frame,clock,pixels)
            counter['completed_updates'] += 1
            return value
        except BaseException as exc:
            counter['error'] = bridge.error = repr(exc)
            raise
    stack.callback(vars(bridge).pop,'update',None)
    bridge.update = update
    state['live_empty_raw_count'] = counter
    return counter
