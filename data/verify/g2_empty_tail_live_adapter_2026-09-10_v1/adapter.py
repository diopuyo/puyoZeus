"""旧rolling実接続を保持し、空tail後発配置と厳格終了検査だけ追加する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace as N
from typing import Any

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'g2_rolling_live_adapter_2026-09-10_v1'
EMPTY=ROOT.parent/'g2_empty_tail_next_2026-09-10_v1'
FINAL=ROOT.parent/'g2_empty_tail_finalizer_2026-09-10_v1'
KEY='empty_tail_runtime'


def load(alias: str, path: Path, stack: Any) -> Any:
    assert alias not in sys.modules
    spec=importlib.util.spec_from_file_location(alias,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[alias]=module
    stack.callback(sys.modules.pop,alias,None)
    spec.loader.exec_module(module)
    return module


def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
    stack.callback(setattr,obj,name,getattr(obj,name))
    setattr(obj,name,value)


def targets(factory: Any, parts: Any) -> Any:
    cls=type(factory.controller)
    return (cls.hand,cls.prepared.__globals__['V1'].prepared,cls.call,cls.consumed_history,
        parts.exits.N.eligible,parts.exits.C.make)


def watch(stack: Any, factory: Any, parts: Any, state: Any, write: Any) -> None:
    before=targets(factory,parts)
    value=dict(installed=False,closed=False,restored=False)
    state[KEY]=factory.controller.empty_runtime_references=value
    def close() -> None:
        after=targets(factory,parts)
        value.update(closed=True,restored=all(a is b for a,b in zip(before,after)))
        write(state['output']/'EMPTY_TAIL_REFERENCES.json',value)
        assert value['restored'],'empty_tail_reference_restore'
    stack.callback(close)


def connect(stack: Any, basis: Any, connection: Any, reuse: Any, completion: Any) -> None:
    extra=sys.modules['_private_live_extra']
    original=extra.install
    def install(inner: Any, factory: Any, pipe: Any, state: Any, parts: Any, write: Any) -> None:
        original(inner,factory,pipe,state,parts,write)
        assert KEY not in state
        watch(inner,factory,parts,state,write)
        placement,exits=reuse.adapted(parts.placement),reuse.adapted(parts.exits)
        connection.install(inner,factory,parts.patch,basis,placement,exits,factory.controller.hidden_history_rows)
        completion.install(inner,factory,basis,placement,parts.patch,parts.completion,write,reuse)
        factory.controller.empty_completion_parts=N(C=parts.completion,module=completion)
        state[KEY]['installed']=True
    patch(stack,extra,'install',install)


def configured(stack: Any) -> Any:
    old=load('_empty_live_previous',PREVIOUS/'adapter.py',stack)
    main=old.configured(stack)
    previous=list(sys.path)
    stack.callback(sys.path.__setitem__,slice(None),previous)
    sys.path[:0]=[str(FINAL),str(EMPTY)]
    for alias in ('_empty_live_basis_prefix','_empty_live_basis_content'):
        before=sys.modules.get(alias)
        if before is None: stack.callback(sys.modules.pop,alias,None)
        else: stack.callback(sys.modules.__setitem__,alias,before)
    basis=load('_empty_live_basis',EMPTY/'empty_basis.py',stack)
    connection=load('_empty_live_connection',EMPTY/'connection.py',stack)
    reuse=load('_empty_live_reuse',EMPTY/'reuse.py',stack)
    completion=load('_empty_live_completion',FINAL/'empty_completion.py',stack)
    finalizer=load('_empty_live_finalizer',ROOT/'live_finalizer.py',stack)
    connect(stack,basis,connection,reuse,completion)
    target=sys.modules['_private_live_finalizer']
    original=target.audit
    def audit(*args: Any, **kwargs: Any) -> Any:
        return finalizer.audit(original,*args,**kwargs)
    patch(stack,target,'audit',audit)
    common=main.__globals__['K']
    old_guards=common.guards
    pins={str(p):common.sha(p) for folder in (ROOT,EMPTY,FINAL) for p in folder.glob('*.py')}
    def guards() -> Any:
        assert all(common.sha(Path(p))==digest for p,digest in pins.items()),'empty_source_changed'
        return old_guards()|pins
    helper=N(**(vars(common)|dict(ROOT=ROOT,guards=guards)))
    return FunctionType(main.__code__,dict(main.__globals__,K=helper),main.__name__,main.__defaults__,main.__closure__)
