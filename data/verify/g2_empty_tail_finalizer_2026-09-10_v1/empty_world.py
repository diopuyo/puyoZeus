"""旧私有world検査を外側へ重ね、新kindと新current sourceを検査する。"""
from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace as N
from typing import Any,Iterator

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'g2_private_suffix_world_2026-09-10_v1/world.py'
KIND='hidden_empty_tail_next_history/v1'
SOURCE_KEY='empty_tail_committed_source'
SOURCE_SHA='5c2cd1ccbdec37a1343a4741fa96e77ecca332c3830d3f20b4a4f90391d98563'


def clone(function: Any, **changed: Any) -> Any:
    value=FunctionType(function.__code__,dict(function.__globals__,**changed),
        function.__name__,function.__defaults__,function.__closure__)
    value.__kwdefaults__=function.__kwdefaults__
    return value


@contextmanager
def loaded() -> Iterator[Any]:
    alias='_empty_world_original_private'
    assert alias not in sys.modules
    before=sha256(SOURCE.read_bytes()).hexdigest()
    assert before==SOURCE_SHA
    spec=importlib.util.spec_from_file_location(alias,SOURCE)
    old=importlib.util.module_from_spec(spec)
    sys.modules[alias]=old
    try:
        spec.loader.exec_module(old)
        yield old
    finally:
        sys.modules.pop(alias,None)
        assert sha256(SOURCE.read_bytes()).hexdigest()==before


def adapted(old: Any) -> Any:
    W=old.libraries()
    geometry=N(**(vars(W.G)|dict(histories=lambda *args:old.histories(W,*args))))
    previous=N(**(vars(W)|dict(G=geometry,
        hidden_histories=lambda *args:old.hidden_histories(W,*args),
        current_sources=lambda *args:old.current_sources(W,*args))))
    changes=dict(PRIVATE=KIND,SOURCE_KEY=SOURCE_KEY)
    functions={name:clone(getattr(old,name),**changes)
        for name in ('histories','hidden_histories','current_sources')}
    return clone(old.verify_world,**changes,**functions,libraries=lambda:previous)


def verify_world(**kwargs: Any) -> Any:
    for event in kwargs['hidden_events']:
        if 'current' in event:
            proof=json.loads(event['current']['evidence_json'])
            assert not (SOURCE_KEY in proof and 'private_suffix_committed_source' in proof),'mixed_private_current_sources'
    with loaded() as old:
        result=adapted(old)(**kwargs)
    found=any(r['prepared'] and r['prepared']['kind']==KIND for r in kwargs['history_rows'])
    prior=any(r['prepared'] and r['prepared']['kind']=='hidden_private_suffix_history/v1'
        for r in kwargs['history_rows'])
    return result|dict(private_suffix_geometry_verified=prior,empty_tail_geometry_verified=found,
        proof_kinds_renamed=False,runtime_finalization_allowed=False)
