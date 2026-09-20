"""既存witnessの実生成関数探索を再用して、原J対象namespaceだけへ接続する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import _g2_inflight_guard as G

ROOT=Path(__file__).resolve().parent
WITNESS=ROOT.parent/'g2_hsv_correction_witness_2026-09-08_v1/observer.py'
WITNESS_SHA='7b3407ccbc46bd5ad48fdcc1fc15810ab3dce2c39866ab42b0d5da3d1084642d'


def install(stack: Any,recovery: Any,state: dict[str,Any]) -> dict[str,Any]:
    G.require('inflight_quarantine' not in state,'duplicate_connection')
    G.require(hashlib.sha256(WITNESS.read_bytes()).hexdigest()==WITNESS_SHA,'witness_source')
    modules=[m for m in tuple(sys.modules.values()) if m is not None and getattr(m,'__file__',None)
             and Path(m.__file__).resolve()==WITNESS.resolve()]
    G.require(len(modules)==1,'actual_witness_module')
    functions=modules[0].step_functions(type(recovery.pipe)._step_side,recovery.journal.codes)
    G.require(set(functions)==recovery.journal.codes and len(functions)==2,'actual_two_codes')
    namespaces={id(fn.__globals__):fn.__globals__ for fn in functions.values()}
    guards=[]
    refs=[(ns,ns['infer_placement']) for ns in namespaces.values()]
    status=dict(installed=False,closed=False,references_restored=False,quality_gate_clear=False)
    state['inflight_quarantine']=status
    def close() -> None:
        status.update(closed=True,references_restored=all(ns['infer_placement'] is old for ns,old in refs),
                      namespace_count=len(refs),code_count=len(functions),rows=[r for g in guards for r in g.rows])
        with (state['output']/'INFLIGHT_QUARANTINE.json').open('x',encoding='utf-8') as stream:
            json.dump(status,stream,ensure_ascii=False,indent=2,allow_nan=False)
        G.require(status['references_restored'],'connection_restore')
    stack.callback(close)
    for ns in namespaces.values(): guards.append(G.install(stack,recovery,ns))
    status['installed']=True
    return status
