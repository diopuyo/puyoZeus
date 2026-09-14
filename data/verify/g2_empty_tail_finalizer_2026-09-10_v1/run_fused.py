"""元collector76と当時の実factory/captureを保持して終了検査へ渡す。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType,SimpleNamespace as N
from typing import Any

ROOT=Path(__file__).resolve().parent
VERIFY=ROOT.parent
EMPTY=VERIFY/'g2_empty_tail_next_2026-09-10_v1'
OLD=VERIFY/'g2_private_suffix_fusion_2026-09-10_v1'
ROLLING=VERIFY/'g2_rolling_two_hand_prefix_2026-09-10_v1'
PRIVATE_LIVE=VERIFY/'g2_private_suffix_live_adapter_2026-09-10_v1'


def connector(base: Any, stack: Any, kept: Any) -> Any:
    connected=base.load('_empty_fused_connection',EMPTY/'connection.py',stack)
    completion=base.load('_empty_fused_completion',ROOT/'empty_completion.py',stack)
    C=base.load('_empty_fused_receipt',PRIVATE_LIVE/'completion.py',stack)
    reuse=base.load('_empty_fused_derivation',EMPTY/'reuse.py',stack)
    original=connected.install
    def install(inner: Any, factory: Any, patch: Any, basis: Any, placement: Any,
                exits: Any, rows: Any) -> None:
        original(inner,factory,patch,basis,placement,exits,rows)
        completion.install(inner,factory,basis,placement,patch,C,base.write,reuse)
        kept['empty_parts']=N(basis=basis,placement=placement,completion=completion,C=C)
    connected.install=install
    return connected


def checked_loader(selected: Any, base: Any) -> Any:
    def load(alias: str, path: Path, stack: Any) -> Any:
        module=selected(alias,path,stack)
        if alias=='_tail_suffix_shared_entry':
            original=module.load
            def shared(inner: Any) -> Any:
                facade=original(inner)
                def guards() -> Any:
                    return facade.guards()|{str(p):module.C.R.H.TAIL.R.K.sha(p)
                        for folder in (ROOT,PRIVATE_LIVE) for p in folder.glob('*.py')}
                return N(**(vars(facade)|dict(guards=guards)))
            module.load=shared
        return module
    return load


def execute() -> int:
    with ExitStack() as stack:
        previous=list(sys.path)
        stack.callback(sys.path.__setitem__,slice(None),previous)
        sys.path[:0]=[str(EMPTY),str(OLD),str(ROLLING)]
        spec=importlib.util.spec_from_file_location('_empty_fused_original',OLD/'run_runtime.py')
        base=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        sys.path[:0]=[str(base.RUNTIME),str(base.OBS)]
        original=base.load('_empty_fused_runtime',base.RUNTIME/'run_runtime.py',stack)
        check=base.load('_empty_fused_checks',EMPTY/'run_cpu.py',stack)
        rolling=base.load('_empty_fused_rolling',ROLLING/'connection.py',stack)
        evidence=base.load('_empty_fused_old_evidence',base.EVIDENCE/'observer.py',stack)
        observer=base.load('_empty_fused_observer',base.OBS/'observer.py',stack)
        calls=base.load('_empty_fused_calls',base.LIVE/'call_connection.py',stack)
        kept: dict[str,Any]=dict(runtime=N(ROOT=ROOT,load=base.load,write=base.write))
        connected=connector(base,stack,kept)
        selected=base.loader(original,evidence,observer,calls,kept)
        delegated=checked_loader(check.loader(N(load=selected),rolling,connected),base)
        status=FunctionType(original.main.__code__,dict(vars(original),ROOT=ROOT,
            I=check.I,verify=check.verify,load=delegated))()
        if status==0:
            calls.verify(kept['state'])
            final=base.load('_empty_fused_finish',ROOT/'finish.py',stack)
            final.finish(kept)
        return status


def main() -> int:
    output=ROOT/sys.argv[1]
    assert output.parent==ROOT and not output.exists()
    paths=[p for folder in (ROOT,EMPTY,ROLLING) for p in folder.glob('*.py')]
    pins=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before,started=pins(),time.monotonic()
    try:
        code,error=execute(),None
    except BaseException:
        code,error=1,traceback.format_exc()
    if not output.exists(): output.mkdir()
    unchanged=pins()==before
    result=dict(pid=os.getpid(),seconds=time.monotonic()-started,exit_code=code,error=error,
        source_unchanged=unchanged,quality_gate_clear=False)
    for name,value in (('FUSION_INPUTS.json',before),('FUSION_RESULT.json',result)):
        with (output/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return code or int(not unchanged)


if __name__=='__main__':
    raise SystemExit(main())
