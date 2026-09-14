"""条件付き全installerを原110通常二連鎖へ同居させる。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType,SimpleNamespace as N
from typing import Any
import sys

ROOT=Path(__file__).resolve().parent
NEXT=ROOT.parent/'g2_conditional_after_settlement_next_2026-09-10_v1'
sys.path.insert(0,str(NEXT))
import bundle as B
import next_connection as NEW
import formula_dispatch as DISPATCH

H=B.E.H.B
COMMON=B.E.H.C
OLD=H.REPEAT/'cpu_v2/FIRING.json'


def guards() -> dict[str,str]:
    paths=[*ROOT.glob('*.py'),*NEXT.glob('*.py'),*B.SETTLED.glob('*.py'),OLD,DISPATCH.FIXED,DISPATCH.SOURCE]
    return COMMON.guards()|B.R.guards()|{str(p):H.K.sha(p) for p in paths}


def installed(stack: Any,factory: Any,pipe: Any,state: Any,rows: Any) -> None:
    COMMON.install(stack,factory,pipe,state,rows)
    original=pipe._apply_chain_formula_early_fire
    NEW.install(stack,factory,pipe,H.TAIL.patch,state['conditional_full_rows'])
    DISPATCH.install(stack,factory,pipe,H.TAIL.patch,original,state['conditional_full_rows'])
    state['conditional_full_installs']+=1


def drive(original: Any,*args: Any) -> Any:
    state,factory=args[3],args[7]
    state['conditional_full_rows']=[]; state['conditional_full_installs']=0
    candidate=N(**(vars(original.C)|dict(install=installed)))
    try:
        result=FunctionType(original.drive.__code__,dict(vars(original),C=candidate))(*args)
        COMMON.accepted(state,factory,original.I.FRAMES[-1])
        control=factory.controller
        assert not control.hidden_history_rows and not control.hidden_current_outputs
        assert not control.conditional_next_hands
        assert all(getattr(b,'conditional_firing_registered',None) is None for b in control.history.values())
        assert state['conditional_full_installs']==1 and not state['conditional_full_rows']
        return result
    finally:
        COMMON.saved(state,factory)
        H.K.write(state['output']/'CONDITIONAL_FULL.json',dict(
            installs=state['conditional_full_installs'],rows=state['conditional_full_rows'],
            conditional_checkpoint_count=len(getattr(factory.controller,'conditional_next_hands',{})),
            physical_certified=False,quality_gate_clear=False))


def accepted(original: Any,output: Path) -> Any:
    result=original.accepted(output)
    receipt=H.K.read(output/'CONDITIONAL_FULL.json')
    assert receipt['installs']==1 and not receipt['rows'] and receipt['conditional_checkpoint_count']==0
    timeline=lambda rows: [(r['stage'],r['frame']) for r in rows]
    rows=H.K.read(output/'FIRING.json')
    assert timeline(rows)==timeline(H.K.read(OLD)) and len(rows)==11
    result.update(conditional_full_installed_once=True,conditional_non_target=True,
        original_11_stage_timeline_equal=True)
    return result


def main() -> int:
    with ExitStack() as stack:
        original=H.K.load('_conditional_full_original110',H.REPEAT/'run_cpu.py',stack)
        def body(*args: Any) -> Any: return drive(original,*args)
        def accept(output: Path) -> Any: return accepted(original,output)
        source=original.P.Q.R
        execute=FunctionType(source.execute.__code__,dict(vars(source),I=original.I,
            MAX_UPDATES=original.MAX_UPDATES,drive=body))
        common=N(**(vars(original.K)|dict(guards=guards)))
        return FunctionType(original.C.G.R.main.__code__,dict(vars(original.C.G.R),K=common,
            ROOT=ROOT,execute=execute,accepted=accept))()


if __name__=='__main__': raise SystemExit(main())
