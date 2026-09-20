"""人工原65＋発火二票を原collectorで採録し、pop前で有限終了する。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import FunctionType,SimpleNamespace as N
from typing import Any
import sys

ROOT=Path(__file__).resolve().parent
ENTRY=ROOT.parent/'g2_hidden_firing_entry_diagnosis_2026-09-10_v1'
sys.path.insert(0,str(ENTRY))
import run_entry as E
import admission as A

LAST=34910
INPUT=N(**(vars(E.I)|dict(FRAMES=tuple(f for f in E.I.FRAMES if f<=LAST))))


def verify(factory: Any, pipe: Any, data: Any) -> dict[str,Any]:
    binding=factory.controller.history['1P']
    votes=[r for r in data['admission'] if r['stage']=='conditional_firing_vote']
    assert [(r['frame'],r['count']) for r in votes]==[(34908,1),(LAST,2)]
    proposals=[r for r in votes if 'proposal' in r]
    assert len(proposals)==1 and proposals[0]['proposal']['proof']['kind']==A.KIND
    state=binding.owner.state
    assert sum(state.counter)==67 and len(state.history)==4 and state.action==5
    assert not state.origins and not state.debts
    assert sum(factory.controller.inventory.S.color_counts(state.current.grid))==61
    assert len(pipe._pending_tsumo_1p)==1 and binding.next_token not in binding.consumed_tokens
    assert getattr(binding,'firing_ticket',None) is None and pipe._active_chain_1p is None
    assert binding.clear_count==0 and binding.clear_first is None
    return dict(original65_verified=data['prefix65'],proposal_count=1,state=asdict(state),
        current=binding.current,inferred=binding.grid,pending=list(pipe._pending_tsumo_1p),
        original_pop=False,original_event=False,physical_certified=False,accounting_permission=False)


def drive(*args: Any) -> Any:
    state,pipe,factory=args[2],args[3][0],args[6]
    data=dict(trace=[],front=[],firing=[],admission=[],primary=None,fixture=E.I.generated())
    original=E.H.C.install
    def installed(stack: Any, f: Any, p: Any, s: Any, rows: Any) -> None:
        original(stack,f,p,s,rows)
        A.install(stack,f,p,E.H.B.TAIL.patch,data['admission'])
    helper=N(**(vars(E.H)|dict(C=N(**(vars(E.H.C)|dict(install=installed))))))
    profile=sys.getprofile()
    try:
        with ExitStack() as stack:
            FunctionType(E.loop.__code__,dict(vars(E),I=INPUT,H=helper))(args,stack,data)
            result=verify(factory,pipe,data)
        state['combined_restored']()
        assert sys.getprofile() is profile
        return result
    finally:
        data['profile_restored']=sys.getprofile() is profile
        E.H.C.saved(state,factory)
        E.K.write(state['output']/'ADMISSION.json',data)


def guards() -> Any:
    return E.guards()|{str(p):E.K.sha(p) for p in ENTRY.glob('*.py')}


def main() -> int:
    helper=N(**(vars(E.K)|dict(guards=guards,load=E.load)))
    return FunctionType(E.R.main.__code__,dict(vars(E.R),ROOT=ROOT,K=helper,I=INPUT,drive=drive))()


if __name__=='__main__': raise SystemExit(main())
