"""人工原65保持と一回配置の有限原collector検証。"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from types import FunctionType,SimpleNamespace as N
from typing import Any
import importlib.util
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'g2_conditional_firing_admission_2026-09-10_v1'
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('_conditional_admission_driver',SOURCE/'run_cpu.py')
R=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=R
spec.loader.exec_module(R)
import commit as C
E=R.E


def verify(factory: Any, pipe: Any, data: Any) -> dict[str,Any]:
    binding=factory.controller.history['1P']
    rows=[r for r in data['admission'] if r['stage']=='conditional_placement_committed']
    assert len(rows)==1 and rows[0]['frame']==R.LAST
    state=binding.owner.state
    assert sum(state.counter)==69 and len(state.history)==5 and state.action==5
    assert not state.origins and not state.debts
    assert sum(factory.controller.inventory.S.color_counts(state.current.grid))==61
    assert not pipe._pending_tsumo_1p and binding.next_token is None
    assert getattr(binding,'firing_ticket',None) is None and pipe._active_chain_1p is None
    assert binding.clear_count==0 and binding.clear_first is None
    return dict(original65_verified=data['prefix65'],placement_count=1,state=asdict(state),
        current=binding.current,inferred=binding.grid,original_pop=True,original_event=False,
        physical_certified=False,settled=False,current_permission=False)


def drive(*args: Any) -> Any:
    helper=N(**(vars(R.A)|dict(install=C.install)))
    return FunctionType(R.drive.__code__,dict(vars(R),A=helper,verify=verify))(*args)


def guards() -> Any:
    return R.guards()|{str(p):E.K.sha(p) for p in SOURCE.glob('*.py')}


def main() -> int:
    helper=N(**(vars(E.K)|dict(guards=guards,load=E.load)))
    return FunctionType(E.R.main.__code__,dict(vars(E.R),ROOT=ROOT,K=helper,I=R.INPUT,drive=drive))()


if __name__=='__main__': raise SystemExit(main())
