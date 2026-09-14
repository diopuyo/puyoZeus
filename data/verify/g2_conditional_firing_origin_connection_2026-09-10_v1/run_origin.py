"""原70更新でconditional origin登録まで確認する。"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from types import FunctionType,SimpleNamespace as N
from typing import Any
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parent
COMMIT=ROOT.parent/'g2_conditional_firing_commit_2026-09-10_v1'
POLICY=ROOT.parent/'g2_conditional_firing_origin_policy_2026-09-10_v1'
sys.path[:0]=[str(COMMIT),str(POLICY)]
spec=importlib.util.spec_from_file_location('_conditional_commit_driver',COMMIT/'run_commit.py')
D=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=D
spec.loader.exec_module(D)
import origin_connection as C
E=D.E


def verify(factory: Any, pipe: Any, data: Any) -> dict[str,Any]:
    binding=factory.controller.history['1P']; state=binding.owner.state
    registered=[r for r in data['admission'] if r['stage']=='conditional_origin_registered']
    withheld=[r for r in data['admission'] if r['stage']=='conditional_origin_current_withheld']
    assert len(registered)==len(withheld)==1 and registered[0]['frame']==D.R.LAST
    assert sum(state.counter)==69 and len(state.history)==5 and state.action==5
    assert len(state.origins)==len(state.debts)==1 and not state.consumed_ids
    assert state.origins[0].event_identity.startswith('conditional_world:')
    assert state.clock.sequence==D.R.LAST*4+2 and state.origins[0].available_at==state.clock
    assert sum(factory.controller.inventory.S.color_counts(state.current.grid))==61
    assert not pipe._pending_tsumo_1p and pipe._active_chain_1p is None
    assert getattr(binding,'firing_ticket',None) is None and binding.hidden_current is None
    return dict(original65_verified=data['prefix65'],placement_count=1,origin_count=1,
        state=asdict(state),current=binding.current,inferred=binding.grid,original_pop=True,
        original_event=False,physical_certified=False,settled=False,current_permission=False)


def drive(*args: Any) -> Any:
    helper=N(**(vars(D.C)|dict(install=C.install)))
    return FunctionType(D.drive.__code__,dict(vars(D),C=helper,verify=verify))(*args)


def guards() -> Any:
    names=('origin_evidence.py','origin_fixed.py','origin_policy.py','origin_policy_v2.py','origin_registration.py')
    backend=POLICY/'cpu_v2/BACKEND.json'
    paths=[*COMMIT.glob('*.py'),*(POLICY/n for n in names),backend]
    return D.guards()|{str(p):E.K.sha(p) for p in paths}|json.loads(backend.read_text())['files']


def main() -> int:
    helper=N(**(vars(E.K)|dict(guards=guards,load=E.load)))
    return FunctionType(E.R.main.__code__,dict(vars(E.R),ROOT=ROOT,K=helper,I=D.R.INPUT,drive=drive))()


if __name__=='__main__': raise SystemExit(main())
