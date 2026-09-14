"""原77更新で条件付き精算と自然STABLEの別PB復帰を検証する。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import FunctionType,SimpleNamespace as N
from typing import Any
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parent
ORIGIN=ROOT.parent/'g2_conditional_firing_origin_connection_2026-09-10_v1'
CLEAR=ROOT.parent/'g2_conditional_firing_clear_observation_2026-09-10_v1'
sys.path[:0]=[str(ORIGIN),str(CLEAR)]
spec=importlib.util.spec_from_file_location('_conditional_origin_driver',ORIGIN/'run_origin.py')
R=importlib.util.module_from_spec(spec); sys.modules[spec.name]=R
spec.loader.exec_module(R)
import clear_inputs as I
spec=importlib.util.spec_from_file_location('_conditional_settlement_connection',ROOT/'connection.py')
C=importlib.util.module_from_spec(spec); sys.modules[spec.name]=C
spec.loader.exec_module(C)
E=R.E


def verify(factory: Any, pipe: Any, data: Any) -> dict[str,Any]:
    control=factory.controller; binding=control.history['1P']; state=binding.owner.state
    kinds=('conditional_placement_committed','conditional_origin_registered','conditional_private_settled')
    counts={kind:sum(r.get('stage')==kind for r in data['admission']) for kind in kinds}
    assert all(value==1 for value in counts.values())
    assert sum(state.counter)==64 and len(state.history)==6 and len(state.origins)==1 and not state.debts
    assert state.consumed_ids==(state.origins[0].origin_id,) and state.action==5
    assert sum(control.inventory.S.color_counts(state.current.grid))==61 and not state.current.available
    assert not pipe._pending_tsumo_1p and pipe._active_chain_1p is None
    values=[r['current'] for r in control.hidden_current_events if r.get('kind')=='conditional_current_after_original_J'
        and json.loads(r['current']['evidence_json'])['kind']=='conditional_hidden_after_firing_current/v1']
    assert values and values[-1]['frame']==I.LAST and pipe._sm_1p.context.state.value=='stable'
    assert sum(control.inventory.S.color_counts(values[-1]['inferred_grid']))==64
    assert values[-1]['integer_anchor']==control.inventory.encoded(asdict(state.current))
    assert all(r['physical_certified'] is False for r in values)
    return dict(original65_verified=data['prefix65'],counts=counts,state=asdict(state),
        conditional_frames=[r['frame'] for r in values],original_event=False,physical_certified=False,
        current_permission=False,integer_current_recovered=False)


def drive(*args: Any) -> Any:
    state,pipe,factory=args[2],args[3][0],args[6]
    data=dict(trace=[],front=[],firing=[],admission=[],primary=None,fixture=E.I.generated())
    original=E.H.C.install
    def installed(stack: Any, f: Any, p: Any, s: Any, rows: Any) -> None:
        original(stack,f,p,s,rows)
        C.install(stack,f,p,E.H.B.TAIL.patch,data['admission'])
    helper=N(**(vars(E.H)|dict(C=N(**(vars(E.H.C)|dict(install=installed))))))
    profile=sys.getprofile()
    try:
        with ExitStack() as stack:
            FunctionType(E.loop.__code__,dict(vars(E),I=I.INPUT,H=helper))(args,stack,data)
            result=verify(factory,pipe,data)
        state['combined_restored']()
        assert sys.getprofile() is profile
        return result
    finally:
        data['profile_restored']=sys.getprofile() is profile
        E.H.C.saved(state,factory)
        E.K.write(state['output']/'ADMISSION.json',data)


def guards() -> Any:
    paths=[*ORIGIN.glob('*.py'),CLEAR/'clear_observation.py']
    return R.guards()|{str(p):E.K.sha(p) for p in paths}


def main() -> int:
    helper=N(**(vars(E.K)|dict(guards=guards,load=E.load)))
    return FunctionType(E.R.main.__code__,dict(vars(E.R),ROOT=ROOT,K=helper,I=I.INPUT,drive=drive))()


if __name__=='__main__': raise SystemExit(main())
