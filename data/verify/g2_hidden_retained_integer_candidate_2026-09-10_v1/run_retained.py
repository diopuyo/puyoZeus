"""旧整数currentを原PP履歴/原current経路で生成し、別型復帰との同居を検収する。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType,SimpleNamespace
from typing import Any
import sys

ROOT = Path(__file__).resolve().parent
CONTINUATION = ROOT.parent/'g2_hidden_continuation_candidate_2026-09-10_v1'
sys.path.insert(0,str(CONTINUATION))
import run_continuation as B
import retained_inputs as I
import retained_wait as W

R = B.R


def guards() -> dict[str,str]:
    return B.guards()|{str(p):R.K.sha(p) for p in (*CONTINUATION.glob('*.py'),CONTINUATION/'PLAN.md')}


def install(stack: Any, factory: Any, rows: Any) -> None:
    B.install(stack,factory,rows)
    W.install(stack,B.B.B.B.patch)


def verify(state: Any, factory: Any, pipe: Any, rows: Any, trace: Any) -> Any:
    control,binding = factory.controller,factory.controller.history['1P']
    owner,outputs = binding.owner.state,control.hidden_current_outputs
    assert len(trace)==len(I.FRAMES) and len(rows)==3
    first = next(r for r in trace if r['owner'] is not None and r['owner']['current'] is not None)
    assert I.PY_FRAME<=first['frame']<I.BB_FRAME and first['owner']['current']['available'] is True
    old = first['owner']['current']
    assert owner.current is not None and not owner.current.available
    assert owner.current.evidence_id==old['evidence_id'] and owner.current.action==old['action']==2
    assert owner.current.grid==tuple(map(tuple,I.P0.saved()[0])) and owner.current.grid!=binding.current
    assert owner.action==4 and len(owner.history)==4 and sum(owner.counter)==67
    assert not pipe._pending_tsumo_1p and binding.next_token is None
    assert not pipe._tsumo_count_1p and not pipe._tsumo_count_2p
    assert not owner.origins and not owner.debts
    assert pipe._sm_1p.context.state.value=='stable' and B.B.L.L.compatible(owner,binding)
    assert binding.hidden_current is outputs[-1][0] and outputs[-1][0].frame==I.FRAMES[-1]
    assert binding.hidden_current.action==4 and outputs[-1][1].probability.cell(0,0).probs=={4:1.0}
    receiver,consumer = state['postcommit_current_receiver'],state['postcommit_publication_consumer']
    assert receiver.issued==1 and receiver.released==0 and not receiver.errors
    assert len(consumer.rows)==len(trace) and not consumer.errors
    assert all(not r['changed_sides'] and r['full_before']==r['full_after'] for r in consumer.rows)
    assert factory.provider.journal.steps==len(trace)*2 and not factory.provider.journal.errors
    assert not control.sticky_error
    return dict(updates=len(trace),original_integer_current_frame=first['frame'],
        original_integer_evidence_retained=True,current_colors=61,counter=67,
        conditional_frames=[v.frame for v,_ in outputs],private_placements=4,
        issued=receiver.issued,released=receiver.released,physical_certified=False,quality_gate_clear=False)


def drive(*args: Any) -> Any:
    state,factory = args[2],args[6]
    control,journal = factory.controller,factory.provider.journal
    refs = journal.complete_step,B.B.B.C.C.compatible,B.B.B.C.publish,W.P.single_head_wait
    try:
        return FunctionType(R.drive.__code__,dict(vars(R),I=I,install=install,verify=verify))(*args)
    finally:
        restored = refs==(journal.complete_step,B.B.B.C.C.compatible,B.B.B.C.publish,W.P.single_head_wait)
        R.K.write(state['output']/'RETAINED_CURRENT.json',dict(
            events=getattr(control,'hidden_current_events',[]),
            lifetimes=getattr(control,'hidden_lifetime_rows',[]),references_restored=restored))
        assert restored


def main() -> int:
    helper = SimpleNamespace(**(vars(R.K)|dict(guards=guards,load=B.B.B.palette_fixture.loader(R.K.load))))
    return FunctionType(R.main.__code__,dict(vars(R),ROOT=ROOT,K=helper,I=I,drive=drive))()


if __name__=='__main__':
    raise SystemExit(main())
