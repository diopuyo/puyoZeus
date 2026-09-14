"""原連続CPU：初回候補→raw不一致保留→原次NEXT→NON-STABLE/候補失効。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType,SimpleNamespace
from typing import Any
import sys

ROOT = Path(__file__).resolve().parent
CURRENT = ROOT.parent/'g2_hidden_current_candidate_2026-09-10_v1'
sys.path.insert(0,str(CURRENT))
import run_current as B
import lifetime_inputs as I
import lifetime_connection as L

R = B.B.R


def guards() -> dict[str,str]:
    return B.guards()|{str(p):R.K.sha(p) for p in (*CURRENT.glob('*.py'),CURRENT/'PLAN.md')}


def install(stack: Any, factory: Any, rows: Any) -> None:
    B.install(stack,factory,rows)
    factory.controller.hidden_lifetime_rows = []
    L.install(stack,factory,B.B.patch,factory.controller.hidden_lifetime_rows)


def verify(state: Any, factory: Any, pipe: Any, rows: Any, trace: Any) -> Any:
    control,binding = factory.controller,factory.controller.history['1P']
    outputs,events = control.hidden_current_outputs,control.hidden_current_events
    assert len(trace)==len(I.FRAMES) and len(rows)==2
    assert [v.frame for v,_ in outputs]==[34856,34858,34860,34862]
    assert binding.hidden_current is None and L.L.compatible(binding.owner.state,binding)
    assert sum(binding.owner.state.counter)==65 and len(binding.owner.state.history)==2
    assert binding.owner.state.current is None and binding.owner.state.action==3
    assert list(pipe._pending_tsumo_1p)==[I.OLD.BP] and binding.next_token is not None
    assert pipe._sm_1p.context.state.value=='tsumo_fall'
    assert binding.current==binding.hidden_anchor.certificate.sm_grid
    assert not pipe._tsumo_count_1p and not pipe._tsumo_count_2p
    assert not binding.owner.state.origins and not binding.owner.state.debts
    assert any(e['kind']=='conditional_candidate_hold' and e['frame']==34864 for e in events)
    assert control.hidden_lifetime_rows[0]['frame']==34864
    receiver,consumer = state['postcommit_current_receiver'],state['postcommit_publication_consumer']
    assert receiver.issued==receiver.released==0 and not receiver.errors
    assert len(consumer.rows)==len(trace) and not consumer.errors
    assert all(not r['changed_sides'] and r['full_before']==r['full_after'] for r in consumer.rows)
    assert factory.provider.journal.steps==len(trace)*2 and not factory.provider.journal.errors
    assert not control.sticky_error
    return dict(updates=len(trace),conditional_frames=[v.frame for v,_ in outputs],
        hold_frame=34864,next_frame=I.NEXT,original_J_next_enqueued=True,
        candidate_retired=True,integer_current_unchanged=True,original_PB_unchanged=True,
        original_counter_unchanged=True,next_hand_held=True,next_hand_consumed=False,
        physical_certified=False,quality_gate_clear=False)


def drive(*args: Any) -> Any:
    state,factory = args[2],args[6]
    assert args[3][0]._enable_next_history_starvation_fix is True
    control,journal = factory.controller,factory.provider.journal
    refs = journal.complete_step,B.C.C.compatible,B.C.publish
    try:
        return FunctionType(R.drive.__code__,dict(vars(R),I=I,install=install,verify=verify))(*args)
    finally:
        restored = refs==(journal.complete_step,B.C.C.compatible,B.C.publish)
        R.K.write(state['output']/'CONDITIONAL_LIFETIME.json',dict(
            events=getattr(control,'hidden_current_events',[]),
            lifetimes=getattr(control,'hidden_lifetime_rows',[]),references_restored=restored))
        assert restored


def main() -> int:
    helper = SimpleNamespace(**(vars(R.K)|dict(guards=guards,load=B.palette_fixture.loader(R.K.load))))
    return FunctionType(R.main.__code__,dict(vars(R),ROOT=ROOT,K=helper,I=I,drive=drive))()


if __name__=='__main__':
    raise SystemExit(main())
