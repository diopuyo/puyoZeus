"""原連続CPUで第三単headの配置と二度目の自然currentを検収する。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType,SimpleNamespace
from typing import Any
import sys

ROOT = Path(__file__).resolve().parent
LIFETIME = ROOT.parent/'g2_hidden_current_lifetime_candidate_2026-09-10_v1'
sys.path.insert(0,str(LIFETIME))
import run_lifetime as B
import continuation_inputs as I
import continuation_connection as C

R = B.R


def guards() -> dict[str,str]:
    return B.guards()|{str(p):R.K.sha(p) for p in (*LIFETIME.glob('*.py'),LIFETIME/'PLAN.md',I.SAVED)}


def install(stack: Any, factory: Any, rows: Any) -> None:
    B.install(stack,factory,rows)
    C.install(stack,factory,B.B.B.patch,rows)


def verify(state: Any, factory: Any, pipe: Any, rows: Any, trace: Any) -> Any:
    control,binding = factory.controller,factory.controller.history['1P']
    outputs,events = control.hidden_current_outputs,control.hidden_current_events
    assert len(trace)==len(I.FRAMES) and len(rows)==3
    assert [r['frame'] for r in rows[:2]]==[34850,34854] and rows[-1]['frame']>=I.LAND+I.STRIDE
    assert binding.owner.state.action==3 and len(binding.owner.state.history)==3
    assert sum(binding.owner.state.counter)==67 and binding.owner.state.current is None
    assert not pipe._pending_tsumo_1p and binding.next_token is None
    assert not pipe._tsumo_count_1p and not pipe._tsumo_count_2p
    assert not binding.owner.state.origins and not binding.owner.state.debts
    assert pipe._sm_1p.context.state.value=='stable'
    assert B.L.L.compatible(binding.owner.state,binding)
    assert binding.hidden_current is outputs[-1][0] and outputs[-1][0].frame==I.FRAMES[-1]
    assert binding.hidden_current.action==3 and binding.hidden_current.inferred_grid==I.generated()[1]
    assert outputs[-1][1].probability.cell(0,0).probs=={4:1.0}
    assert len([e for e in events if e['kind']=='natural_hidden_exit_preview' and e['visible_matches']])==2
    receiver,consumer = state['postcommit_current_receiver'],state['postcommit_publication_consumer']
    assert receiver.issued==receiver.released==0 and not receiver.errors
    assert len(consumer.rows)==len(trace) and not consumer.errors
    assert all(not r['changed_sides'] and r['full_before']==r['full_after'] for r in consumer.rows)
    assert factory.provider.journal.steps==len(trace)*2 and not factory.provider.journal.errors
    assert not control.sticky_error
    return dict(updates=len(trace),pop_frames=[r['frame'] for r in rows],counter=67,
        conditional_frames=[v.frame for v,_ in outputs],next_hand_consumed=True,second_natural_current=True,
        integer_current_unchanged=True,original_PB_unchanged=True,original_counter_unchanged=True,
        issued=0,released=0,physical_certified=False,quality_gate_clear=False)


def drive(*args: Any) -> Any:
    state,factory = args[2],args[6]
    R.K.write(state['output']/'ARTIFICIAL_NEXT_INPUT.json',dict(
        generated_before=I.generated()[0],generated_final=I.generated()[1],generated_raw=I.generated()[2],
        next_frame=I.NEXT,raw_from=I.LAND,artificial=True,original_video_checkpoint=False))
    control,journal = factory.controller,factory.provider.journal
    refs = journal.complete_step,B.B.C.C.compatible,B.B.C.publish,C.E.N.eligible,C.E.C.make
    try:
        return FunctionType(R.drive.__code__,dict(vars(R),I=I,install=install,verify=verify))(*args)
    finally:
        restored = refs==(journal.complete_step,B.B.C.C.compatible,B.B.C.publish,C.E.N.eligible,C.E.C.make)
        R.K.write(state['output']/'CONTINUATION_CURRENT.json',dict(
            events=getattr(control,'hidden_current_events',[]),
            lifetimes=getattr(control,'hidden_lifetime_rows',[]),references_restored=restored))
        assert restored


def main() -> int:
    helper = SimpleNamespace(**(vars(R.K)|dict(guards=guards,load=B.B.palette_fixture.loader(R.K.load))))
    return FunctionType(R.main.__code__,dict(vars(R),ROOT=ROOT,K=helper,I=I,drive=drive))()


if __name__=='__main__':
    raise SystemExit(main())
