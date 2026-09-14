"""元47更新に自然STABLEと条件付きPBの別型生成を追加する。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
TAIL = ROOT.parent/'g2_hidden_tail_candidate_2026-09-10_v1'
sys.path.insert(0,str(TAIL))
import run_tail as B
import hidden_current_connection as C
import palette_fixture

PROVISIONAL = ROOT.parent/'g2_hidden_probability_provisional_2026-09-08_v1/provisional.py'


def guards() -> dict[str,str]:
    return B.guards() | {str(p):B.R.K.sha(p) for p in (*TAIL.glob('*.py'),TAIL/'PLAN.md',PROVISIONAL)}


def install(stack: Any, factory: Any, rows: Any) -> None:
    B.install(stack,factory,rows)
    provisional = B.R.K.load('_hidden_current_existing_provisional',PROVISIONAL,stack)
    control = factory.controller
    control.hidden_current_events,control.hidden_current_outputs = [],[]
    C.install(stack,factory,B.patch,provisional,control.hidden_current_events,control.hidden_current_outputs)


def verify(state: Any, factory: Any, pipe: Any, rows: Any, trace: Any) -> dict[str,Any]:
    control,binding = factory.controller,factory.controller.history['1P']
    outputs,events = control.hidden_current_outputs,control.hidden_current_events
    assert len(trace)==len(B.R.I.FRAMES) and len(rows)==2 and outputs,'conditional_missing_exit'
    assert sum(binding.owner.state.counter)==65 and len(binding.owner.state.history)==2
    assert not pipe._pending_tsumo_1p and binding.owner.state.current is None
    assert not pipe._tsumo_count_1p and not pipe._tsumo_count_2p
    assert not binding.owner.state.origins and not binding.owner.state.debts
    last,side = outputs[-1]
    assert binding.hidden_current is last and last.frame==B.R.I.FRAMES[-1]
    assert pipe._sm_1p.context.state.value=='stable' and C.C.compatible(binding.owner.state,binding)
    assert C.C.intact(last) and last.sm_grid[0][0]==0 and side.probability.cell(0,0).probs=={4:1.0}
    assert not last.integer_current_permission and not last.accounting_permission
    receiver,consumer = state['postcommit_current_receiver'],state['postcommit_publication_consumer']
    assert receiver.issued==receiver.released==0 and not receiver.errors
    assert len(consumer.rows)==len(trace) and not consumer.errors
    assert all(not r['changed_sides'] and r['full_before']==r['full_after'] for r in consumer.rows)
    assert factory.provider.journal.steps==len(trace)*2 and not factory.provider.journal.errors
    assert not control.sticky_error and not factory.provider.handoff_proofs
    return dict(completed_updates=len(trace),prefix_frame=rows[0]['frame'],tail_frame=rows[1]['frame'],
        conditional_frames=[v.frame for v,_ in outputs],natural_stable=True,visible_colors=64,
        conditional_hidden='Y at row0col0 under unique-world conditions',original_hidden_retained_empty=True,
        integer_current_unchanged=True,original_PB_unchanged=True,current_compatibility=True,
        issued=0,released=0,physical_certified=False,quality_gate_clear=False,next_hand_tested=False)


def drive(*args: Any) -> Any:
    state,factory = args[2],args[6]
    control,journal = factory.controller,factory.provider.journal
    assert args[3][0]._enable_next_history_starvation_fix is True,'fixture_not_live_palette_flag'
    cls = type(control)
    lifecycle = cls.prepared.__globals__['V1'].L
    module = sys.modules[cls.prepared.__globals__['__name__']]
    previous = cls.hold_transition,journal.complete_step,lifecycle.compatible_current,module.should_fall
    try:
        return FunctionType(B.R.drive.__code__,dict(vars(B.R),install=install,verify=verify))(*args)
    finally:
        restored = previous==(cls.hold_transition,journal.complete_step,lifecycle.compatible_current,module.should_fall)
        B.R.K.write(state['output']/'CONDITIONAL_CURRENT.json',dict(
            events=getattr(control,'hidden_current_events',[]),references_restored=restored))
        assert restored


def main() -> int:
    helper = SimpleNamespace(**(vars(B.R.K)|dict(guards=guards,load=palette_fixture.loader(B.R.K.load))))
    return FunctionType(B.R.main.__code__,dict(vars(B.R),ROOT=ROOT,K=helper,drive=drive))()


if __name__=='__main__':
    raise SystemExit(main())
