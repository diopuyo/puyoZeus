"""先頭の原47更新を再用し、後続一枠の原popを追加検収する。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
PREFIX = ROOT.parent/'g2_hidden_two_hand_candidate_2026-09-10_v1'
sys.path.insert(0,str(PREFIX))
import run_prefix as R
import tail_connection as C


def guards() -> dict[str,str]:
    return R.K.guards() | {str(p):R.K.sha(p) for p in (*PREFIX.glob('*.py'),PREFIX/'PLAN.md')}


def verify(state: Any, factory: Any, pipe: Any, rows: Any, trace: Any) -> dict[str,Any]:
    binding = factory.controller.history['1P']
    private,base = binding.owner.state,tuple(map(tuple,R.I.saved()[0]))
    assert len(trace)==len(R.I.FRAMES) and len(rows)==2,'tail_count'
    assert rows[0]['kind']==R.C.H.KIND and rows[1]['kind']==C.H.KIND
    assert rows[1]['source']['occurred'][0]>rows[0]['frame'],'tail_fresh_votes'
    assert sum(private.counter)==65 and len(private.history)==2 and private.action==2
    assert not pipe._pending_tsumo_1p and binding.next_token is None and binding.next_started is None
    assert private.current is None and binding.current==base
    assert factory.controller._parts.T.board_key(pipe._sm_1p.context.confirmed_board)==base
    assert not pipe._tsumo_count_1p and not pipe._tsumo_count_2p
    assert not private.debts and not private.origins
    receiver,consumer = state['postcommit_current_receiver'],state['postcommit_publication_consumer']
    assert receiver.issued==receiver.released==0 and not receiver.errors
    assert len(consumer.rows)==len(R.I.FRAMES) and not consumer.errors
    assert all(not r['changed_sides'] and r['full_before']==r['full_after'] for r in consumer.rows)
    journal = factory.provider.journal
    assert journal.steps==len(R.I.FRAMES)*2 and not journal.errors
    assert not factory.controller.sticky_error and not factory.provider.handoff_proofs
    return dict(completed_updates=len(trace),prefix_frame=rows[0]['frame'],tail_frame=rows[1]['frame'],
        original_J_pop_count=2,counter=65,current_retained=61,remaining_FIFO=0,origin_count=0,
        issued=0,released=0,new_current_permission=False,next_action_created=False,
        physical_certified=False,quality_gate_clear=False)


def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
    present,original = name in vars(obj),vars(obj).get(name)
    stack.callback(lambda:setattr(obj,name,original) if present else delattr(obj,name))
    setattr(obj,name,value)


def install(stack: Any, factory: Any, rows: Any) -> None:
    R.install(stack,factory,rows)
    C.install(stack,factory,patch,rows)


def drive(*args: Any) -> Any:
    return FunctionType(R.drive.__code__,dict(vars(R),install=install,verify=verify))(*args)


def main() -> int:
    helper = SimpleNamespace(**(vars(R.K)|dict(guards=guards)))
    return FunctionType(R.main.__code__,dict(vars(R),ROOT=ROOT,K=helper,drive=drive))()


if __name__=='__main__':
    raise SystemExit(main())
