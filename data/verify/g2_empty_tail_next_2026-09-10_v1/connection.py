"""尾手で空になった原FIFOに後発した一手を、旧経路と別属性へ接続する。"""
from __future__ import annotations
from dataclasses import asdict
import json
from typing import Any

TAIL_KIND = 'hidden_single_tail_history/v1'
HISTORY_KIND = 'hidden_empty_tail_next_history/v1'
STRIDE, FPS, REQUIRED_VOTES, QUEUE_SIZE, PLACEMENT_OPERATION = 2,60,2,1,1


def captured_receipt(record: Any) -> Any:
    return dict(kind='empty_tail_captured/v1',scope=record.scope,frame=record.frame,clock=record.clock,
        binding_id=id(record.binding),state=asdict(record.state),grid=record.grid,old_current=record.old_current,
        proof=record.proof,digest=record.digest,policy_content=record.policy_content,
        source_content=record.source_content,votes_content=record.votes_content,
        journal_content=record.journal_content,head=None,token=None,current_permission=False,
        physical_certified=False)


def bound_receipt(record: Any) -> Any:
    return dict(kind='empty_tail_new_head_started/v1',scope=record.scope,frame=record.started[0],
        clock=record.started[1],binding_id=id(record.binding),old_tail_digest=record.digest,
        token=record.token,head=tuple(record.head),state=asdict(record.started_state),
        bound_content=record.bound_content,current_permission=False,physical_certified=False)


def recording(stack: Any, control: Any) -> None:
    def save() -> None:
        value=dict(basis_events=control.empty_tail_basis_events,prepop_checks=control.empty_tail_prepop_checks,
            output_identity_is_not_live_authorization=True,physical_certified=False,quality_gate_clear=False)
        path=control.provider.journal.output/'EMPTY_TAIL_LIVE_EVIDENCE.json'
        with path.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    stack.callback(save)


def proof_binding(basis: Any, control: Any, call: Any, record: Any) -> None:
    """元producerが付けた固定欄を、消費前に元基点と型付き全文で結ぶ。"""
    binding,view,prepared=call['binding'],call['view'],call['prepared']
    proof,votes=prepared['proof'],prepared['continuation_votes']
    assert type(proof) is dict and prepared['kind']==HISTORY_KIND
    expected=dict(kind=HISTORY_KIND,pair=record.head,token=record.token,
        scope=asdict(binding.owner.state.scope),available_frame=view.frame,available_time=view.clock,
        prefix_source=basis.V.validate(control,binding,record.source),inferred_final=votes.final,
        available_window=False,new_token=None,inferred_grid_is_observed=False,
        physical_certified=False,current_permission=False)
    actual={key:proof[key] for key in expected}
    assert basis.SP.content(actual)==basis.SP.content(expected)


def prepop(basis: Any, control: Any, call: Any) -> Any:
    """実native pop前に、提案と元票/基点/元S evidenceの全結合を再確認する。"""
    binding,view,prepared = call['binding'],call['view'],call['prepared']
    record = basis.validate(control,binding,prepared['private_basis'])
    proof_binding(basis,control,call,record)
    votes,state,p = prepared['continuation_votes'],binding.owner.state,control.inventory
    proof,evidence = prepared['proof'],prepared['evidence']
    assert votes.state is prepared['old_state'] is state and votes.binding is binding
    assert view.queue is record.queue is votes.queue and len(view.queue)==QUEUE_SIZE
    assert view.queue[0] is votes.head is record.head and view.tokens==(record.token,)
    assert votes.token==record.token==binding.next_token and record.token not in binding.consumed_tokens
    assert votes.item is binding.candidate and votes.scope==view.scope==binding.scope
    assert type(evidence) is p.S.PlacementEvidence and evidence.event_id==p.digest(proof)
    assert evidence.scope==state.scope and evidence.action==state.action
    assert proof['before_grid']==record.grid and proof['grid']==prepared['grid']==votes.final
    assert proof['observed_raw']==votes.raw and proof['raw_capture']==votes.raw_proof
    assert proof['occurred']==votes.first and proof['clear_last']==votes.last
    assert proof['clear_observations']==votes.count and type(votes.count) is int and votes.count==REQUIRED_VOTES
    assert votes.last==(view.frame,view.clock) and votes.first==(view.frame-STRIDE,(view.frame-STRIDE)/FPS)
    assert proof['previous_private_placement']==record.digest and proof['previous_private_frame']==record.frame
    assert evidence.available_at.frame==view.frame and evidence.available_at.time_sec==view.clock
    assert evidence.occurred_at==control._parts.C.H.clock(p,*votes.first)
    assert evidence.available_at==control._parts.C.H.clock(p,view.frame,view.clock,PLACEMENT_OPERATION)
    p.S.validate_counts(evidence.added)
    assert evidence.added==tuple(a-b for a,b in zip(p.S.color_counts(votes.final),state.counter))
    assert evidence.added==tuple(votes.head.count(c) for c in range(1,p.S.COLORS+1))
    p.S.available(state.action_since,evidence.occurred_at)
    p.S.available(evidence.occurred_at,evidence.available_at)
    assert proof['current_permission'] is False and proof['physical_certified'] is False
    return dict(frame=view.frame,token=record.token,event_id=evidence.event_id,
        queue_slots=QUEUE_SIZE,state_action=state.action,current_permission=False)


def install(stack: Any, factory: Any, patch: Any, basis: Any, placement: Any,
            exit_module: Any, rows: list[Any]) -> None:
    from src import puyo_core_bridge as core
    control,cls = factory.controller,type(factory.controller)
    control.empty_tail_prepop_checks,control.empty_tail_basis_events = [],[]
    recording(stack,control)
    v1 = cls.prepared.__globals__['V1']
    old_hand,old_prepare,old_call,old_consume = cls.hand,v1.prepared,cls.call,cls.consumed_history
    def hand(self: Any, binding: Any, view: Any) -> Any:
        item = old_hand(self,binding,view)
        record = getattr(binding,'empty_tail_basis',None)
        if (record is not None and getattr(binding,'empty_tail_placement',None) is None
            and getattr(binding,'hidden_anchor',None) is None and item is not None):
            first=record.started is None
            basis.bind_head(self,binding,view,item)
            if first: self.empty_tail_basis_events.append(bound_receipt(record))
        return item
    def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
        if (getattr(binding,'empty_tail_basis',None) is not None
            and getattr(binding,'empty_tail_placement',None) is None
            and getattr(binding,'hidden_anchor',None) is None):
            return placement.prepare(basis,v1.L,core,self,binding,item,sm,signals,view)
        return old_prepare(self,binding,item,sm,raw,signals,view)
    def call(self: Any, caller: Any, binding: Any, view: Any, pipe: Any, side: str, proposal: Any) -> Any:
        if proposal is not None and proposal.get('kind') == placement.KIND:
            value = self._parts.C.Controller.call(self,caller,binding,view,pipe,side,proposal)
            self.empty_tail_prepop_checks.append(prepop(basis,self,value))
            return value
        return old_call(self,caller,binding,view,pipe,side,proposal)
    def consumed(self: Any, value: Any, caller: Any) -> None:
        if value['prepared'].get('kind') == placement.KIND:
            return placement.consumed(basis,self,value,caller,rows)
        old_consume(self,value,caller)
        if (value['prepared'].get('kind') == TAIL_KIND and not value['view'].queue
            and getattr(value['binding'],'private_suffix_basis',None) is None):
            record=basis.capture(self,value['binding'],value,caller)
            self.empty_tail_basis_events.append(captured_receipt(record))
    for obj,name,value in ((cls,'hand',hand),(v1,'prepared',prepared),(cls,'call',call),(cls,'consumed_history',consumed)):
        patch(stack,obj,name,value)
    exit_module.install(stack,patch,basis,placement)
