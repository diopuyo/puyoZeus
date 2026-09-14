"""実call配置後だけ元Sへ条件世界originを登録する。原RPイベントは作らない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict,replace
from typing import Any
import origin_evidence as E
import origin_fixed as F


def checked(control: Any, binding: Any, call: Any) -> Any:
    import continuation_witness as T
    saved=binding.conditional_firing_committed
    view,prepared,frame=call['view'],call['prepared'],call['frame']
    values=frame.f_locals; votes=saved['votes']; p=control.inventory
    F.require(call['binding'] is binding and call['consumed'] is True,'uncommitted_call')
    F.require(control.calls.get(id(frame)) is call and frame.f_code in control.provider.journal.codes,'actual_J_call')
    F.require(saved['evidence'] is prepared['evidence'] and votes is prepared['votes'],'prepared_identity')
    F.require(votes.state is prepared['old_state'] and values['committed'] is votes.head,'native_pop')
    F.require(view.queue is votes.queue and len(view.queue)==0 and binding.scope==view.scope==votes.scope,'queue_scope')
    F.require(view.frame==prepared['evidence'].available_at.frame and view.clock==view.frame/E.FPS,'same_call')
    F.require(votes.token in binding.consumed_tokens and binding.next_token is None,'consumed_token')
    F.require(binding.candidate is None and binding.next_started is None,'no_next_action')
    F.require(binding.owner.state.current is votes.state.current,'old_current_slot')
    F.require(T.L.C.intact(saved['previous_anchor'].certificate),'original_anchor_intact')
    E.equal(p,T.V.validate(control,binding,binding.hidden_prefix_votes),saved['proof']['prefix_source'],'live_prefix_source')
    E.equal(p,saved['proof'],prepared['proof'],'prepared_contents')
    F.require(p.digest(saved['proof'])==saved['evidence'].event_id,'placement_digest')
    F.require(control._parts.T.board_key(values['sm'].context.confirmed_board)==binding.current,'original_current')
    control.unchanged(call,values['self'],values['side'])
    return saved


def build(p: Any, state: Any, source: Any, previous_anchor: Any, now: Any) -> tuple[Any,Any]:
    grid=tuple(tuple(r) for r in source['grid'])
    final=tuple(tuple(r) for r in source['predicted_final'])
    identifier=E.identity(p,state,source,final)
    erased=tuple(a-b for a,b in zip(p.S.color_counts(grid),p.S.color_counts(final)))
    value=p.S.OriginEvidence(state.scope,identifier,state.action,now,now,grid,final,erased,'',E.PREFIX+identifier)
    value=replace(value,prediction_sha256=p.S.prediction_digest(value))
    proof=dict(kind=E.KIND,source_token=source['token'],placement_event_id=p.digest(source),
        evidence=asdict(value),placement=deepcopy(source),previous_anchor=deepcopy(previous_anchor),
        backend=F.backend(),conditional_world=True,physical_certified=False,original_event=False)
    return value,proof


def register(control: Any, binding: Any, call: Any) -> tuple[Any,Any]:
    saved=checked(control,binding,call)
    p,state,view=control.inventory,binding.owner.state,call['view']
    now=control._parts.C.H.clock(p,view.frame,view.clock,E.OP_ORIGIN)
    value,proof=build(p,state,saved['proof'],asdict(saved['previous_anchor']),now)
    binding.policy.arm('origin',value,state,proof)
    result=binding.owner.register_debt(value,now)
    F.require(result.current is state.current and result.counter==state.counter and result.history==state.history,
        'origin_changed_non_debt')
    binding.conditional_firing_origin=dict(evidence=value,proof=deepcopy(proof),registered=True,
        original_event=False,physical_certified=False,settled=False)
    control.unchanged(call,call['frame'].f_locals['self'],call['frame'].f_locals['side'])
    return value,proof
