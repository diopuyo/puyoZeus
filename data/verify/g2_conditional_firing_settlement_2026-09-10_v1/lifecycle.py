"""条件付き起点の未精算HOLDと精算済み原自然STABLEを明確に分ける。"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from types import FunctionType,SimpleNamespace
from typing import Any
import json
import inspect


def registered(control: Any, binding: Any) -> Any:
    import path_support as P
    p,state=control.inventory,binding.owner.state
    saved=binding.conditional_firing_registered
    origin=saved['origin']; proof=saved['proof']
    P.require(saved['scope']==binding.scope and origin.scope==state.scope and origin.action==state.action,
        'conditional_lifecycle_scope')
    P.require(state.origins==(origin,) and origin.event_identity.startswith('conditional_world:'),
        'conditional_lifecycle_origin')
    policy=getattr(binding.policy,'accounting',binding.policy)
    records=[r for r in policy.proofs if r['purpose']=='origin' and r['proof']['evidence']['origin_id']==origin.origin_id]
    P.require(len(records)==1 and records[0]['proof_sha']==p.digest(proof)
        and p.encoded(records[0]['proof'])==p.encoded(proof),'conditional_lifecycle_origin_log')
    P.require(proof['kind']=='conditional_firing_origin/v1' and proof['original_event'] is False
        and proof['physical_certified'] is False,'conditional_lifecycle_kind')
    anchor=proof['previous_anchor']['certificate']
    P.require(p.encoded(asdict(state.current))==anchor['integer_anchor'],'conditional_lifecycle_old_slot')
    P.require(saved['token'] in binding.consumed_tokens and binding.next_token is None,'conditional_lifecycle_token')
    if saved.get('settled',False):
        settlement=saved['settlement_proof']
        rows=[r for r in policy.proofs if r['purpose']=='settlement' and r['proof_sha']==p.digest(settlement)]
        P.require(len(rows)==1 and p.encoded(rows[0]['proof'])==p.encoded(settlement),'conditional_lifecycle_settlement_log')
        P.require(not state.debts and origin.origin_id in state.consumed_ids
            and state.history[-1].kind=='settlement' and state.history[-1].event_id==origin.origin_id,
            'conditional_lifecycle_closed')
        expected=origin.predicted_final
    else:
        P.require(len(state.debts)==1 and state.debts[0].origin==origin
            and origin.origin_id not in state.consumed_ids,'conditional_lifecycle_pending')
        expected=origin.before_grid
    P.require(binding.grid==expected and state.counter==p.S.color_counts(expected),'conditional_lifecycle_grid')
    return saved


def eligible(control: Any, binding: Any, signals: Any, call: Any) -> Any:
    import prefix_witness as W
    import path_support as P
    saved=registered(control,binding)
    view=call['view']
    if not saved.get('settled',False): return None
    if (call['prepared'] is not None or call['consumed'] or view.refs or view.tokens or view.queue
        or signals.is_match_active is not True or signals.chain_event is not None
        or signals.effect_gate_window_active is not False or view.quiet is not True): return None
    P.require(view.scope==binding.scope and view.frame>=saved['settlement_proof']['evidence']['available_at']['frame'],
        'conditional_lifecycle_exit_clock')
    captured=W.observed(control,binding,signals,view)
    return captured if captured is not None and P.compatible(captured[0],binding.grid) else None


def certificate(control: Any, call: Any, result: Any, provisional: Any, raw: Any, raw_proof: Any) -> Any:
    import conditional_current as C
    binding,view=call['binding'],call['view']; state=binding.owner.state
    saved=registered(control,binding)
    current=C.returned_board(control,call,result)
    if not C.observation_matches(control,call,current,raw): return None
    inferred,original=C.probability(provisional.ProbabilisticBoard,result.prob_board,current,binding.grid)
    key=provisional.FrameKey(view.scope[0],view.scope[1],view.frame,view.clock)
    side=provisional.SideInput(key,result.side,'STABLE',result.confirmed_board,inferred,key,result.side,
        'conditional_firing_world_after_observed_clear;not_physical_certification')
    provisional.validate_side(side,result.side)
    _,hidden,visible_sha=provisional.prepare(side)
    proof=dict(kind='conditional_hidden_after_firing_current/v1',frame=view.frame,clock=view.clock,scope=view.scope,
        original_PB=original,conditional_PB=C.cells(inferred),raw=raw,raw_capture=raw_proof,
        sm=current,inferred_final=binding.grid,integer_anchor=C.anchor(state),visible_sha256=visible_sha,
        action=state.action,counter=state.counter,conditional_origin=saved['proof'],
        conditional_settlement=saved['settlement_proof'],natural_transition=call.get('hidden_transition'),
        original_PB_unchanged=True,conditional_not_recognition_confidence=True,physical_certified=False)
    value=C.ConditionalCurrent(view.scope,view.frame,view.clock,state.action,current,binding.grid,
        hidden,C.anchor(state),C.encoded(proof))
    C.P.require(C.intact(value),'conditional_firing_current_intact')
    return value,side


def make(control: Any, call: Any, result: Any, provisional: Any) -> Any:
    import conditional_current as C
    binding,view=call['binding'],call['view']
    saved=registered(control,binding)
    values=call['frame'].f_locals
    if not saved.get('settled',False): return None
    if result is None or result.state.value!='stable' or values['ctx'].state.value!='stable': return None
    if values['signals'].is_match_active is not True or values['signals'].effect_gate_window_active is not False: return None
    if not C.fresh(control,call,result): return None
    captured=C.W.observed(control,binding,values['signals'],view)
    if captured is None: return None
    C.P.require(result.side==view.scope[-1] and values['ctx'].frame_idx==view.frame,'conditional_firing_current_clock')
    return certificate(control,call,result,provisional,*captured)


def install(stack: Any, factory: Any, patch: Any, rows: list[Any]) -> None:
    import hidden_bundle as B
    import natural_exit as N
    control=factory.controller; cls=type(control)
    old_hold,old_make=cls.hold_transition,B.CURRENT.C.C.make
    module=SimpleNamespace(**cls.prepared.__globals__)
    transition=FunctionType(N.transition.__code__,dict(vars(N),eligible=eligible))
    wrappers=FunctionType(N.wrappers.__code__,dict(vars(N),transition=transition))
    generator=inspect.unwrap(N.installed)
    installed=contextmanager(FunctionType(generator.__code__,dict(vars(N),wrappers=wrappers),
        generator.__name__,generator.__defaults__,generator.__closure__))
    def hold(self: Any, sm: Any, signals: Any, binding: Any) -> Any:
        if getattr(binding,'conditional_firing_registered',None) is None: return old_hold(self,sm,signals,binding)
        registered(self,binding)
        return installed(self,sm,signals,binding,module,rows)
    def current(self: Any, call: Any, result: Any, provisional: Any) -> Any:
        if getattr(call['binding'],'conditional_firing_registered',None) is None:
            return old_make(self,call,result,provisional)
        return make(self,call,result,provisional)
    patch(stack,cls,'hold_transition',hold)
    patch(stack,B.CURRENT.C.C,'make',current)
