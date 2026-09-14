"""同原Jの次一手に限定し、閉鎖済み条件付きoriginを消さず継続する。"""
from __future__ import annotations
from dataclasses import asdict,replace
from types import FunctionType,SimpleNamespace as N
from typing import Any
import json
import sys
import bundle as B
import next_state as S
import continuation_history as H


def prepare(control: Any,binding: Any,item: Any,sm: Any,signals: Any,view: Any) -> Any:
    from src import puyo_core_bridge as core
    value=S.current(control,binding)
    S.P.require(value.placement is None,'postnext_duplicate_prepare')
    lifecycle=type(control).prepared.__globals__['V1'].L
    actual=lifecycle.prepare; previous=actual.__globals__['closed_origins']
    def closed(state: Any,owner: Any) -> bool: return S.closed_origins(state,owner,control,previous)
    generated=FunctionType(actual.__code__,dict(actual.__globals__,closed_origins=closed),
        actual.__name__,actual.__defaults__,actual.__closure__)
    proxy=N(**(vars(lifecycle)|dict(prepare=generated)))
    make=FunctionType(H.prepare.__code__,dict(vars(H),KIND=S.KIND))
    return make(proxy,core,control,binding,item,sm,signals,view)


def consume(control: Any,call: Any,caller: Any,rows: list[Any]) -> None:
    binding,prepared=call['binding'],call['prepared']; values=caller.f_locals
    prior=S.current(control,binding); votes=prepared['continuation_votes']
    S.P.require(prior.state is prepared['old_state'] is votes.state,'postnext_commit_state')
    S.P.require(values['committed'] is votes.head is prior.head and not prior.queue
        and votes.token==prior.token,'postnext_actual_pop')
    control.unchanged(call,values['self'],values['side'])
    control.provider._parts.V.Provider.after_history_consume(control.provider,call,caller)
    binding.policy.arm('placement',prepared['evidence'],prior.state,prepared['proof'])
    binding.owner.add_placement(prepared['evidence'],prepared['evidence'].available_at)
    binding.grid=votes.final; binding.consumed_tokens.add(votes.token)
    binding.next_token=binding.next_started=binding.candidate=None
    binding.clear_grid=binding.clear_first=binding.clear_last=None; binding.clear_count=0
    binding.hidden_continuation_votes=None; call['consumed']=True
    S.consumed(control,binding,prepared['proof'],votes.final)
    control.unchanged(call,values['self'],values['side'])
    rows.append(dict(stage='conditional_next_placement_committed',frame=call['view'].frame,
        proof=prepared['proof'],state=asdict(binding.owner.state),original_pop=True,
        physical_certified=False,current_permission=False))


def formula(stack: Any,factory: Any,pipe: Any,patch: Any,rows: list[Any]) -> None:
    import firing_handoff_fixed as F
    def deferred(self: Any,side: str,clock: float,previous: Any,binding: Any,frame: Any) -> Any:
        S.P.require(sys._getframe(1) is frame,'postnext_formula_frame')
        actual=F.V5.V4.bound_update(self); caller=frame.f_back
        S.P.require(caller.f_code is actual.__code__ and caller.f_globals is actual.__globals__
            and caller.f_locals['self'] is self is pipe and caller.f_locals['time_sec']==clock,'postnext_formula_update')
        S.P.require(actual.__globals__['__next_live'] is factory.provider.journal.controller,'postnext_formula_N')
        S.registered(factory.controller,binding)
        queue=self._pending_tsumo_1p
        S.P.require(side=='1P' and self._active_chain_1p is None and len(queue)<=1,'postnext_formula_pending')
        value=S.get(factory.controller,binding)
        if value is not None and value.placement is None:
            S.P.require(queue is value.queue and len(queue)==1 and queue[0] is value.head,'postnext_formula_head')
        rows.append(dict(stage='conditional_next_formula_deferred',frame=caller.f_locals['frame_idx'],
            original_event=False,physical_certified=False,pending_count=len(queue)))
    B.C.FC.install(stack,pipe,patch,deferred)


def install_calls(stack: Any,factory: Any,patch: Any,rows: list[Any]) -> None:
    control=factory.controller; cls=type(control); v1=cls.prepared.__globals__['V1']
    old_hand,old_prepare,old_call,old_consume=cls.hand,v1.prepared,cls.call,cls.consumed_history
    def hand(self: Any,binding: Any,view: Any) -> Any:
        value=S.get(self,binding)
        if value is None and getattr(binding,'conditional_firing_registered',None) is not None and view.refs:
            S.P.require(sys._getframe(1).f_code is self._parts.C.Controller.update.__code__,'postnext_hand_caller')
            value=S.start(self,binding,view,rows)
        if value is None: return old_hand(self,binding,view)
        S.current(self,binding)
        S.P.require(len(view.refs)<=1 and len(view.tokens)==len(view.refs),'postnext_single_head')
        if view.refs and view.quiet is not True: return None
        return self._parts.C.Controller.hand(self,binding,view)
    def prepared(self: Any,binding: Any,item: Any,sm: Any,raw: Any,signals: Any,view: Any) -> Any:
        if S.get(self,binding) is None: return old_prepare(self,binding,item,sm,raw,signals,view)
        return prepare(self,binding,item,sm,signals,view)
    def call(self: Any,caller: Any,binding: Any,view: Any,pipe: Any,side: str,proposal: Any) -> Any:
        if proposal is not None and proposal.get('kind')==S.KIND:
            return self._parts.C.Controller.call(self,caller,binding,view,pipe,side,proposal)
        return old_call(self,caller,binding,view,pipe,side,proposal)
    def consumed(self: Any,value: Any,caller: Any) -> Any:
        if value['prepared'].get('kind')==S.KIND: return consume(self,value,caller,rows)
        return old_consume(self,value,caller)
    for obj,name,value in ((cls,'hand',hand),(v1,'prepared',prepared),(cls,'call',call),(cls,'consumed_history',consumed)):
        patch(stack,obj,name,value)


def install(stack: Any,factory: Any,pipe: Any,patch: Any,rows: list[Any]) -> None:
    FunctionType(B.C.install.__code__,dict(vars(B.C),delayed_formula=formula))(stack,factory,pipe,patch,rows)
    control=factory.controller; control.conditional_next_hands={}
    patch(stack,B.C.L,'registered',S.registered)
    original=B.C.L.certificate
    def certificate(*args: Any) -> Any:
        made=original(*args)
        if made is None: return None
        binding=args[1]['binding']; value=S.get(args[0],binding)
        if value is None: return made
        S.current(args[0],binding)
        S.P.require(value.placement is not None,'postnext_current_before_placement')
        current,side=made; proof=json.loads(current.evidence_json)
        proof.update(kind='conditional_hidden_after_firing_next_current/v1',next_placement=value.placement)
        return replace(current,evidence_json=S.L.C.encoded(proof)),side
    patch(stack,B.C.L,'certificate',certificate)
    install_calls(stack,factory,patch,rows)
