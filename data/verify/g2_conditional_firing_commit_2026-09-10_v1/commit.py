"""元J単head消費後のprivate配置一回。予測消去はまだ記帳しない。"""
from __future__ import annotations
from dataclasses import asdict
from types import FunctionType
from typing import Any
import admission as A

KIND='conditional_firing_placement/v1'


def captured(control: Any, binding: Any, item: Any, signals: Any, view: Any, rows: Any) -> bool:
    original=FunctionType(A.proposal.__code__,dict(vars(A),KIND=KIND))
    def proposal(*args: Any) -> Any:
        value=original(*args)
        assert value['old_state'] is binding.owner.state
        binding.conditional_firing_proposal=value
        return value
    run=FunctionType(A.prepare.__code__,dict(vars(A),proposal=proposal))
    return run(control,binding,item,signals,view,rows)


def consumed(control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
    import continuation_witness as T
    binding,view,prepared=call['binding'],call['view'],call['prepared']
    votes,values=prepared['votes'],caller.f_locals
    state,current=binding.owner.state,binding.current
    T.P.require(state is prepared['old_state'] is votes.state,'conditional_commit_state')
    T.P.require(values['committed'] is votes.head and len(view.queue)==0,'conditional_commit_original_pop')
    T.P.require(binding.next_token==votes.token and votes.token not in binding.consumed_tokens,
        'conditional_commit_token')
    control.unchanged(call,values['self'],values['side'])
    control.provider._parts.V.Provider.after_history_consume(control.provider,call,caller)
    binding.policy.arm('placement',prepared['evidence'],state,prepared['proof'])
    binding.owner.add_placement(prepared['evidence'],prepared['evidence'].available_at)
    binding.grid=prepared['grid']
    binding.consumed_tokens.add(votes.token)
    binding.next_token=binding.next_started=binding.candidate=None
    binding.conditional_firing_committed=dict(proof=T.V.deepcopy(prepared['proof']),
        evidence=prepared['evidence'],votes=votes,previous_anchor=binding.hidden_anchor)
    binding.conditional_firing_proposal=None
    call['consumed']=True
    control.unchanged(call,values['self'],values['side'])
    T.P.require(control._parts.T.board_key(values['sm'].context.confirmed_board)==binding.current==current,
        'conditional_commit_current')
    T.P.require(binding.owner.state.current is state.current and not binding.owner.state.origins
        and not binding.owner.state.debts,'conditional_commit_old_slot')
    rows.append(dict(stage='conditional_placement_committed',frame=view.frame,
        proof=prepared['proof'],state=asdict(binding.owner.state),old_state=asdict(state),
        original_pop=True,native_counter_unchanged=True,physical_certified=False,
        original_event=False,settled=False,current_permission=False))


def install(stack: Any, factory: Any, pipe: Any, patch: Any, rows: list[Any]) -> None:
    adapter=FunctionType(A.install.__code__,dict(vars(A),prepare=captured))
    adapter(stack,factory,pipe,patch,rows)
    control=factory.controller
    cls=type(control)
    v1=cls.prepared.__globals__['V1']
    old_prepare,old_call,old_consume=v1.prepared,cls.call,cls.consumed_history
    def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
        value=old_prepare(self,binding,item,sm,raw,signals,view)
        proposal=getattr(binding,'conditional_firing_proposal',None)
        if proposal is None: return value
        assert value is None and proposal['old_state'] is binding.owner.state
        assert proposal['proof']['available_frame']==view.frame and proposal['proof']['kind']==KIND
        return proposal|dict(kind=KIND,votes=getattr(binding,A.VOTES))
    def call(self: Any, caller: Any, binding: Any, view: Any, obj: Any, side: str, proposal: Any) -> Any:
        if proposal is not None and proposal.get('kind')==KIND:
            return self._parts.C.Controller.call(self,caller,binding,view,obj,side,proposal)
        return old_call(self,caller,binding,view,obj,side,proposal)
    def completed(self: Any, call: Any, caller: Any) -> Any:
        if call['prepared'].get('kind')==KIND: return consumed(self,call,caller,rows)
        return old_consume(self,call,caller)
    patch(stack,v1,'prepared',prepared)
    patch(stack,cls,'call',call)
    patch(stack,cls,'consumed_history',completed)
