"""原Jの二観測から提案だけを作る。pop/S/元イベントは許可しない。"""
from __future__ import annotations
from dataclasses import asdict
from types import MethodType
from typing import Any
import sys

KIND='hidden_conditional_firing_proposal/v1'
VOTES='conditional_firing_admission_votes'
FPS,STRIDE,MIN_VOTES=60,2,2


def world(control: Any, binding: Any, item: Any, signals: Any, view: Any) -> Any:
    import continuation_witness as T
    from src import puyo_core_bridge as core
    from src.production_config import GHOST_CHAIN_RULE_ENABLED
    ready=T.source(control,binding,item,view)
    observed=T.W.observed(control,binding,signals,view)
    if not (ready and observed is not None and signals.is_match_active is True
        and signals.chain_event is None and signals.effect_gate_window_active is False): return None
    raw,proof=observed
    options=tuple(g for g in T.P.options(core,binding.grid,T.P.pair(item.pair)) if T.P.compatible(raw,g))
    if len(options)!=1: return None
    prediction=core.simulate_chain(T.P.board(core,options[0]),exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    if prediction.chain_count<1: return None
    T.P.require(GHOST_CHAIN_RULE_ENABLED is True,'admission_adopted_rule')
    return raw,proof,options[0],prediction


def vote(control: Any, binding: Any, item: Any, view: Any, values: Any) -> Any:
    import continuation_witness as T
    raw,proof,placed,prediction=values
    previous=getattr(binding,VOTES,None)
    state,now=binding.owner.state,(view.frame,view.clock)
    same=(previous is not None and previous.binding is binding and previous.state is state
        and previous.item is item and previous.scope==view.scope and previous.queue is view.queue
        and previous.head is item.pair and previous.token==item.token and previous.final==placed
        and previous.raw==raw and control.provider.adjacent(previous.last,view))
    result=T.T.TailVotes(binding,state,item,view.scope,view.queue,item.pair,item.token,placed,raw,proof,
        previous.first if same else now,now,previous.count+1 if same else 1)
    setattr(binding,VOTES,result)
    return result


def proposal(control: Any, binding: Any, view: Any, votes: Any, values: Any, observations: Any) -> Any:
    import continuation_witness as T
    raw,raw_proof,placed,prediction=values
    proof=dict(kind=KIND,token=votes.token,pair=tuple(votes.head),occurred=votes.first,
        available_frame=view.frame,available_time=view.clock,clear_last=votes.last,
        clear_observations=votes.count,observed_raw=raw,raw_capture=raw_proof,observations=observations,
        inferred_final=placed,previous_conditional_certificate=binding.hidden_anchor.certificate.evidence_json,
        prefix_source=T.V.validate(control,binding,binding.hidden_prefix_votes),
        inferred_grid_is_observed=False,physical_certified=False,current_permission=False,
        simulation_rule='GHOST_CHAIN_RULE_ENABLED=True',predicted_final=prediction.final_board.to_dict()['grid'])
    lifecycle=type(control).prepared.__globals__['V1'].L
    # 原clear欄を書換えず、元関数に読取専用の同値viewを渡す。提案はwriterに渡さない。
    from types import SimpleNamespace
    proxy=SimpleNamespace(**vars(binding))
    proxy.clear_first,proxy.clear_last,proxy.clear_grid,proxy.clear_count=votes.first,votes.last,placed,votes.count
    value=lifecycle.prepare(control.inventory,proxy,view,placed,proof)
    T.P.require(value is not None,'admission_lifecycle_proposal')
    return value


def prepare(control: Any, binding: Any, item: Any, signals: Any, view: Any, rows: list[Any]) -> bool:
    import continuation_witness as T
    values=world(control,binding,item,signals,view)
    if values is None:
        setattr(binding,VOTES,None)
        return False
    votes=vote(control,binding,item,view,values)
    record=dict(stage='conditional_firing_vote',frame=view.frame,count=votes.count,
        scope=view.scope,token=item.token,raw=values[0],raw_proof=values[1],placed=values[2],
        prediction=values[3].final_board.to_dict()['grid'],state=asdict(binding.owner.state),
        observed_next=view.next_pair,observed_dnext=view.dnext_pair,quiet=view.quiet,
        physical_certified=False,original_pop=False,accounting_permission=False)
    if votes.count>=MIN_VOTES:
        deferred=getattr(binding,'conditional_firing_deferred',None)
        T.P.require(deferred is not None and deferred['frame']==view.frame and deferred['token']==item.token,
            'admission_same_call_formula')
        prior=[r for r in rows if r['stage']=='conditional_firing_vote' and r['frame']==votes.first[0]]
        T.P.require(len(prior)==1 and prior[0]['token']==item.token and prior[0]['scope']==view.scope
            and prior[0]['raw']==values[0] and prior[0]['state']==asdict(votes.state),'admission_first_capture')
        observations=tuple({k:r[k] for k in ('frame','scope','token','raw','raw_proof')} for r in (prior[0],record))
        value=proposal(control,binding,view,votes,values,observations)
        record['proposal']={k:asdict(v) if k in ('evidence','old_state') else v for k,v in value.items()}
    rows.append(record)
    return True


def formula(factory: Any, pipe: Any, frame: Any, binding: Any, rows: list[Any]) -> None:
    import firing_handoff_fixed as F
    import continuation_witness as T
    T.P.require(sys._getframe(1) is frame,'admission_immediate_frame')
    actual=F.V5.V4.bound_update(pipe)
    caller=frame.f_back
    T.P.require(caller.f_code is actual.__code__ and caller.f_globals is actual.__globals__, 'admission_update')
    T.P.require(caller.f_locals['self'] is frame.f_locals['self'] is pipe,'admission_pipe')
    T.P.require(actual.__globals__['__next_live'] is factory.provider.journal.controller,'admission_N')
    votes=getattr(binding,VOTES,None)
    clock=caller.f_locals['frame_idx']
    T.P.require(votes is not None and votes.count==1 and votes.last[0]+STRIDE==clock,'admission_prior_vote')
    T.P.require(votes.state is binding.owner.state and votes.item is binding.candidate
        and votes.head is pipe._pending_tsumo_1p[0] and votes.token==binding.next_token,'admission_prior_identity')
    key=factory.controller._parts.T.board_key
    T.P.require(key(caller.f_locals['cnn_1p_raw'])==key(caller.f_locals['cnn_1p'])==votes.raw,'admission_formula_raw')
    T.P.require(key(frame.f_locals['prev_confirmed'])==binding.current and clock/FPS==frame.f_locals['time_sec'],
        'admission_formula_current_clock')
    T.P.require(pipe._active_chain_1p is None and not binding.owner.state.origins and not binding.owner.state.debts,
        'admission_no_origin')
    binding.conditional_firing_deferred=dict(frame=clock,token=votes.token,scope=binding.scope,
        original_pop=False,original_event=False,physical_certified=False)
    rows.append(dict(stage='conditional_formula_deferred',**binding.conditional_firing_deferred))


def install(stack: Any, factory: Any, pipe: Any, patch: Any, rows: list[Any]) -> None:
    control=factory.controller
    v1=type(control).prepared.__globals__['V1']
    old_prepare,old_formula=v1.prepared,pipe._apply_chain_formula_early_fire
    def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
        if getattr(binding,'hidden_anchor',None) is not None:
            if prepare(self,binding,item,signals,view,rows): return None
        return old_prepare(self,binding,item,sm,raw,signals,view)
    def delayed(self: Any, side: str, time_sec: float, prev_confirmed: Any) -> Any:
        binding=control.history.get(side)
        if side=='1P' and binding is not None and getattr(binding,VOTES,None) is not None:
            return formula(factory,self,sys._getframe(),binding,rows)
        return old_formula(side=side,time_sec=time_sec,prev_confirmed=prev_confirmed)
    patch(stack,v1,'prepared',prepared)
    patch(stack,pipe,'_apply_chain_formula_early_fire',MethodType(delayed,pipe))
