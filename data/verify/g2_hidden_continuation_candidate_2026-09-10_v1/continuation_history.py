"""原単head pop後だけS配置をcommit。確率盤面を整数currentへ昇格させない。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import continuation_witness as T

P,W,KIND = T.P,T.W,T.KIND


def prepare(lifecycle: Any, core: Any, control: Any, binding: Any, item: Any,
            sm: Any, signals: Any, view: Any) -> Any:
    ready = T.source(control,binding,item,view)
    captured = W.observed(control,binding,signals,view)
    if not (ready and captured is not None and signals.is_match_active is True
        and signals.chain_event is None and signals.effect_gate_window_active is False):
        binding.hidden_continuation_votes = None
        return None
    raw,raw_proof = captured
    possible = tuple(g for g in P.options(core,binding.grid,P.pair(item.pair)) if P.compatible(raw,g))
    if len(possible)!=1 or core.simulate_chain(P.board(core,possible[0])).chain_count:
        binding.hidden_continuation_votes = None
        return None
    votes = T.vote(control,binding,item,possible[0],raw,raw_proof,view)
    if votes.count<W.MIN_VOTES: return None
    binding.clear_first,binding.clear_last = votes.first,votes.last
    binding.clear_grid,binding.clear_count = votes.final,votes.count
    proof = dict(kind=KIND,token=item.token,pair=item.pair,occurred=votes.first,
        available_frame=view.frame,available_time=view.clock,clear_last=votes.last,
        clear_observations=votes.count,available_window=False,new_token=None,
        observed_raw=raw,raw_capture=raw_proof,inferred_final=votes.final,
        previous_conditional_certificate=binding.hidden_anchor.certificate.evidence_json,
        prefix_source=T.V.validate(control,binding,binding.hidden_prefix_votes),
        inferred_grid_is_observed=False,physical_certified=False,current_permission=False)
    value = lifecycle.prepare(control.inventory,binding,view,votes.final,proof)
    P.require(value is not None,'continuation_lifecycle')
    return value|dict(kind=KIND,continuation_votes=votes)


def consumed(control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
    binding,view,prepared = call['binding'],call['view'],call['prepared']
    values,votes = caller.f_locals,prepared['continuation_votes']
    state,current = binding.owner.state,binding.current
    P.require(state is prepared['old_state'] is votes.state,'continuation_state_changed')
    P.require(values['committed'] is votes.head and len(view.queue)==0,'continuation_native_pop')
    control.unchanged(call,values['self'],values['side'])
    control.provider._parts.V.Provider.after_history_consume(control.provider,call,caller)
    binding.policy.arm('placement',prepared['evidence'],state,prepared['proof'])
    binding.owner.add_placement(prepared['evidence'],prepared['evidence'].available_at)
    binding.grid = votes.final
    binding.consumed_tokens.add(votes.token)
    binding.next_token = binding.next_started = binding.candidate = None
    binding.clear_grid = binding.clear_first = binding.clear_last = None
    binding.clear_count,call['consumed'] = 0,True
    binding.hidden_last_placement = dict(proof=T.V.deepcopy(prepared['proof']),votes=votes,action=state.action,
        previous_anchor=binding.hidden_anchor)
    binding.hidden_continuation_votes = None
    control.unchanged(call,values['self'],values['side'])
    P.require(control._parts.T.board_key(values['sm'].context.confirmed_board)==binding.current==current,
        'continuation_integer_current_changed')
    P.require(not binding.owner.state.origins and not binding.owner.state.debts,'continuation_origins')
    T.committed(control,binding)
    rows.append(dict(kind=KIND,frame=view.frame,current_permission=False,source=prepared['proof'],
        state=asdict(binding.owner.state),old_current=current,inferred_final=binding.grid,
        native_counter_unchanged=True,physical_certified=False,next_action_created=False))
