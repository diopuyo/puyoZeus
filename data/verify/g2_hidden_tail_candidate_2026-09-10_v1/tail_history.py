"""非発火の最終単headを原配置会計へ閉じ、次手と現在許可は作らない。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import path_support as P
import prefix_witness as W
import tail_witness as T

KIND = 'hidden_single_tail_history/v1'


def prepare(lifecycle: Any, core: Any, control: Any, binding: Any, item: Any,
            sm: Any, signals: Any, view: Any) -> Any:
    previous = T.source(control,binding,item,view)
    captured = W.observed(control,binding,signals,view)
    if (previous is None or captured is None or signals.is_match_active is not True
        or signals.chain_event is not None or signals.effect_gate_window_active is not False):
        binding.hidden_tail_votes = None
        return None
    raw,raw_proof = captured
    possible = tuple(g for g in P.options(core,binding.grid,P.pair(item.pair)) if P.compatible(raw,g))
    if len(possible)!=1 or possible[0]!=previous.support.final or not W.nonfiring(core,previous.support):
        binding.hidden_tail_votes = None
        return None
    votes = T.vote(control,binding,item,possible[0],raw,raw_proof,view)
    if votes.count<W.MIN_VOTES: return None
    binding.clear_first,binding.clear_last = votes.first,votes.last
    binding.clear_grid,binding.clear_count = votes.final,votes.count
    proof = dict(kind=KIND,token=item.token,pair=item.pair,occurred=votes.first,
        available_frame=view.frame,available_time=view.clock,clear_last=votes.last,
        clear_observations=votes.count,available_window=False,new_token=None,
        observed_raw=raw,raw_capture=raw_proof,inferred_final=votes.final,
        prefix_consumed_frame=previous.last[0],prefix_source=T.V.validate(control,binding,previous),
        inferred_grid_is_observed=False,physical_certified=False,current_permission=False)
    value = lifecycle.prepare(control.inventory,binding,view,votes.final,proof)
    P.require(value is not None,'tail_lifecycle_rejected')
    return value | dict(kind=KIND,tail_votes=votes)


def consumed(control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
    binding,view,prepared = call['binding'],call['view'],call['prepared']
    values,votes = caller.f_locals,prepared['tail_votes']
    state,current = binding.owner.state,binding.current
    P.require(state is prepared['old_state'] is votes.state,'tail_state_changed')
    P.require(values['committed'] is votes.head and len(view.queue)==0,'tail_native_pop')
    control.unchanged(call,values['self'],values['side'])
    # 二枠handoffの保存枠ではなく、発火単headと同じ原J消費票を検査する。
    control.provider._parts.V.Provider.after_history_consume(control.provider,call,caller)
    binding.policy.arm('placement',prepared['evidence'],state,prepared['proof'])
    binding.owner.add_placement(prepared['evidence'],prepared['evidence'].available_at)
    binding.grid = votes.final
    binding.consumed_tokens.add(votes.token)
    binding.next_token = binding.next_started = binding.candidate = None
    binding.clear_grid = binding.clear_first = binding.clear_last = None
    binding.clear_count,call['consumed'],binding.hidden_tail_consumed = 0,True,True
    control.unchanged(call,values['self'],values['side'])
    P.require(control._parts.T.board_key(values['sm'].context.confirmed_board)==binding.current==current,
        'tail_current_changed')
    P.require(not binding.owner.state.debts and not binding.owner.state.origins,'tail_origin_created')
    rows.append(dict(kind=KIND,frame=view.frame,current_permission=False,source=prepared['proof'],
        state=asdict(binding.owner.state),old_current=current,inferred_final=binding.grid,
        native_counter_unchanged=True,physical_certified=False,next_action_created=False))
