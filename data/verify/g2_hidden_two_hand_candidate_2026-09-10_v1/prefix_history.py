"""二手の現観測から先頭だけを原履歴へ記帳し、現在盤面を公開しない。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import path_support as P
import prefix_witness as W

KIND = 'hidden_two_hand_prefix_history/v1'


def prepare(lifecycle: Any, held_module: Any, core: Any, control: Any, binding: Any,
            item: Any, sm: Any, signals: Any, view: Any) -> Any:
    if len(view.refs)!=P.PAIR_SIZE or getattr(binding,'hidden_prefix_consumed',False): return None
    captured = W.observed(control,binding,signals,view)
    if captured is None: return None
    held = held_module.capture(control,binding,item,view)
    if held is None: return None
    raw,raw_proof = captured
    hits = P.infer(core,binding.grid,view.refs[0],view.refs[1],raw)
    ready = (len(hits)==1 and signals.is_match_active and signals.chain_event is None
        and signals.effect_gate_window_active is False and held_module.ready(held,view))
    if not ready:
        W.clear(binding)
        return None
    support = hits[0]
    if not W.nonfiring(core,support):
        W.clear(binding)
        return None
    votes = W.vote(control,binding,held,support,view)
    if votes.count < W.MIN_VOTES: return None
    authorization = held_module.authorize(control,held,binding,item,view)
    binding.clear_first,binding.clear_last = votes.first,votes.last
    binding.clear_grid,binding.clear_count = support.prefix,votes.count
    proof = dict(kind=KIND,token=item.token,pair=item.pair,new_token=held.tokens[1],
        occurred=votes.first,available_frame=view.frame,available_time=view.clock,
        clear_last=votes.last,clear_observations=votes.count,available_window=False,
        directional_commit=held_module.deepcopy(held.proof),directional_consumption=authorization,
        observed_raw=raw,raw_capture=raw_proof,inferred_path=asdict(support),
        intermediate_grid_is_observed=False,physical_certified=False,current_permission=False)
    proposal = lifecycle.prepare(control.inventory,binding,view,support.prefix,proof)
    P.require(proposal is not None,'lifecycle_rejected')
    return proposal | dict(kind=KIND,hidden_prefix_votes=votes)


def consumed(control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
    binding,old_current = call['binding'],call['binding'].current
    old_confirmed = control._parts.T.board_key(caller.f_locals['sm'].context.confirmed_board)
    control._history_current_base.consumed_history(control,call,caller)
    P.require(call['consumed'] and binding.current==old_current==old_confirmed,'current_changed')
    P.require(control._parts.T.board_key(caller.f_locals['sm'].context.confirmed_board)==old_current,
              'SM_current_changed')
    P.require('current_merge' not in call and 'current_proof' not in call,'prefix_publication')
    votes = call['prepared']['hidden_prefix_votes']
    P.require(binding.grid==votes.support.prefix and len(call['view'].queue)==1,'prefix_commit')
    binding.hidden_prefix_consumed = True
    rows.append(dict(kind=KIND,frame=call['view'].frame,current_permission=False,
        inferred_prefix=binding.grid,observed_raw=votes.support.raw,old_current=old_current,
        state=asdict(binding.owner.state),source=call['prepared']['proof'],
        native_counter_unchanged=True,physical_certified=False))
