"""最後の条件付き盤面と原Jの新単headから、別の配置を検証する。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import path_support as P
import prefix_witness as W
import prefix_provenance as V
import tail_witness as T
import lifetime as L

KIND = 'hidden_conditional_continuation_history/v1'


def source(control: Any, binding: Any, item: Any, view: Any) -> bool:
    P.require(L.compatible(binding.owner.state,binding),'continuation_anchor')
    value = binding.hidden_anchor.certificate
    P.require(view.scope==binding.scope==item.scope and len(view.refs)==len(view.tokens)==len(view.queue)==1,
        'continuation_scope_slots')
    P.require(view.queue is item.queue and view.refs[0] is item.pair is view.queue[0]
        and binding.next_token==item.token==view.tokens[0] and item.token not in binding.consumed_tokens,
        'continuation_reference')
    P.require(item.baseline==binding.grid==value.inferred_grid
        and binding.owner.state.counter==control.inventory.S.color_counts(binding.grid),'continuation_baseline')
    P.require(binding.owner.state.action==value.action+1 and item.started>value.frame,'continuation_action')
    P.require(view.scope[-1] not in control.provider.handoff_proofs,'continuation_stale_handoff')
    live = control.provider.link.current(view)
    owner = control.provider.journal.fifo.entries[(id(live['pipe']),view.scope[-1])]
    P.require(owner['queue'] is view.queue and tuple(owner['tokens'])==view.tokens
        and len(owner['refs'])==1 and owner['refs'][0] is item.pair,'continuation_original_J')
    V.validate(control,binding,binding.hidden_prefix_votes)
    return (view.frame>item.started and view.quiet is True and not view.added
        and live['accepted']==view.next_pair==item.next_pair
        and live['dnext']==view.dnext_pair==item.dnext_pair)


def vote(control: Any, binding: Any, item: Any, final: Any, raw: Any,
         proof: Any, view: Any) -> Any:
    previous = getattr(binding,'hidden_continuation_votes',None)
    now,state = (view.frame,view.clock),binding.owner.state
    same = (previous is not None and previous.binding is binding and previous.state is state
        and previous.item is item and previous.scope==view.scope and previous.queue is view.queue
        and previous.head is item.pair and previous.token==item.token and previous.final==final
        and previous.raw==raw and control.provider.adjacent(previous.last,view))
    value = T.TailVotes(binding,state,item,view.scope,view.queue,item.pair,item.token,final,raw,proof,
        previous.first if same else now,now,previous.count+1 if same else 1)
    binding.hidden_continuation_votes = value
    return value


def prior_source(control: Any, binding: Any, saved: Any, proof: Any) -> None:
    """新anchorへ更新後も、消費時に保持した旧証明を元scope/配置基点へ結ぶ。"""
    prior = saved['previous_anchor']
    P.require(type(prior) is L.AnchorCorrespondence,'continuation_prior_type')
    value,state = prior.certificate,saved['votes'].state
    P.require(type(value) is L.C.ConditionalCurrent and L.C.intact(value),'continuation_prior_certificate')
    P.require(prior.owner_scope==L.C.encoded(asdict(binding.owner.state.scope))
        and L.scope_matches(state,value) and value.scope==binding.scope,'continuation_prior_scope')
    P.require(value.action+1==saved['action']==state.action
        and value.inferred_grid==proof['before_grid']
        and value.integer_anchor==L.C.anchor(state),'continuation_prior_action_grid')
    P.require(value.evidence_json==proof['previous_conditional_certificate'],'continuation_prior_contents')
    actual = V.validate(control,binding,binding.hidden_prefix_votes)
    P.require(control.inventory.digest(actual)==control.inventory.digest(proof['prefix_source']),
        'continuation_prefix_source_changed')


def committed(control: Any, binding: Any) -> dict[str,Any]:
    saved = binding.hidden_last_placement
    state,proof = binding.owner.state,saved['proof']
    entries = [e for e in state.history if e.kind=='placement' and e.action==saved['action']]
    P.require(len(entries)==1 and entries[0].event_id==control.inventory.digest(proof),
        'continuation_committed_event')
    policy = getattr(binding.policy,'accounting',binding.policy)
    logs = [r for r in policy.proofs if r['purpose']=='placement' and r['proof_sha']==entries[0].event_id]
    P.require(len(logs)==1 and control.inventory.digest(logs[0]['proof'])==entries[0].event_id,
        'continuation_committed_policy')
    P.require(proof['kind']==KIND and proof['scope']==asdict(state.scope)
        and proof['grid']==binding.grid==saved['votes'].final
        and proof['token'] in binding.consumed_tokens,'continuation_committed_scope_grid')
    P.require(proof['available_frame']==entries[0].available_at.frame==saved['votes'].last[0]
        and proof['available_time']==entries[0].available_at.time_sec,'continuation_committed_clock')
    prior_source(control,binding,saved,proof)
    return V.deepcopy(proof)
