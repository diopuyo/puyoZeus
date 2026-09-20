"""消費後も保持する原prefix来歴の内容整合性。現raw/物理認証は別責務。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from typing import Any

FPS, PAIR_SIZE = 60, 2
KIND = 'directional_commit_original_J/v1'


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('hidden_prefix_provenance:' + reason)


def contents(control: Any, held: Any) -> dict[str, Any]:
    proof=held.proof
    require(type(proof) is dict and control.inventory.digest(proof)==held.proof_digest,
        'proof_changed')
    require(proof['kind']==KIND and len(held.tokens)==len(held.refs)==PAIR_SIZE,'proof_kind_or_slots')
    require(proof['old_token']==held.tokens[0] and proof['new_token']==held.tokens[1]
        and held.tokens[0]!=held.tokens[1], 'tokens')
    scope=held.scope
    require(proof['source_id']==scope[0] and proof['run_id']==scope[1]
        and proof['software_epoch']==scope[2] and proof['side']==scope[-1], 'source_scope')
    return proof


def identity(control: Any, binding: Any, previous: Any) -> Any:
    held=previous.held
    require(previous.binding is binding and held.binding is binding and held.consumed is True,
        'prefix_not_consumed')
    require(binding.scope==held.scope==held.item.scope==held.view.scope,'binding_scope')
    require(held.item.queue is held.queue is held.view.queue and held.item.pair is held.refs[0]
        and held.item.token==held.tokens[0], 'retained_head')
    require(tuple(held.view.tokens)==held.tokens and len(held.view.refs)==PAIR_SIZE
        and all(a is b for a,b in zip(held.view.refs,held.refs)), 'retained_view')
    require(held.tokens[0] in binding.consumed_tokens
        and held.tokens[1] in control.provider.link.used,'consumption_not_retained')
    state_scope=binding.owner.state.scope
    require(state_scope.source_sha256==held.scope[0].removeprefix('sha256:')
        and state_scope.run_id==held.scope[1] and state_scope.side==held.scope[-1]
        and state_scope.reset_epoch==held.scope[2],'owner_scope')
    return held


def clocks(proof: Any, held: Any, previous: Any) -> None:
    source=proof['available_frame']
    first,last=previous.first,previous.last
    require(type(source) is int and proof['available_time']==source/FPS,'source_clock')
    require(type(first[0]) is int and type(last[0]) is int
        and first[1]==first[0]/FPS and last[1]==last[0]/FPS,'vote_clock')
    require(type(held.item.started) is int and held.item.started<source<=first[0]<last[0],
        'source_vote_order')
    require(held.view.frame==last[0] and held.view.clock==last[1],'consumed_clock')



def committed(control: Any, binding: Any, previous: Any, proof: Any) -> dict[str, Any]:
    """原S履歴event_idから既commit policyログを引き、元来歴全内容を照合。"""
    held=previous.held
    entries=[e for e in binding.owner.state.history if e.kind=='placement'
        and e.action==held.state.action and e.available_at.frame==held.view.frame]
    require(len(entries)==1,'committed_placement_missing')
    entry=entries[0]
    accounting=getattr(binding.policy,'accounting',binding.policy)
    logs=getattr(accounting,'proofs',None)
    require(type(logs) is list,'proof_log_missing')
    matched=[r for r in logs if r['purpose']=='placement' and r['proof_sha']==entry.event_id]
    require(len(matched)==1,'committed_log_missing')
    saved=matched[0]['proof']
    require(control.inventory.digest(saved)==entry.event_id,'committed_proof_digest')
    require(saved['kind']=='hidden_two_hand_prefix_history/v1'
        and saved['scope']==asdict(binding.owner.state.scope),'committed_kind_scope')
    require(saved['token']==held.tokens[0] and saved['new_token']==held.tokens[1]
        and saved['pair']==held.refs[0],'committed_tokens')
    require(saved['available_frame']==held.view.frame and saved['available_time']==held.view.clock
        and tuple(saved['occurred'])==previous.first and tuple(saved['clear_last'])==previous.last,
        'committed_clock')
    require(saved['before_grid']==held.item.baseline and saved['grid']==previous.support.prefix
        and saved['observed_raw']==previous.support.raw,'committed_grids')
    require(control.inventory.digest(saved['directional_commit'])==control.inventory.digest(proof),
        'committed_directional')
    return saved['directional_commit']

def validate(control: Any, binding: Any, previous: Any) -> dict[str, Any]:
    """現在queue長/grid/next_tokenに依存せず、BB消費後も使える元来歴票を返す。"""
    require(previous is not None,'missing_prefix')
    held=identity(control,binding,previous)
    proof=contents(control,held)
    clocks(proof,held,previous)
    return deepcopy(committed(control,binding,previous,proof))
