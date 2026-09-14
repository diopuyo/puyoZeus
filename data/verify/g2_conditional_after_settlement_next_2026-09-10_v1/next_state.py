"""閉鎖済みの元S履歴を保持し、原Jで始めた一手だけを所有する。"""
from __future__ import annotations
from dataclasses import dataclass,replace,asdict
from typing import Any
import json
import bundle as B
import lifetime as L
import path_support as P

KIND='conditional_after_settlement_next_placement/v1'
ORIGINAL_REGISTERED=B.C.L.registered


@dataclass(frozen=True)
class NextHand:
    binding: Any
    before: Any
    state: Any
    grid: Any
    anchor: Any
    scope: Any
    queue: Any
    head: Any
    token: str
    started: Any
    log_digest: str
    consumed_before: frozenset[str]
    registration_digest: str
    placement: Any=None


def get(control: Any,binding: Any) -> Any:
    return getattr(control,'conditional_next_hands',{}).get(id(binding))


def registration_digest(p: Any,saved: Any) -> str:
    """元OriginEvidenceだけを直列化し、登録の全欄を改変せず固定する。"""
    return p.digest(dict(saved,origin=asdict(saved['origin'])))


def current(control: Any,binding: Any) -> NextHand:
    value=get(control,binding); p=control.inventory; state=binding.owner.state
    P.require(type(value) is NextHand and value.binding is binding,'postnext_owner')
    P.require(state is value.state and binding.grid==value.grid and binding.scope==value.scope,'postnext_state')
    P.require(state.scope==value.before.scope and state.action==value.before.action+1,'postnext_action')
    P.require(state.origins==value.before.origins and state.consumed_ids==value.before.consumed_ids
        and not state.debts and state.counter==p.S.color_counts(value.grid),'postnext_closed_history')
    P.require(L.C.anchor(state)==value.anchor.certificate.integer_anchor,'postnext_old_slot')
    policy=getattr(binding.policy,'accounting',binding.policy)
    P.require(p.digest(policy.proofs)==value.log_digest,'postnext_proof_log')
    P.require(registration_digest(p,binding.conditional_firing_registered)==value.registration_digest,
        'postnext_registration_log_binding')
    if value.placement is None:
        P.require(state.history==value.before.history and binding.next_token==value.token
            and frozenset(binding.consumed_tokens)==value.consumed_before,'postnext_pending')
    else:
        P.require(state.history[:-1]==value.before.history and state.history[-1].event_id==p.digest(value.placement)
            and state.history[-1].kind=='placement' and binding.next_token is None,'postnext_placement')
        P.require(frozenset(binding.consumed_tokens)==value.consumed_before|{value.token},'postnext_consumed')
    return value


def registered(control: Any,binding: Any) -> Any:
    if get(control,binding) is None: return ORIGINAL_REGISTERED(control,binding)
    current(control,binding)
    return binding.conditional_firing_registered


def start(control: Any,binding: Any,view: Any,rows: list[Any]) -> NextHand:
    original=ORIGINAL_REGISTERED(control,binding)
    saved_digest=registration_digest(control.inventory,original)
    P.require(original.get('settled') is True and get(control,binding) is None,'postnext_settled')
    state,anchor=binding.owner.state,binding.hidden_anchor
    P.require(L.compatible(state,binding) and anchor.certificate.action==state.action
        and anchor.certificate.frame<view.frame,'postnext_anchor')
    proof=json.loads(anchor.certificate.evidence_json)
    P.require(proof['kind']=='conditional_hidden_after_firing_current/v1'
        and proof['conditional_settlement']==json.loads(control.inventory.encoded(original['settlement_proof'])),
        'postnext_settlement_certificate')
    P.require(view.scope==binding.scope and len(view.refs)==len(view.tokens)==len(view.queue)==1
        and view.added==view.tokens and view.refs[0] is view.queue[0],'postnext_fresh_head')
    live=control.provider.link.current(view)
    owner=control.provider.journal.fifo.entries[(id(live['pipe']),view.scope[-1])]
    P.require(owner['queue'] is view.queue and owner['refs'][0] is view.refs[0]
        and tuple(owner['tokens'])==view.tokens,'postnext_original_J')
    P.require(control._parts.T.valid_pair(view.refs[0]),'postnext_pair')
    p=control.inventory; consumed=frozenset(binding.consumed_tokens)
    control._parts.C.H.start(p,binding,view.tokens[0],view.frame,view.clock)
    policy=getattr(binding.policy,'accounting',binding.policy)
    value=NextHand(binding,state,binding.owner.state,binding.grid,anchor,view.scope,view.queue,
        view.refs[0],view.tokens[0],(view.frame,view.clock),p.digest(policy.proofs),consumed,saved_digest)
    control.conditional_next_hands[id(binding)]=value
    current(control,binding)
    rows.append(dict(stage='conditional_next_action_started',frame=view.frame,token=value.token,
        action=value.state.action,original_origin_retained=True,current_permission=False))
    return value


def closed_origins(state: Any,binding: Any,control: Any,legacy: Any) -> bool:
    if get(control,binding) is None: return legacy(state,binding)
    value=current(control,binding)
    return state is value.state and all(o.origin_id in state.consumed_ids for o in state.origins)


def consumed(control: Any,binding: Any,proof: Any,grid: Any) -> None:
    value=get(control,binding); state=binding.owner.state
    policy=getattr(binding.policy,'accounting',binding.policy)
    control.conditional_next_hands[id(binding)]=replace(value,state=state,grid=grid,
        placement=json.loads(control.inventory.encoded(proof)),log_digest=control.inventory.digest(policy.proofs))
    current(control,binding)
