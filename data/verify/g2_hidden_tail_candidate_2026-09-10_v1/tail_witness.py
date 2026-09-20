"""先頭の原消費後に、残った同headへ新しい支持票だけを束縛する。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import path_support as P
import prefix_witness as W
import prefix_provenance as V


@dataclass
class TailVotes:
    binding: Any
    state: Any
    item: Any
    scope: Any
    queue: Any
    head: Any
    token: str
    final: Any
    raw: Any
    raw_proof: Any
    first: tuple[int,float]
    last: tuple[int,float]
    count: int


def source(control: Any, binding: Any, item: Any, view: Any) -> Any:
    previous = getattr(binding,'hidden_prefix_votes',None)
    P.require(previous is not None and previous.binding is binding,'tail_prefix_source')
    V.validate(control,binding,previous)
    held = previous.held
    P.require(held.consumed and held.binding is binding and binding.grid==previous.support.prefix,
        'tail_prefix_not_committed')
    P.require(view.scope==binding.scope==held.scope and not view.added
        and len(view.refs)==len(view.tokens)==len(view.queue)==1,'tail_scope_or_slots')
    P.require(view.queue is held.queue and view.refs[0] is held.refs[1] is item.pair
        and item.queue is view.queue and view.queue[0] is item.pair,'tail_reference')
    P.require(binding.next_token==item.token==view.tokens[0]==held.tokens[1]
        and item.token not in binding.consumed_tokens,'tail_token')
    P.require(view.scope[-1] not in control.provider.handoff_proofs,'tail_stale_held')
    row = control.provider.link.current(view)
    owner = control.provider.journal.fifo.entries[(id(row['pipe']),view.scope[-1])]
    P.require(owner['queue'] is view.queue and tuple(owner['tokens'])==view.tokens
        and len(owner['refs'])==1 and owner['refs'][0] is item.pair,'tail_J_owner')
    if not (view.quiet is True and row['accepted']==view.next_pair==item.next_pair
        and row['dnext']==view.dnext_pair==item.dnext_pair): return None
    P.require(binding.next_started is not None and view.frame>binding.next_started[0]
        and view.frame>previous.last[0],'tail_started_clock')
    return previous


def vote(control: Any, binding: Any, item: Any, final: Any, raw: Any,
         proof: Any, view: Any) -> TailVotes:
    previous = getattr(binding,'hidden_tail_votes',None)
    now,state = (view.frame,view.clock),binding.owner.state
    same = (previous is not None and previous.binding is binding and previous.state is state
        and previous.item is item and previous.scope==view.scope and previous.queue is view.queue
        and previous.head is item.pair and previous.token==item.token
        and previous.final==final and previous.raw==raw
        and control.provider.adjacent(previous.last,view))
    result = TailVotes(binding,state,item,view.scope,view.queue,item.pair,item.token,final,raw,proof,
        previous.first if same else now,now,previous.count+1 if same else 1)
    binding.hidden_tail_votes = result
    return result
