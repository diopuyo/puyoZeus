"""既存整数currentありの未消費単headも、二手票が揃うまで原Jを待つ。"""
from __future__ import annotations
from typing import Any
import prefix_connection as P


def waiting(original: Any, control: Any, binding: Any, item: Any, view: Any) -> bool:
    if original(control,binding,item,view): return True
    state = binding.owner.state
    if len(view.refs)!=1 or state.current is None or state.current.grid!=binding.current:
        return False
    P.H.P.require(binding.scope==view.scope and len(view.tokens)==1,'retained_single_scope')
    P.H.P.require(view.scope[-1] not in control.provider.handoff_proofs,'retained_single_stale_held')
    P.H.P.require(binding.next_token==item.token==view.tokens[0]
        and item.token not in binding.consumed_tokens,'retained_single_token')
    P.H.P.require(item.queue is view.queue and item.pair is view.refs[0]
        and len(view.queue)==1 and view.queue[0] is item.pair,'retained_single_reference')
    return True


def install(stack: Any, patch: Any) -> None:
    old = P.single_head_wait
    def selected(control: Any, binding: Any, item: Any, view: Any) -> bool:
        return waiting(old,control,binding,item,view)
    patch(stack,P,'single_head_wait',selected)
