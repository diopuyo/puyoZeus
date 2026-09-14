"""次配置の元commit来歴を検査して、同じ原自然STABLE入口を再利用する。"""
from __future__ import annotations
from dataclasses import replace
import json
from typing import Any
import continuation_witness as T
import natural_exit as N
import conditional_current as C


def eligible(control: Any, binding: Any, signals: Any, call: Any) -> Any:
    view,state = call['view'],binding.owner.state
    if getattr(binding,'hidden_current',None) is not None: return None
    if (call['prepared'] is not None or call['consumed'] or binding.next_token is not None
        or view.refs or view.tokens or signals.is_match_active is not True
        or signals.chain_event is not None or signals.effect_gate_window_active is not False): return None
    if view.quiet is not True or not control._parts.T.valid_pair(view.next_pair): return None
    T.P.require(not state.origins and not state.debts and binding.scope==view.scope,'continuation_exit_scope')
    T.P.require(state.counter==control.inventory.S.color_counts(binding.grid),'continuation_exit_counter')
    proof = T.committed(control,binding)
    T.P.require(view.frame>proof['available_frame'],'continuation_exit_clock')
    captured = T.W.observed(control,binding,signals,view)
    return captured if captured is not None and T.P.compatible(captured[0],binding.grid) else None


def install(stack: Any, patch: Any) -> None:
    old_eligible,old_make = N.eligible,C.make
    def selected(control: Any, binding: Any, signals: Any, call: Any) -> Any:
        if getattr(binding,'hidden_last_placement',None) is not None:
            return eligible(control,binding,signals,call)
        return old_eligible(control,binding,signals,call)
    def make(control: Any, call: Any, result: Any, provisional: Any) -> Any:
        value = old_make(control,call,result,provisional)
        if value is None or getattr(call['binding'],'hidden_last_placement',None) is None: return value
        certificate,side = value
        proof = json.loads(certificate.evidence_json)
        proof['continuation_committed_source'] = T.committed(control,call['binding'])
        return replace(certificate,evidence_json=C.encoded(proof)),side
    patch(stack,N,'eligible',selected)
    patch(stack,C,'make',make)
