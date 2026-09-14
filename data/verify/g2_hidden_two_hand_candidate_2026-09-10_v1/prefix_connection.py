"""元v3 caller/grace検査を保持し、新kindだけ現在mergeを呼ばずに消費する。"""
from __future__ import annotations
from typing import Any
import prefix_history as H


def single_head_wait(control: Any, binding: Any, item: Any, view: Any) -> bool:
    """初回/先頭処理後の一枠はhandoffではない。原J消費を許可しない。"""
    if len(view.refs)!=1: return False
    state = binding.owner.state
    initial = not state.history and state.current is None
    if not initial and not getattr(binding,'hidden_prefix_consumed',False): return False
    H.P.require(binding.scope==view.scope and len(view.tokens)==1,'single_scope')
    H.P.require(view.scope[-1] not in control.provider.handoff_proofs,'single_stale_held')
    H.P.require(binding.next_token==item.token==view.tokens[0],'single_token')
    H.P.require(item.queue is view.queue and item.pair is view.refs[0]
        and len(view.queue)==1 and view.queue[0] is item.pair,'single_reference')
    return True


def install(stack: Any, factory: Any, base: Any, rows: list[Any]) -> None:
    from src import puyo_core_bridge as core
    control = factory.controller
    cls = type(control)
    v1 = cls.prepared.__globals__['V1']
    lifecycle, old_prepare = v1.L,v1.prepared
    old_call,old_consume = cls.call,cls.consumed_history
    def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
        if single_head_wait(self,binding,item,view):
            self.observe_clear(binding,item,sm,raw,signals,view)
            return None
        value = H.prepare(lifecycle,base.D,core,self,binding,item,sm,signals,view)
        if value is not None: return value
        return old_prepare(self,binding,item,sm,raw,signals,view)
    def call(self: Any, caller: Any, binding: Any, view: Any, pipe: Any, side: str, proposal: Any) -> Any:
        if proposal is None or proposal.get('kind')!=H.KIND:
            return old_call(self,caller,binding,view,pipe,side,proposal)
        value = self._parts.C.Controller.call(self,caller,binding,view,pipe,side,proposal)
        value['deferred_handoff'] = self.provider.handoff_proofs[side]
        base.D.from_call(value,consumed=False)
        return value
    def consumed(self: Any, value: Any, caller: Any) -> None:
        if value['prepared'].get('kind')==H.KIND:
            return H.consumed(self,value,caller,rows)
        return old_consume(self,value,caller)
    base.patch(stack,v1,'prepared',prepared)
    base.patch(stack,cls,'call',call)
    base.patch(stack,cls,'consumed_history',consumed)
