"""既存prepared caller資格を維持し、条件付きanchorの新単headだけを接続する。"""
from __future__ import annotations
from typing import Any
import continuation_history as H
import continuation_exit as E


def install(stack: Any, factory: Any, patch: Any, rows: list[Any]) -> None:
    from src import puyo_core_bridge as core
    control = factory.controller
    cls = type(control)
    v1 = cls.prepared.__globals__['V1']
    lifecycle,old_prepare = v1.L,v1.prepared
    old_call,old_consume = cls.call,cls.consumed_history
    def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
        if getattr(binding,'hidden_anchor',None) is not None:
            return H.prepare(lifecycle,core,self,binding,item,sm,signals,view)
        return old_prepare(self,binding,item,sm,raw,signals,view)
    def call(self: Any, caller: Any, binding: Any, view: Any, pipe: Any, side: str, proposal: Any) -> Any:
        if proposal is not None and proposal.get('kind')==H.KIND:
            return self._parts.C.Controller.call(self,caller,binding,view,pipe,side,proposal)
        return old_call(self,caller,binding,view,pipe,side,proposal)
    def consumed(self: Any, value: Any, caller: Any) -> None:
        if value['prepared'].get('kind')==H.KIND: return H.consumed(self,value,caller,rows)
        return old_consume(self,value,caller)
    patch(stack,v1,'prepared',prepared)
    patch(stack,cls,'call',call)
    patch(stack,cls,'consumed_history',consumed)
    E.install(stack,patch)
