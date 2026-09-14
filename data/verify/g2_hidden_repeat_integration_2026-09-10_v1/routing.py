"""未認証の乗換を拒否し、既存発火の委譲先を保持する。"""
from __future__ import annotations
import sys
from types import SimpleNamespace
from typing import Any
import hidden_bundle as B

N = B.CURRENT.C.H


def hidden_lane(binding: Any) -> bool:
    state = binding.owner.state
    return not state.origins and not state.debts and getattr(binding,'firing_ticket',None) is None


def owned(binding: Any) -> bool:
    return getattr(binding,'hidden_prefix_consumed',False)


def permitted(binding: Any) -> bool:
    ready = hidden_lane(binding)
    N.P.require(ready or not owned(binding),'combined_hidden_firing_transition_not_certified')
    return ready


def natural_install(stack: Any, control: Any, patch: Any, rows: list[Any]) -> None:
    cls,old = type(control),type(control).hold_transition
    values = cls.prepared.__globals__
    N.P.require(vars(sys.modules[values['__name__']]) is values,'combined_original_v3_module')
    module = SimpleNamespace(**values)
    N.P.require(hasattr(module,'T') and hasattr(module,'original_within'),'combined_exit_module')
    def hold(self: Any, sm: Any, signals: Any, binding: Any) -> Any:
        if permitted(binding) and getattr(binding,'hidden_tail_consumed',False):
            return N.installed(self,sm,signals,binding,module,rows)
        return old(self,sm,signals,binding)
    patch(stack,cls,'hold_transition',hold)


def dispatch(stack: Any, control: Any, patch: Any, original: Any) -> None:
    v1 = type(control).prepared.__globals__['V1']
    extended = v1.prepared
    def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
        if permitted(binding): return extended(self,binding,item,sm,raw,signals,view)
        return original(self,binding,item,sm,raw,signals,view)
    patch(stack,v1,'prepared',prepared)
    old_make = B.CURRENT.C.C.make
    def make(self: Any, call: Any, result: Any, provisional: Any) -> Any:
        if not permitted(call['binding']): return None
        return old_make(self,call,result,provisional)
    patch(stack,B.CURRENT.C.C,'make',make)
