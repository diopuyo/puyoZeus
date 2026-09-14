"""元sealed armのE参照だけを派生。条件付き精算への旧権限流入は拒否する。"""
from __future__ import annotations
from types import FunctionType,SimpleNamespace
from typing import Any
import origin_evidence as N
import origin_fixed as F


def conditional(origin: Any, logs: Any) -> bool:
    if origin.event_identity.startswith(N.PREFIX): return True
    return any(r['purpose']=='origin' and r['proof'].get('kind')==N.KIND
        and r['proof']['evidence']['origin_id']==origin.origin_id for r in logs)


def policy_type(accounting: Any) -> type:
    base=accounting.BoundPolicy
    values,e=F.original(base)
    def origin(p: Any, logs: Any, value: Any, state: Any, proof: Any) -> str:
        if proof.get('kind')==N.KIND: return N.origin(p,e,logs,value,state,proof)
        F.require(not value.event_identity.startswith(N.PREFIX),'legacy_origin_conditional_identity')
        return e.origin(p,logs,value,state,proof)
    def settlement(p: Any, logs: Any, value: Any, state: Any, proof: Any) -> str:
        F.require(not conditional(value.origin,logs),'conditional_settlement_not_connected')
        return e.settlement(p,logs,value,state,proof)
    facade=SimpleNamespace(**(vars(e)|dict(origin=origin,settlement=settlement)))
    arm=FunctionType(base.arm.__code__,dict(values,E=facade),base.arm.__name__,base.arm.__defaults__,base.arm.__closure__)
    arm.__kwdefaults__=base.arm.__kwdefaults__
    result=type('ConditionalOriginPolicy',(base,),{'arm':arm,'__module__':__name__})
    F.require(result.authorize is base.authorize and result._base_policy is base._base_policy,'sealed_delegate')
    return result
