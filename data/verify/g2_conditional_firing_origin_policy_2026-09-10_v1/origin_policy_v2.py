"""条件付き配置を旧起源kindへ偽装する交差経路も入口で拒否する。"""
from __future__ import annotations
from types import FunctionType,SimpleNamespace
from typing import Any
import origin_evidence as N
import origin_fixed as F
import origin_policy as OLD


def conditional_placement(logs: Any,proof: Any) -> bool:
    return any(row['purpose']=='placement' and row['proof_sha']==proof.get('placement_event_id')
        and row['proof'].get('kind')==N.PLACEMENT for row in logs)


def policy_type(accounting: Any) -> type:
    base=accounting.BoundPolicy
    values,e=F.original(base)
    def origin(p: Any,logs: Any,value: Any,state: Any,proof: Any) -> str:
        if proof.get('kind')==N.KIND: return N.origin(p,e,logs,value,state,proof)
        F.require(not value.event_identity.startswith(N.PREFIX),'legacy_origin_conditional_identity')
        F.require(not conditional_placement(logs,proof),'legacy_origin_conditional_placement')
        return e.origin(p,logs,value,state,proof)
    def settlement(p: Any,logs: Any,value: Any,state: Any,proof: Any) -> str:
        F.require(not OLD.conditional(value.origin,logs),'conditional_settlement_not_connected')
        return e.settlement(p,logs,value,state,proof)
    facade=SimpleNamespace(**(vars(e)|dict(origin=origin,settlement=settlement)))
    arm=FunctionType(base.arm.__code__,dict(values,E=facade),base.arm.__name__,base.arm.__defaults__,base.arm.__closure__)
    arm.__kwdefaults__=base.arm.__kwdefaults__
    result=type('ConditionalOriginPolicyV2',(base,),{'arm':arm,'__module__':__name__})
    F.require(result.authorize is base.authorize and result._base_policy is base._base_policy,'sealed_delegate')
    return result
