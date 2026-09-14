"""外精算派生から元originV2・sealed armまで全帰属を辿る。"""
from __future__ import annotations
from types import FunctionType,SimpleNamespace
from typing import Any
import compat_fixed as K

CLASS_KEYS=frozenset(('__module__','arm','__doc__'))


def modules(fixed: Any) -> tuple[Any,Any]:
    import conditional_settlement as C
    import settlement_provenance as V
    K.guards(); K.module(C,K.SETTLEMENT/'conditional_settlement.py')
    K.module(V,K.SETTLEMENT/'settlement_provenance.py')
    for name in ('policy_type','registered','observation','validate'):
        K.function(getattr(C,name),C,(name,),fixed)
    for name in ('same_globals','validator','verify'):
        K.function(getattr(V,name),V,(name,),fixed)
    K.require((C.KIND,C.ORIGIN_KIND,C.OBS_KIND,C.FPS,C.STRIDE,C.OPERATIONS,C.ALLOWED_STATES)==
        ('conditional_firing_settlement/v1','conditional_firing_origin/v1','conditional_clear_raw/v1',
         60,2,4,frozenset(('tsumo_fall','stable'))),'settlement_constants')
    return C,V


def validator(value: Any, module: Any, prior: Any, fixed: Any) -> None:
    K.function(value,module,('policy_type','settlement'),fixed)
    K.require(value.__code__.co_freevars==('prior',) and type(value.__closure__) is tuple
        and len(value.__closure__)==1 and value.__closure__[0].cell_contents is prior,'validator_closure')
    K.require(value.__defaults__ is None and value.__kwdefaults__ is None,'validator_defaults')


def verify(policy: Any, fixed: Any) -> Any:
    C,V=modules(fixed); cls=type(policy)
    K.require(type(cls) is type and len(cls.__bases__)==1 and cls.__name__=='ConditionalSettlementPolicy'
        and cls.__module__==C.__name__ and set(vars(cls))==CLASS_KEYS,'accounting_class')
    base=cls.__bases__[0]; prior=V.verify(base)
    K.require(all(getattr(cls,key) is getattr(base,key) for key in
        ('__init__','authorize','_base_policy','_inventory')),'accounting_inheritance')
    arm=cls.arm; original=base.arm
    K.require(type(arm) is FunctionType and arm.__code__ is original.__code__
        and arm.__closure__ is original.__closure__ and arm.__defaults__ is original.__defaults__
        and arm.__kwdefaults__ is original.__kwdefaults__,'accounting_arm_code')
    facade=arm.__globals__.get('E')
    K.require(type(facade) is SimpleNamespace and vars(facade).keys()==vars(prior).keys(),'facade_shape')
    V.same_globals(arm.__globals__,original.__globals__,facade)
    K.require(all(vars(facade)[key] is value for key,value in vars(prior).items()
        if key!='settlement'),'facade_values')
    validator(facade.settlement,C,prior,fixed)
    K.require(policy.arm.__func__ is arm and policy.arm.__self__ is policy,'accounting_bound_arm')
    K.require(policy.authorize.__func__ is base.authorize and policy.authorize.__self__ is policy,
        'accounting_bound_authorize')
    return arm

