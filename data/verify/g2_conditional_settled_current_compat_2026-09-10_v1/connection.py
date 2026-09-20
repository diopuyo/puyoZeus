"""元verifyのfunction判定だけ限定派生し、元recoverの同bodyへ渡す。"""
from __future__ import annotations
from types import FunctionType,SimpleNamespace
from typing import Any
import inspect
import sys
import compat_fixed as K
import compat_policy as P
import compat_scope as S


def verify(original: Any, inventory: Any, binding: Any) -> None:
    K.guards(); K.module(original,K.REVISION/'settled_fixed.py')
    for name in ('verify','function','guards','require'):
        K.function(getattr(original,name),original,(name,),original)
    S.ordinary(binding)
    accounting=binding.policy.accounting; arm=accounting.arm.__func__
    def function(value: Any, source: Any, names: Any) -> None:
        if value is arm and source==original.FIRING/'firing_policy.py' and names==('FiringMethods','arm'):
            if value.__globals__ is vars(sys.modules[value.__module__]):
                return original.function(value,source,names)
            K.require(P.verify(accounting,original) is value,'accounting_arm_identity')
            return None
        return original.function(value,source,names)
    made=FunctionType(original.verify.__code__,dict(vars(original),function=function),
        original.verify.__name__,original.verify.__defaults__,original.verify.__closure__)
    made.__kwdefaults__=original.verify.__kwdefaults__
    made(inventory,binding)


def derive(original: Any) -> Any:
    """原関数を渡す。呼出元のpatch/ExitStackで設置し、共有moduleは変更しない。"""
    K.guards()
    K.require(inspect.isfunction(original),'recover_function')
    module=sys.modules.get(original.__module__)
    K.require(module is not None,'recover_module')
    K.module(module,K.REVISION/'recover.py'); fixed=original.__globals__['F']
    K.function(original,module,('recover_settled_current',),fixed)
    K.module(original.__globals__['E'],K.REVISION/'settled_evidence.py')
    def validate(inventory: Any,binding: Any) -> None: verify(fixed,inventory,binding)
    facade=SimpleNamespace(**(vars(fixed)|dict(verify=validate)))
    function=FunctionType(original.__code__,dict(original.__globals__,F=facade),original.__name__,
        original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    return function

