"""元delayed内の条件付き登録済み分岐だけ追加し、直接update callerを守る。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType,MethodType
from typing import Any
import ast
import hashlib
import observed_connection as OBS

SOURCE=Path(__file__).resolve().parent.parent/'g2_conditional_firing_admission_2026-09-10_v1/admission.py'
HOOK='_conditional_registered_formula'
SOURCE_SHA='ec5eb6d57c621882fe8738527ffeeab37bf862e1e6725f8202b8c37e5bb01621'


def install(stack: Any,pipe: Any,patch: Any,callback: Any) -> None:
    import admission as A
    import commit as C
    import firing_fixed as F
    original=pipe._apply_chain_formula_early_fire.__func__
    values=original.__globals__; data=SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest()==SOURCE_SHA
    assert Path(A.__file__).resolve()==SOURCE and values.keys()==vars(A).keys()
    assert all(values[k] is (C.captured if k=='prepare' else v) for k,v in vars(A).items())
    assert F.code_value(original.__code__)==F.code_value(OBS.nested(
        compile(data,str(SOURCE),'exec',dont_inherit=True),('install','delayed')))
    tree=ast.parse(data,filename=str(SOURCE))
    outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    node=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='delayed')
    assert ast.unparse(node.body[0])=='binding = control.history.get(side)'
    source="if binding is not None and getattr(binding,'conditional_firing_registered',None) is not None:\n"
    source+='    return '+HOOK+'(self,side,time_sec,prev_confirmed,binding,sys._getframe())'
    branch=ast.parse(source).body[0]
    ast.copy_location(branch,node.body[1]); node.body.insert(1,branch); ast.fix_missing_locations(tree)
    code=OBS.nested(compile(tree,str(SOURCE),'exec',dont_inherit=True),('install','delayed'))
    assert code.co_freevars==original.__code__.co_freevars and HOOK not in values
    function=FunctionType(code,dict(values,**{HOOK:callback}),original.__name__,
        original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    patch(stack,pipe,'_apply_chain_formula_early_fire',MethodType(function,pipe))
