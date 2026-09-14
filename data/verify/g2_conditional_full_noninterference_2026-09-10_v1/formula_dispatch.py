"""元handoffの同一frame内で通常・条件付き分岐を選び、直接callerを維持する。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType,MethodType
from typing import Any
import ast
import hashlib
import inspect
import importlib.util
import observed_connection as OBS
import formula_connection as FC
import admission as A

FIXED=Path(__file__).resolve().parent.parent/'g2_firing_policy_2026-09-10_v1/firing_fixed.py'
SPEC=importlib.util.spec_from_file_location('_joined_formula_fixed',FIXED)
F=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)

SOURCE=Path(__file__).resolve().parent.parent/'g2_firing_ticket_handoff_2026-09-10_v1/handoff.py'
SOURCE_SHA='963427b24b1c61a6f965b30624588f14d210ce8ddb5e0b3f0edb7037ba7ac061'
REGISTERED='_joined_conditional_registered'
ADMISSION='_joined_conditional_admission'
ROWS='_joined_conditional_rows'
VOTES='conditional_firing_admission_votes'


def transformed(original: Any) -> Any:
    data=SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest()==SOURCE_SHA
    declared=OBS.nested(compile(data,str(SOURCE),'exec',dont_inherit=True),('install','formula'))
    assert F.code_value(original.__code__)==F.code_value(declared),'joined_original_formula_code'
    assert Path(original.__globals__['__file__']).resolve()==SOURCE
    tree=ast.parse(data,filename=str(SOURCE))
    outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    node=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='formula')
    assert ast.unparse(node.body[1])=='binding = factory.controller.history[side]'
    source="if getattr(binding,'conditional_firing_registered',None) is not None:\n"
    source+='    return '+REGISTERED+'(self,side,time_sec,prev_confirmed,binding,sys._getframe())\n'
    source+='if getattr(binding,'+repr(VOTES)+',None) is not None:\n'
    source+='    return '+ADMISSION+'(factory,self,sys._getframe(),binding,'+ROWS+')'
    branches=ast.parse(source).body
    for branch in branches: ast.copy_location(branch,node.body[2])
    node.body[2:2]=branches; ast.fix_missing_locations(tree)
    code=OBS.nested(compile(tree,str(SOURCE),'exec',dont_inherit=True),('install','formula'))
    assert code.co_freevars==original.__code__.co_freevars
    return code


def install(stack: Any,factory: Any,pipe: Any,patch: Any,original: Any,rows: Any) -> None:
    delayed=pipe._apply_chain_formula_early_fire.__func__
    closure=inspect.getclosurevars(delayed).nonlocals
    assert closure['old_formula'] is original and closure['factory'] is factory and closure['rows'] is rows
    assert closure['control'] is factory.controller
    callback=delayed.__globals__[FC.HOOK]
    old=original.__func__; code=transformed(old)
    names={REGISTERED:callback,ADMISSION:A.formula,ROWS:rows}
    assert not set(names)&old.__globals__.keys()
    function=FunctionType(code,dict(old.__globals__,**names),old.__name__,old.__defaults__,old.__closure__)
    function.__kwdefaults__=old.__kwdefaults__
    assert function.__closure__ is old.__closure__
    patch(stack,pipe,'_apply_chain_formula_early_fire',MethodType(function,pipe))
