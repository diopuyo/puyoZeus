"""原observedの直接callerを維持し、返却直前だけ条件付き消去票を採る。"""
from __future__ import annotations
from pathlib import Path
from types import CodeType,FunctionType
from typing import Any
import ast
import hashlib
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'g2_firing_hand_connection_2026-09-10_v1/continuation_v1/settlement.py'
HOOK='_conditional_observation_after_original'
SOURCE_SHA='b0d4ad20715d3b50c038c2d2e656471966f874d4ea810149ab14c03fe64e5b2a'


def nested(code: Any, names: tuple[str,...]) -> Any:
    for name in names:
        choices=[c for c in code.co_consts if type(c) is CodeType and c.co_name==name]
        assert len(choices)==1
        code=choices[0]
    return code


def compile_observed(original: Any, callback: Any) -> Any:
    import firing_fixed as F
    values=original.__globals__; source=SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest()==SOURCE_SHA
    assert Path(values['__file__']).resolve()==SOURCE
    assert values is vars(sys.modules[values['__name__']])
    old=compile(source,str(SOURCE),'exec',dont_inherit=True)
    assert F.code_value(original.__code__)==F.code_value(nested(old,('install','observed')))
    tree=ast.parse(source,filename=str(SOURCE))
    install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    node=next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name=='observed')
    assert isinstance(node.body[-1],ast.Return) and ast.unparse(node.body[-1].value)=='raw'
    hook=ast.parse(HOOK+'(self,binding,view,signals,sm,pipe)').body[0]
    ast.copy_location(hook,node.body[-1]); node.body.insert(-1,hook)
    ast.fix_missing_locations(tree)
    modified=nested(compile(tree,str(SOURCE),'exec',dont_inherit=True),('install','observed'))
    assert modified.co_freevars==original.__code__.co_freevars and HOOK not in values
    function=FunctionType(modified,dict(values,**{HOOK:callback}),original.__name__,
        original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    assert function.__closure__ is original.__closure__
    return function


def install(stack: Any, factory: Any, patch: Any, callback: Any, rows: list[Any]) -> None:
    cls=type(factory.controller); original=cls.observed
    def after(control: Any, binding: Any, view: Any, signals: Any, sm: Any, pipe: Any) -> None:
        frame=sys._getframe(1)
        assert frame.f_code is function.__code__ and frame.f_globals is function.__globals__
        assert frame.f_locals['self'] is control is factory.controller
        assert frame.f_locals['signals'] is signals and frame.f_locals['pipe'] is pipe
        if binding is not None and getattr(binding,'conditional_firing_registered',None) is not None:
            callback(control,binding,view,signals,sm,pipe,rows)
    function=compile_observed(original,after)
    patch(stack,cls,'observed',function)
