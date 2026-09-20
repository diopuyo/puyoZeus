"""自己申告module名でなく元compiled codeと実globalsでSinkを認証。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from types import CodeType, FunctionType
from typing import Any
import checked_tail as C

OBSERVER_SHA = '98b68c0f6a523bc1c756f359e02fc3f345a4bfcc234ea0ba9262610daa9b7e1a'


def compiled(module: Any, expected: str) -> Any:
    path = Path(module.__file__)
    data = path.read_bytes()
    C.G.require(hashlib.sha256(data).hexdigest()==expected,'metadata_original_source')
    return compile(data,str(path),'exec',dont_inherit=True)


def nested(code: Any, name: str) -> Any:
    values = [c for c in code.co_consts if isinstance(c,CodeType) and c.co_name==name]
    C.G.require(len(values)==1,'metadata_compiled_member')
    return values[0]


def authenticate(sink: Any) -> None:
    module = sys.modules[type(sink).__module__]
    code = compiled(module,C.BOUNDED_SHA)
    for name,methods in (('Sink',('__init__','close')),('RowStream',('__init__','append'))):
        cls = getattr(module,name)
        for method_name in methods:
            actual = getattr(cls,method_name)
            C.G.require(hasattr(actual,'__code__') and actual.__code__==nested(nested(code,name),method_name)
                        and actual.__globals__ is vars(module),'metadata_actual_code')
    observer = module.O
    code = compiled(observer,OBSERVER_SHA)
    C.G.require(module.Sink.__bases__==(observer.Sink,),'metadata_original_base')
    for name in ('__init__','invoke','wrapper'):
        actual = getattr(observer.Sink,name)
        C.G.require(actual.__code__==nested(nested(code,'Sink'),name)
                    and actual.__globals__ is vars(observer),'metadata_observer_code')


class Tail(C.Tail):
    def __init__(self, sink: Any) -> None:
        authenticate(sink)
        super().__init__(sink)


def install(stack: Any, state: Any) -> Tail:
    original = FunctionType(C.M.install.__code__,dict(vars(C.M),Tail=Tail))
    tail = original(stack,state)
    tail.callback = tail.stream.append
    return tail
