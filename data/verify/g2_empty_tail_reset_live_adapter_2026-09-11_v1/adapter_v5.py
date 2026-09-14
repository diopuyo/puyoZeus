"""原v4配線を維持し、独立指摘を閉鎖した最終検査だけ選ぶ。"""
from __future__ import annotations
import ast
import inspect
from types import FunctionType
import adapter_v4 as A

ROOT = A.ROOT
tree = ast.parse(inspect.getsource(A.finalize))
found = [n for n in ast.walk(tree) if isinstance(n,ast.Import)
         and len(n.names)==1 and n.names[0].name=='live_finish']
assert len(found)==1
found[0].names[0].name = 'live_finish_v2'
namespace = dict(vars(A))
exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
configured = FunctionType(A.configured.__code__,dict(vars(A),finalize=namespace['finalize']))
