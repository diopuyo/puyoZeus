"""実逐次保存済みmetadataを次update入口の前提にする。"""
from __future__ import annotations
from types import FunctionType, SimpleNamespace as N
import entry_boundary_v2 as V
import metadata_tail as M


class Entry(V.E.Entry):
    before = FunctionType(V.E.Entry.before.__code__, dict(vars(V.E), completed_metadata=M.completed))


install = FunctionType(V.install.__code__, dict(vars(V), E=N(**(vars(V.E)|dict(Entry=Entry)))))
