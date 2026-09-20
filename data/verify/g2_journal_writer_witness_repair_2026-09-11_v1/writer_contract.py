"""固定原Recorderと既存adoption中継だけを認証する。元の呼出鎖は変更しない。"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import CodeType
from typing import Any
import journal_context as C

SOURCE=Path(__file__).resolve().parent.parent/'g2_atomic_journal_capture_2026-09-09_v1/observer.py'
SOURCE_SHA='b0d956a4fef51846a7baa2cee95a3679887bd2e3e9e620462e8375dc238d830c'
ADOPTION_SHAS=frozenset(('33973e6555b8fda610e300ab392624297ee87d589688fe467976f22241608a0a',
    '9ec086d887433c23a783e36d91306edbdc43d5e53a8539520208c765e17eddd1'))


def canonical(value: Any) -> str:
    return json.dumps(value,allow_nan=False,sort_keys=True)


def intermediary(journal: Any) -> Any:
    emit=journal.emit
    if inspect.ismethod(emit):
        C.require(emit.__self__ is journal and emit.__func__ is type(journal).emit,'J_writer_original_method')
        return None
    C.require(inspect.isfunction(emit),'J_writer_wrapper_type')
    path=Path(emit.__code__.co_filename).resolve()
    raw=path.read_bytes()
    C.require(hashlib.sha256(raw).hexdigest() in ADOPTION_SHAS,'J_writer_wrapper_source')
    expected=compile(raw,str(path),'exec')
    for name in ('bind','attached','captured'):
        expected=next(c for c in expected.co_consts if isinstance(c,CodeType) and c.co_name==name)
    C.require(emit.__code__.co_code==expected.co_code and emit.__code__.co_consts==expected.co_consts
        and emit.__code__.co_names==expected.co_names and emit.__code__.co_freevars==expected.co_freevars,'J_writer_wrapper_code')
    cells=dict(zip(emit.__code__.co_freevars,(c.cell_contents for c in emit.__closure__),strict=True))
    original=cells['emit']
    C.require(inspect.ismethod(original) and original.__self__ is journal
        and original.__func__ is type(journal).emit and emit.__globals__['sys'] is sys,'J_writer_wrapper_owner')
    return emit


class Contract:
    def __init__(self,journal: Any) -> None:
        C.require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'J_witness_source')
        self.journal=journal
        self.code=type(journal).complete_step.__code__
        self.emit_code=type(journal).emit.__code__
        C.require(all(Path(c.co_filename).resolve()==SOURCE for c in (self.code,self.emit_code)),'J_writer_code_source')
        self.outer=intermediary(journal)
        self.emit=journal.emit

    def row(self,text: str,caller: Any) -> dict[str,Any]:
        journal=self.journal
        C.require(journal.emit==self.emit and caller.f_code is self.emit_code
            and caller.f_globals is type(journal).emit.__globals__
            and caller.f_locals.get('self') is journal,'J_writer_original_caller')
        row=json.loads(text)
        C.require(type(row) is dict and canonical(row)==canonical(caller.f_locals['row']),'J_writer_encoded_row')
        C.require(type(row['row_index']) is int and type(journal.count) is int
            and row['row_index']==journal.count,'J_writer_preincrement')
        if row.get('kind')!='step': return row
        parent=caller.f_back
        if self.outer is not None:
            C.require(parent.f_code is self.outer.__code__ and parent.f_globals is self.outer.__globals__
                and parent.f_locals['row'] is caller.f_locals['value'],'J_writer_adoption_caller')
            parent=parent.f_back
        C.require(parent.f_code is self.code and parent.f_globals is type(journal).complete_step.__globals__
            and parent.f_locals.get('self') is journal,'J_original_caller')
        item=parent.f_locals['item']
        C.require(row['token']==item['token'] and row['side']==item['scope']['side'],'J_original_item')
        return row
