"""実保存で確認したprovider/adoption中継を、各原codeとclosureで認証する。"""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import CodeType
from typing import Any
import writer_contract as OLD

C=OLD.C
SOURCE,SOURCE_SHA=OLD.SOURCE,OLD.SOURCE_SHA
PROVIDER_SHA='5d0b400d965700943887b2bb833393689c84a46aa2a5d302e1ff7f6e600dca6f'
MAX_WRAPPERS=2
ORDERS=((),('adoption',),('provider',),('adoption','provider'))
canonical=OLD.canonical


def checked(emit: Any,journal: Any) -> tuple[str,Any]:
    C.require(inspect.isfunction(emit),'J_writer_wrapper_type')
    path=Path(emit.__code__.co_filename).resolve()
    raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest in OLD.ADOPTION_SHAS: kind,names='adoption',('bind','attached','captured')
    elif digest==PROVIDER_SHA: kind,names='provider',('Provider','attach','captured')
    else: raise ValueError('belief_publication:J_writer_wrapper_source')
    expected=compile(raw,str(path),'exec')
    for name in names:
        candidates=[c for c in expected.co_consts if isinstance(c,CodeType) and c.co_name==name]
        C.require(len(candidates)==1,'J_writer_wrapper_definition')
        expected=candidates[0]
    C.require(all(getattr(emit.__code__,k)==getattr(expected,k) for k in
        ('co_code','co_consts','co_names','co_freevars')),'J_writer_wrapper_code')
    cells=dict(zip(emit.__code__.co_freevars,(c.cell_contents for c in emit.__closure__),strict=True))
    if kind=='adoption':
        C.require(cells['witness'].provider.journal is journal and emit.__globals__['sys'] is sys,'J_writer_adoption_owner')
    else:
        C.require(cells['self'].journal is journal and emit.__globals__['copy'] is copy,'J_writer_provider_owner')
    return kind,cells['emit']


def chain(journal: Any) -> tuple[Any,...]:
    emit=journal.emit
    wrappers,kinds=[],[]
    while inspect.isfunction(emit):
        C.require(len(wrappers)<MAX_WRAPPERS,'J_writer_wrapper_depth')
        kind,inner=checked(emit,journal)
        wrappers.append((emit,inner))
        kinds.append(kind)
        emit=inner
    C.require(tuple(kinds) in ORDERS,'J_writer_wrapper_order')
    C.require(inspect.ismethod(emit) and emit.__self__ is journal
        and emit.__func__ is type(journal).emit,'J_writer_terminal_owner')
    return tuple(wrappers)


class Contract:
    def __init__(self,journal: Any) -> None:
        C.require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'J_witness_source')
        self.journal=journal
        self.code=type(journal).complete_step.__code__
        self.emit_code=type(journal).emit.__code__
        C.require(all(Path(c.co_filename).resolve()==SOURCE for c in (self.code,self.emit_code)),'J_writer_code_source')
        self.wrappers=chain(journal)
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
        for wrapper,inner in reversed(self.wrappers):
            C.require(parent.f_code is wrapper.__code__ and parent.f_globals is wrapper.__globals__
                and parent.f_locals['row'] is caller.f_locals['value']
                and parent.f_locals['emit'] is inner,'J_writer_middleware_caller')
            parent=parent.f_back
        C.require(parent.f_code is self.code and parent.f_globals is type(journal).complete_step.__globals__
            and parent.f_locals.get('self') is journal,'J_original_caller')
        item=parent.f_locals['item']
        C.require(row['token']==item['token'] and row['side']==item['scope']['side'],'J_original_item')
        return row
