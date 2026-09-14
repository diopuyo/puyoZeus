"""元writerへ常に転送し、成功した原J行だけを追加観測する。"""
from __future__ import annotations
import io
import json
from pathlib import Path
import sys
from typing import Any
import writer_contract as K


class Witness:
    def __init__(self,journal: Any) -> None:
        self.contract=K.Contract(journal)
        self.journal,self.code=journal,self.contract.code
        self.original=journal.stream
        K.C.require(isinstance(self.original,io.TextIOBase) and not self.original.closed,'J_original_stream')
        if not isinstance(self.original,io.StringIO):
            K.C.require(Path(self.original.name).name=='atomic_journal.jsonl','J_original_stream_path')
        K.C.require(type(journal.count) is int and journal.count>=0,'J_initial_count')
        self.last_index=journal.count-1
        self.rows: dict[str,str]={}
        self.error: Any=None
        self.closed=False

    def written(self,text: str,caller: Any) -> None:
        if self.closed or self.error is not None: return
        try:
            row=self.contract.row(text,caller)
            K.C.require(row['row_index']==self.last_index+1,'J_writer_sequence')
            self.last_index=row['row_index']
            if row.get('kind')=='step': self.rows[row['side']]=K.canonical(row)
        except BaseException as error: self.error=error

    def pair(self,frame: int) -> list[dict[str,Any]]:
        K.C.require(not self.closed and self.error is None and set(self.rows)==set(K.C.SIDES),'J_witness_pair')
        K.C.require(self.journal.count==self.last_index+1,'J_writer_final_increment')
        rows=[json.loads(self.rows[side]) for side in K.C.SIDES]
        K.C.require(rows[0]['row_index']<rows[1]['row_index'],'J_writer_side_order')
        K.C.require(all(row['frame_idx']==frame for row in rows),'J_witness_frame')
        return rows


class Stream:
    def __init__(self,witness: Witness) -> None:
        self.witness=witness

    def write(self,text: str) -> Any:
        value=self.witness
        try: result=value.original.write(text)
        except BaseException as error:
            value.error=error
            raise
        if type(result) is not int or result!=len(text):
            value.error=value.error or ValueError('belief_publication:J_writer_short_write')
        value.written(text,sys._getframe(1))
        return result

    def flush(self) -> Any: return self.witness.original.flush()

    def close(self) -> Any: return self.witness.original.close()

    @property
    def closed(self) -> bool: return self.witness.original.closed


def install(stack: Any,journal: Any) -> Witness:
    value=Witness(journal)
    proxy=Stream(value)
    def close(kind: Any,body: Any,trace: Any) -> bool:
        try:
            K.C.require(journal.stream is proxy,'J_witness_foreign_stream')
            journal.stream=value.original
            if body is None: K.C.require(value.error is None,'J_writer_prior_error')
        except BaseException as error:
            value.error=value.error or error
            if body is None: raise
        finally: value.closed=True
        return False
    stack.push(close)
    journal.stream=proxy
    return value
