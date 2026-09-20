"""原update完了後に2P登録・両側評価・専用保存を行い、同stackで解放する。"""
from __future__ import annotations

from functools import wraps
import json
from pathlib import Path
from typing import Any
import journal_context as C
import journal_witness as W
import second_observation as O
import second_basis as B
import second_physical as P
import second_tracking as T
import live_binding_v3 as V
import trained_sidecar as SAVE
import serialization as S


class Session:
    def __init__(self,stack: Any,context: Any,policy: Any,physical: Any,contract: Any,
                 members: Any,frames: tuple[int,...]) -> None:
        self.stack,self.context=stack,context
        self.policy,self.physical,self.contract,self.members=policy,physical,contract,members
        self.frames=frames
        self.state,self.factory,self.pipe=(context[k] for k in ('state','factory','pipe'))
        self.journal=self.factory.provider.journal
        self.mode: Any=None
        self.modes: list[Any]=[]
        self.saved: list[Any]=[]
        self.holds: list[Any]=[]
        self.error: Any=None
        self.restored=False
        stack.push(self.close)
        self.witness=W.install(stack,self.journal)
        self.evidence=O.install(stack,self.journal,self.state)

    def basis(self) -> None:
        if self.mode is not None and not self.mode.closed: return
        first=self.state['probabilistic_basis_connection']
        if first.binding is None: return
        try:
            mode=P.Mode(self.evidence,self.witness,self.pipe,first.registry,self.factory,
                first.registry.current(first.binding).deadline,self.policy,self.physical,self.factory.provider)
        except B.BasisHold as hold:
            self.holds.append(dict(frame=self.evidence.latest['frame'],reason=str(hold)))
            return
        T.install(self.stack,mode)
        self.mode=mode
        self.modes.append(mode)

    def capture(self) -> Any:
        first=self.state['probabilistic_basis_connection']
        modes=(self.state['probabilistic_tracking_mode'],self.mode)
        return V.current(self.state['provisional_context_observer'],self.factory,self.pipe,
            self.journal,first.registry,(first.binding,self.mode.connection.binding),
            self.contract,self.witness,modes)

    def completed(self,frame: int) -> None:
        rec=self.state['provisional_context_observer']
        C.require(rec.active is None and not rec.errors and rec.rows
            and rec.rows[-1]['frame_idx']==frame,'session_after_original_update')
        self.basis()
        if frame not in self.frames: return
        C.require(self.mode is not None and not self.mode.closed,'session_selected_basis_missing')
        C.require(not any(row['frame']==frame for row in self.saved),'session_duplicate_evaluation')
        path=self.state['output']/f'BELIEF_M1_{frame}.json'
        self.saved.append(SAVE.run(self.capture,self.members,path,seed=frame))

    def close(self,kind: Any,body: Any,trace: Any) -> bool:
        try:
            packets=[dict(initial=m.initial_receipt,retired=m.retired_receipt,applied=m.applied,
                current=None if m.retired_receipt is not None else S.encode(m.connection.registry.current(m.connection.binding)),
                closed=m.closed,error=None if m.error is None else repr(m.error)) for m in self.modes]
            packet=dict(saved=self.saved,holds=self.holds,modes=packets,restored=self.restored,
                observer_closed=self.evidence.closed,witness_closed=self.witness.closed,
                error=None if body is None else repr(body),
                session_error=None if self.error is None else repr(self.error),
                observer_error=None if self.evidence.error is None else repr(self.evidence.error),
                witness_error=None if self.witness.error is None else repr(self.witness.error),quality_gate_clear=False)
            write(self.state['output']/'BELIEF_M1_SESSION.json',packet)
            if body is None:
                C.require(self.error is None and self.evidence.error is None and self.witness.error is None
                    and all(m.error is None for m in self.modes),'session_prior_error')
                C.require(self.restored and self.evidence.closed and self.witness.closed,'session_cleanup')
                C.require(tuple(r['frame'] for r in self.saved)==self.frames,'session_evaluation_coverage')
        except BaseException as caught:
            self.error=caught
            if body is None: raise
        return False


def write(path: Path,packet: Any) -> None:
    raw=json.dumps(packet,ensure_ascii=False,allow_nan=False,sort_keys=True).encode()
    with path.open('xb') as stream: stream.write(raw)
    C.require(path.read_bytes()==raw,'session_saved_bytes')


def attach(stack: Any,value: Session) -> None:
    cls=type(value.pipe)
    original=cls.update
    @wraps(original)
    def update(pipe: Any,frame_idx: int,time_sec: float,*args: Any,**kwargs: Any) -> Any:
        result=original(pipe,frame_idx,time_sec,*args,**kwargs)
        if pipe is value.pipe:
            try: value.completed(frame_idx)
            except BaseException as error:
                value.error=error
                raise
        return result
    def restore(kind: Any,body: Any,trace: Any) -> bool:
        try:
            C.require(cls.update is update,'session_foreign_update')
            cls.update=original
            value.restored=cls.update is original
        except BaseException as error:
            value.error=error
            if body is None: raise
        return False
    stack.push(restore)
    cls.update=update
