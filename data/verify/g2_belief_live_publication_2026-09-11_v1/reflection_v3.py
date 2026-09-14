"""未変化2Pの初回確率を、実登録票と連続原Jが保たれる間だけ使用する。"""
from __future__ import annotations

from typing import Any
import reflection as OLD
import second_physical as P
import journal_context as C
import serialization as S


def verify(mode: Any,value: Any,token: str,frame: int) -> None:
    C.require(not getattr(mode,'closed',False) and getattr(mode,'retired_receipt',None) is None,'reflection_closed')
    if mode.applied or token==mode.connection.binding.initial_call_token:
        OLD.verify(mode,value,token,frame)
        return
    C.require(type(mode) is P.Mode and mode.error is None,'initial_reflection_mode')
    c,native=mode.connection,mode.native
    C.require(native.connection is c and c.registry.current(c.binding) is value,'initial_reflection_current')
    C.require(not native.pending and not native.seen_occurrences,'initial_reflection_unapplied_hand')
    C.require(native.last_frame==frame and token in native.seen_calls,'initial_reflection_J')
    physical=mode.physical
    C.require(physical.connection is c and physical.native is native and physical.applied is mode.applied
        and not physical.origins and physical.basis_origin is None,'initial_reflection_origin')
    receipt=mode.initial_receipt
    C.require(receipt['source_call_token']==c.binding.initial_call_token
        and mode.activation['source_call_token']==c.binding.initial_call_token,'initial_reflection_token')
    C.require(value.frame==mode.activation['frame'] and S.decode(receipt['state'])==value,'initial_reflection_state')

