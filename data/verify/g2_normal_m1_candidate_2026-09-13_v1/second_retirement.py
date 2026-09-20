"""正当な世代進行で旧2Pを退役する。原J成功前にはRegistryを変えない。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any
import journal_context as C
import serialization as S
import native_consumption as N


def prepare(mode: Any,item: Any) -> dict[str,Any] | None:
    c,r=mode.connection,mode.connection.recovery
    current=c.registry.current(c.binding)
    following=mode.scope(r.factory,r.pipe)
    if following==current.scope: return None
    old=current.scope
    C.require(all(following[i]==old[i] for i in (0,1,3,4,6)),'second_reset_identity')
    C.require(all(type(following[i]) is int and following[i]>=old[i] for i in (2,5))
              and any(following[i]>old[i] for i in (2,5)),'second_reset_regression')
    frame=item['scope']['frame_idx']
    C.require(item['frame'] is not None and item['frame'].f_code in r.journal.codes
        and item['pipe'] is r.pipe and item['scope']['side']==mode.side and r.journal.active is None,'second_reset_actual_J')
    C.require(frame==mode.native.last_frame+N.FRAME_STRIDE and frame<=current.deadline,'second_reset_frame')
    boundary=N.extract(item)
    return dict(frame=frame,source_call_token=item['token'],start_epoch=item['epoch'],
        code_sha256=hashlib.sha256(item['frame'].f_code.co_code).hexdigest(),
        old_state=S.encode(current),new_scope=list(following),
        pending=[asdict(value) for value in mode.native.pending],quality_gate_clear=False,
        reset_call_consumption=None if boundary is None else asdict(boundary),
        reset_call_attribution='UNATTRIBUTED',reset_call_accounting_permission=False)


def commit(mode: Any,packet: Any) -> dict[str,Any]:
    c,r=mode.connection,mode.connection.recovery
    step=mode.witness.pair(packet['frame'])[0 if mode.side == '1P' else 1]
    C.require(step['status']=='returned' and step['exception'] is None
        and step['token']==packet['source_call_token'] and step['code_sha256']==packet['code_sha256']
        and step['software_reset']==packet['start_epoch'],'second_reset_completed_J')
    following=mode.scope(r.factory,r.pipe)
    C.require(list(following)==packet['new_scope'] and step['generation_after']['reset_epoch']==following[5]
        and step['source_id']==following[0] and step['run_id']==following[1]
        and step['pipe_object_id']==following[3],'second_reset_current_scope')
    value=c.registry.current(c.binding)
    C.require(S.encode(value)==packet['old_state']
        and [asdict(v) for v in mode.native.pending]==packet['pending'],'second_reset_old_state_changed')
    c.registry.retire(r.factory,c.binding)
    mode.retired_receipt=packet|dict(reason='generation_changed',retired=True,pending_discarded=False)
    mode.closed=True
    mode.evidence.registered_scope=None
    return mode.retired_receipt
