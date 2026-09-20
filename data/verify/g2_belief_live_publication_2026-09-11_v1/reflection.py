"""同Jの観測済み・未消費なし・最後の確率更新とRegistry一致を検査する。"""
from __future__ import annotations

from typing import Any
import serialization as S
import journal_context as C


def verify(mode: Any, value: Any, token: str, frame: int) -> None:
    connection, native = mode.connection, mode.native
    C.require(mode.error is None and native is not None and native.connection is connection, 'reflection_owner')
    C.require(connection.registry.current(connection.binding) is value, 'reflection_current')
    C.require(not native.pending, 'reflection_pending')
    if token == connection.binding.initial_call_token:
        C.require(value.frame == frame == mode.activation['frame'], 'reflection_initial_frame')
        return
    C.require(native.last_frame == frame and token in native.seen_calls, 'reflection_J_not_observed')
    C.require(bool(mode.applied), 'reflection_transition_evidence_missing')
    last = mode.applied[-1]
    C.require(last['applied_frame'] <= frame and last['applied_frame'] == value.frame, 'reflection_frame')
    C.require(S.decode(last['state']) == value, 'reflection_state_mismatch')
