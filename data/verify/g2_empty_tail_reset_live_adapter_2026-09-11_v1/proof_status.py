"""復帰開始の成功と、後続baseline待ちの失敗を同じ保存票で区別する。"""
from __future__ import annotations
import json
from typing import Any


def recovery_status(recovery: Any) -> Any:
    if recovery is None: return None
    return dict(error=recovery.error,failure=None if recovery.failure is None else repr(recovery.failure),
        reset_count=recovery.reset_count,baseline_count=recovery.baseline_count,
        wait_count=recovery.wait_count,pending=recovery.pending)


def close(context: Any) -> None:
    state = context.state
    arming = state.get('live_empty_arming')
    armed = None if arming is None else dict(events=arming.rows,
        entry=None if arming.entry is None else arming.entry.rows,
        error=None if arming.error is None else repr(arming.error))
    recovery = recovery_status(context.recovery)
    guard = state.get('repeat_scope_guard')
    lease = None if guard is None else guard.reset_lease
    leased = None if lease is None else dict(events=lease.events,retirement=lease.retirement,
        used=lease.used,active=lease.active,waiting=lease.waiting)
    errors = [context.error,None if recovery is None else recovery['error'],
              None if recovery is None else recovery['failure'],None if armed is None else armed['error']]
    value = dict(events=context.rows,error=next((e for e in errors if e is not None),None),
        perform_error=context.error,quality_gate_clear=False,arming=armed,
        recovery=None if context.recovery is None else context.recovery.rows,
        recovery_status=recovery,lease=leased,raw=state.get('live_empty_raw_count'),
        baseline_recovered=False if recovery is None else recovery['baseline_count']==1 and recovery['pending'] is None)
    sink = state.get('live_history_sink')
    if sink is not None: value = sink.base.json_value(value)
    with (state['output']/'LIVE_EMPTY_RESET.json').open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def derived(original: Any) -> Any:
    class Context(original):
        def close(self) -> None:
            close(self)
    return Context
