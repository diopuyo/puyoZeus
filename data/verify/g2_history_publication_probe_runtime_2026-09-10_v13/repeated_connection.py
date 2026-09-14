"""実Factory/constructorへ検収済み反復連鎖候補を接続する。"""
from __future__ import annotations
import sys
from typing import Any
import common as K

CANDIDATE = K.VERIFY/'g2_repeated_firing_runtime_candidate_2026-09-10_v1'
STOP = K.VERIFY/'g2_same_scope_stop_guard_2026-09-10_v1'


def load(stack: Any) -> tuple[Any, Any]:
    candidate = K.load('_live_repeated_connection', CANDIDATE/'connection.py', stack)
    hook = K.load('_live_repeated_constructor', CANDIDATE/'constructor_v1/hook.py', stack)
    sys.path.insert(0, str(STOP))
    guard = K.load('_live_scope_stop', STOP/'scope_stop.py', stack)
    adapter = K.load('_live_scope_candidate', CANDIDATE/'scope_composition_v1/adapter.py', stack)
    return adapter.adapted(candidate, guard), hook


def install(stack: Any, collector: Any, state: Any, env: Any) -> None:
    candidate, hook = env['repeated_dependencies']
    hook.install(stack, collector, env['factory'], state, candidate)


def verify(state: Any) -> dict[str, Any]:
    evidence = state['repeated_firing_constructor']
    K.require(evidence['attempts'] == 1 and evidence['installed'] and evidence['closed']
        and evidence['references_restored'] and evidence['qualification']['qualified_restored'],
        'repeated_constructor_or_restore')
    guard = scope_status(state)
    K.require(guard['installed'] and guard['error'] is None and guard['record'] is None
        and guard['same_binding'] and guard['last_frame'] == K.FRAMES[-1], 'same_scope_not_closed')
    return dict(shared_installer=True, actual_constructor_attached=True, references_restored=True,
        firing_rows=len(evidence['rows']), same_scope_guard=guard, quality_gate_clear=False)


def scope_status(state: Any) -> dict[str, Any]:
    guard = state.get('repeat_scope_guard')
    return dict(installed=guard is not None,
        error=None if guard is None or guard.error is None else repr(guard.error),
        record=None if guard is None else guard.record,
        last_frame=None if guard is None else guard.frame,
        same_binding=guard is not None and guard.binding is not None
            and guard.binding is guard.factory.controller.history.get('1P'),
        quality_gate_clear=False, new_scope_continuation=False)
