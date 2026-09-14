"""原Jのframe解放前に採録資格の局所flagだけを両side分捕捉する。"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

SIDES = ('1P', '2P')


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('evaluation_flags:' + reason)


def capture(journal: Any, item: dict, result: Any) -> dict:
    frame, pipe, scope = item['frame'], item['pipe'], item['scope']
    require(frame is not None and frame.f_code in journal.codes and pipe is journal.pipe,
            'original_J_owner')
    side, local = scope['side'], frame.f_locals
    require(side in SIDES and local['side'] == side and local['frame_idx'] == scope['frame_idx'], 'side_clock')
    base = dict(scope=dict(scope), token=item['token'], software_reset=item['epoch'])
    if result is None or 'signals' not in local:
        return base | dict(hold_reason='signals_or_result_missing')
    sm = getattr(pipe, '_sm_' + side.lower())
    require(local['sm'] is sm and sm.context.frame_idx == scope['frame_idx'], 'SM_clock')
    signals, grace = local['signals'], getattr(pipe, '_landing_grace_' + side.lower())
    value = base | dict(state=result.state.value, match_active=signals.is_match_active,
        effect_window=signals.effect_gate_window_active,
        origin_present=getattr(pipe, '_active_chain_' + side.lower()) is not None,
        grace_end=None if grace is None else grace[2], hold_reason=None)
    return json.loads(json.dumps(value, allow_nan=False))


def install(stack: Any, journal: Any) -> Any:
    evidence = SimpleNamespace(journal=journal, latest={}, closed=False, error=None)
    existed, previous = 'complete_step' in vars(journal), vars(journal).get('complete_step')
    original = journal.complete_step
    def completed(item: dict, result: Any, error: Any, profile: Any) -> Any:
        packet, failure = None, None
        if error is None:
            try: packet = capture(journal, item, result)
            except BaseException as caught: failure = caught
        try:
            returned = original(item, result, error, profile)
        except BaseException as caught:
            evidence.error = caught
            raise
        if failure is not None:
            evidence.error = failure
            raise failure
        if error is not None: evidence.error = error
        if packet is not None: evidence.latest[packet['scope']['side']] = packet
        return returned
    def close(kind: Any, body: Any, trace: Any) -> bool:
        try:
            require(vars(journal).get('complete_step') is completed, 'foreign_hook')
            setattr(journal, 'complete_step', previous) if existed else delattr(journal, 'complete_step')
        except BaseException as caught:
            evidence.error = caught
            if body is None: raise
        finally: evidence.closed = True
        return False
    stack.push(close)
    journal.complete_step = completed
    return evidence
