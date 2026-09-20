"""2P原Jのframe解放前に基準取得用の実観測を捕捉する。基準自体は発行しない。"""
from __future__ import annotations

import hashlib
import json
from typing import Any
import journal_context as C

SIDE = '2P'


def same_generation(before: Any, after: Any) -> bool:
    # 原Jの途中で落下へ進むaction更新は正常。reset世代・他の値は一致必須。
    def identity(value: Any) -> Any:
        return dict(value,generation={k:v for k,v in value['generation'].items() if k!='action_revision'})
    return identity(before)==identity(after)


def capture(journal: Any, state: dict[str, Any], item: Any, result: Any, error: Any,
            registered_scope: Any = None, side: str = SIDE) -> dict[str, Any]:
    C.require(side in ('1P', '2P'), 'normal_observation_side')
    C.require(error is None and not journal.errors and journal.active is None, 'second_J_error')
    frame, pipe = item['frame'], item['pipe']
    C.require(frame is not None and frame.f_code in journal.codes and journal.pipe is pipe, 'second_actual_J')
    local, scope = frame.f_locals, item['scope']
    C.require(scope['side'] == side and local['side'] == side, 'second_side')
    if 'signals' not in local or result is None:
        return dict(frame=scope['frame_idx'], journal_token=item['token'], side=side,
            hold_reason='signals_or_result_missing', basis_registered=False, quality_gate_clear=False)
    if local['signals'].is_match_active is not True:
        return dict(frame=scope['frame_idx'], journal_token=item['token'], side=side,
            hold_reason='match_inactive', basis_registered=False, quality_gate_clear=False)
    sm = getattr(pipe, '_sm_' + side.lower())
    C.require(local['sm'] is sm and local['frame_idx'] == scope['frame_idx']
        and local['time_sec'] == scope['time_sec'] == sm.context.frame_idx/60, 'second_SM_clock')
    current_scope = journal.scope(pipe, side, scope['frame_idx'], scope['time_sec'])
    current_epoch=journal.epoch(pipe,side)
    if registered_scope is not None and (not same_generation(current_scope,scope) or item['epoch']!=current_epoch):
        C.require(all(current_scope[k]==scope[k] for k in ('source_id','run_id','side','pipe_object_id','frame_idx','time_sec'))
            and current_scope['generation']['reset_epoch']>=scope['generation']['reset_epoch']
            and current_epoch>=item['epoch'] and (current_scope['generation']['reset_epoch']>scope['generation']['reset_epoch']
                or current_epoch>item['epoch']), 'second_reset_observation_identity')
        return dict(frame=scope['frame_idx'],journal_token=item['token'],side=side,
            hold_reason='generation_changed',basis_registered=False,quality_gate_clear=False)
    C.require(same_generation(current_scope,scope) and item['epoch'] == journal.epoch(pipe, side), 'second_generation')
    observer = state['hidden_probability_observer']
    C.require(observer.active is None and not observer.failures and observer.rows, 'second_PB_lifetime')
    pb = observer.rows[-1]
    C.require(all(pb[key] == scope[key] for key in ('source_id','run_id','frame_idx','time_sec','side')), 'second_PB_scope')
    typed = state['postcommit_current_receiver'].rec.side_value(result)
    C.require(typed['probability']['cells'] == pb['probability']['cells']
        and typed['confirmed'] == pb['confirmed'], 'second_returned_PB')
    signals = local['signals']
    grace = getattr(pipe, '_landing_grace_' + side.lower())
    value = dict(scope=[scope['source_id'], scope['run_id'], item['epoch'], id(pipe), id(sm),
        scope['generation']['reset_epoch'], side], frame=scope['frame_idx'], clock=scope['time_sec'],
        journal_token=item['token'], original_code_sha256=hashlib.sha256(frame.f_code.co_code).hexdigest(),
        match_active=signals.is_match_active, effect_window=signals.effect_gate_window_active,
        origin_present=getattr(pipe, '_active_chain_' + side.lower()) is not None,
        landing_grace_expired=grace is None or scope['time_sec'] >= grace[2],
        state=typed['state_value'], raw=pb['raw'], confirmed=typed['confirmed'],
        probability=pb['probability'], pb_holds=pb['hold_reasons'], pb_errors=pb['instrumentation_errors'],
        original_J_before_cleanup=True, basis_registered=False, quality_gate_clear=False)
    return json.loads(json.dumps(value, allow_nan=False))


def install(stack: Any, journal: Any, state: dict[str, Any], side: str = SIDE) -> Any:
    C.require(side in ('1P', '2P'), 'normal_observer_side')
    from types import SimpleNamespace
    evidence = SimpleNamespace(latest=None, error=None, closed=False, journal=journal, state=state,registered_scope=None)
    existed, previous = 'complete_step' in vars(journal), vars(journal).get('complete_step')
    original = journal.complete_step
    def completed(item: Any, result: Any, error: Any, profile: Any) -> Any:
        packet, failure = None, None
        if item['scope']['side'] == side and error is None:
            try: packet = capture(journal, state, item, result, error,evidence.registered_scope,side=side)
            except BaseException as caught: failure = caught
        try:
            returned = original(item, result, error, profile)
        except BaseException as original_error:
            evidence.error, evidence.latest = original_error, None
            raise
        if failure is not None:
            evidence.error = failure
            raise failure
        evidence.latest = packet if item['scope']['side'] == side else evidence.latest
        return returned
    def close(kind: Any, body: Any, trace: Any) -> bool:
        try:
            C.require(vars(journal).get('complete_step') is completed, 'second_foreign_hook')
            setattr(journal, 'complete_step', previous) if existed else delattr(journal, 'complete_step')
        except BaseException as error:
            evidence.error = error
            if body is None: raise
        finally: evidence.closed = True
        return False
    stack.push(close)
    journal.complete_step = completed
    return evidence
