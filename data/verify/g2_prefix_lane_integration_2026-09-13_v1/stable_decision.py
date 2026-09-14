"""全STABLE判定呼出しの入力・否決理由を残し、資格票の欠落を検査する。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import math

VERSION = 'prefix-stable-decision/v1'
BOARD_KEYS = ('raw', 'cnn', 'sm', 'returned')


def reason(value: dict, hidden_rows: int) -> str | None:
    if value['state'] != 'stable':
        return 'not_STABLE'
    if value['match_active'] is not True or value['effect_window'] is not False:
        return 'inactive_or_effect_window'
    if value['no_origin'] is not True:
        return 'active_origin'
    end = value['grace_end']
    if end is not None and value['time_sec'] < end:
        return 'landing_grace'
    raw = value['raw']
    if raw is None:
        raise ValueError('prefix_decision_raw_missing')
    visible = raw[hidden_rows:]
    if any(value[key] is None or value[key][hidden_rows:] != visible for key in BOARD_KEYS[1:]):
        return 'visible_channel_mismatch'
    return None


def capture(module: Any, mode: Any, item: dict, result: Any, actual_reason: str | None) -> dict:
    local, scope = item['frame'].f_locals, item['scope']
    recovery, side = mode.connection.recovery, scope['side']
    signals = local['signals']
    grace = getattr(recovery.pipe, '_landing_grace_' + side.lower())
    end = None if grace is None else grace[2]
    module.B.require(end is None or (type(end) in (int, float) and math.isfinite(end)), 'prefix_decision_grace')
    value = dict(kind=VERSION, source_call_token=item['token'], scope=dict(scope), software_reset=item['epoch'],
        time_sec=scope['time_sec'], state=None if result is None else result.state.value,
        match_active=signals.is_match_active, effect_window=signals.effect_gate_window_active,
        no_origin=getattr(recovery.pipe, '_active_chain_' + side.lower(), object()) is None,
        grace_end=end, raw=None, cnn=None, sm=None, returned=None, reason=actual_reason,
        flags_source='owned_original_step_locals', quality_gate_clear=False)
    if actual_reason in (None, 'visible_channel_mismatch'):
        view = SimpleNamespace(frame=scope['frame_idx'], clock=scope['time_sec'])
        raw, _ = recovery.provider.raw(recovery.pipe, side, view)
        value['raw'] = [list(row) for row in raw]
        for key, board in zip(BOARD_KEYS[1:], (signals.cnn_board, local['sm'].context.confirmed_board, result.confirmed_board)):
            value[key] = None if board is None else [list(row) for row in module.B.grid(board)]
    module.B.require(reason(value, module.B.HIDDEN_ROWS) == actual_reason, 'prefix_decision_live_disagreement')
    return value


def verify(module: Any, value: dict, step: dict, context: dict) -> None:
    require = module.B.require
    require(value['kind'] == VERSION and value['quality_gate_clear'] is False
        and value['flags_source'] == 'owned_original_step_locals', 'prefix_decision_schema')
    require(value['source_call_token'] == step['token'] and value['software_reset'] == step['software_reset']
        and all(value['scope'][key] == step[key] for key in value['scope']), 'prefix_decision_saved_scope')
    require(step['status'] == 'returned' and step['exception'] is None and step['returned'] is not None,
        'prefix_decision_original_return')
    require(value['state'] == step['returned']['state'].lower()
        and value['no_origin'] == (step['returned']['active_origin'] is None), 'prefix_decision_original_state')
    require(value['time_sec'] == step['time_sec']
        and reason(value, module.B.HIDDEN_ROWS) == value['reason'], 'prefix_decision_saved_reason')
    if value['reason'] in (None, 'visible_channel_mismatch'):
        require(all(context[key] == step[key] for key in ('source_id', 'run_id', 'frame_idx', 'time_sec')),
            'prefix_decision_context_scope')
        side = context['sides'][step['side']]
        require(value['raw'] == side['pb']['raw']['grid'] and value['cnn'] == side['sm']['input_cnn']['grid']
            and value['returned'] == step['returned']['confirmed']['grid'], 'prefix_decision_channels')
