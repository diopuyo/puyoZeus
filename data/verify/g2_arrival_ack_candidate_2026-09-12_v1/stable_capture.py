"""元STABLE資格通過時の実局所値を保存し、元採録盤面と別途突合する。"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import base_mode as BASE
import source_replay as R

B, L = BASE.B, R.L


def grid(board: Any) -> list:
    return R.normalized(B.grid(board))


def capture(mode: Any, item: dict, result: Any, observed: Any) -> dict:
    r, side = mode.connection.recovery, mode.connection.binding.scope[-1]
    local, scope = item['frame'].f_locals, item['scope']
    L.require(scope['side'] == side and item['pipe'] is r.pipe, 'stable_capture_owner')
    signals = local['signals']
    view = SimpleNamespace(frame=scope['frame_idx'], clock=scope['time_sec'])
    grace = getattr(r.pipe, '_landing_grace_' + side.lower())
    raw, _ = r.provider.raw(r.pipe, side, view)
    value = dict(source_call_token=item['token'], scope=R.normalized(scope),
        software_reset=item['epoch'], match_active=signals.is_match_active,
        effect_window=signals.effect_gate_window_active,
        no_origin=r.provider.no_origin(r.pipe, side, view),
        grace_end=None if grace is None else grace[2], state=result.state.value,
        raw=R.normalized(raw), cnn=grid(signals.cnn_board), sm=grid(local['sm'].context.confirmed_board),
        returned=grid(result.confirmed_board), observed=grid(observed), quality_gate_clear=False)
    validate(value)
    return value


def validate(value: dict) -> None:
    clock = value['scope']['time_sec']
    L.require(value['state'] == 'stable' and value['match_active'] is True
        and value['effect_window'] is False and value['no_origin'] is True
        and value['quality_gate_clear'] is False, 'stable_saved_flags')
    end = value['grace_end']
    L.require(end is None or (type(end) in (int, float) and math.isfinite(end) and clock >= end),
              'stable_saved_grace')
    boards = [B.grid(B.Board.from_dict({'grid': value[k]})) for k in ('raw', 'cnn', 'sm', 'returned', 'observed')]
    L.require(all(b[B.HIDDEN_ROWS:] == boards[0][B.HIDDEN_ROWS:] for b in boards), 'stable_saved_visible')


def verify(row: dict, step: dict, context: dict) -> bool:
    value = row['stable_qualification']
    validate(value)
    L.require(value['source_call_token'] == step['token'] == row['journal_token'], 'stable_saved_call')
    for key in ('frame_idx', 'time_sec', 'side', 'source_id', 'run_id', 'pipe_object_id', 'generation'):
        L.require(value['scope'][key] == step[key], 'stable_saved_scope')
    L.require(value['software_reset'] == step['software_reset'], 'stable_saved_epoch')
    L.require(step['returned']['state'] == 'STABLE'
        and value['returned'] == step['returned']['confirmed']['grid'], 'stable_saved_returned')
    L.require(all(context[k] == step[k] for k in ('source_id', 'run_id', 'frame_idx', 'time_sec')),
              'stable_context_scope')
    side = context['sides'][step['side']]
    pairs = (('raw', side['pb']['raw']), ('cnn', side['sm']['input_cnn']),
             ('sm', side['sm']['context_after']['confirmed']), ('returned', side['before_hold']['confirmed']))
    L.require(all(value[k] == saved['grid'] for k, saved in pairs), 'stable_context_channels')
    return True
