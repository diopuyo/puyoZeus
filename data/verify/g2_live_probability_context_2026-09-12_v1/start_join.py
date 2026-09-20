"""原canonical buildを保持し、新開始証拠を会計原票へ再突合する。"""
from __future__ import annotations
from types import FunctionType
from typing import Any
import join as J
import start_qualification as Q

FIELDS = {'start_counter_trace', 'start_parameters', 'start_decision'}
SIDES = ('p1', 'p2')


def checked_accounting(value: dict) -> None:
    """同prefixの保存会計を積み上げ、独立traceとの食い違いを拒否する。"""
    accounting = value['accounting']
    gross = dict(accounting['initial_gross_counters'])
    changes = {row['frame_idx']: row for row in accounting['rows']}
    Q.require(len(changes) == len(accounting['rows']), 'duplicate_accounting_frame')
    pending = [0, 0]
    finalized = [0, 0]
    for trace in value['start_counter_trace']:
        row = changes.pop(trace['frame'], None)
        if row is not None:
            Q.require([row['pending_before'][s] for s in SIDES] == pending, 'pending_before')
            gross = {key: amount + row['deltas'][key] for key, amount in gross.items()}
            pending = [row['pending_after'][s] for s in SIDES]
            for attack in row['attack_finalizations']:
                side = SIDES.index(attack['side'])
                finalized[side] += 1
                Q.require(attack['side_ordinal'] == finalized[side], 'finalized_ordinal')
        Q.require(trace['pending'] == pending, 'trace_pending_mismatch')
        Q.require(trace['resets'] == [gross['boundary_resets_' + s] for s in SIDES], 'trace_reset_mismatch')
        Q.require(trace['activity']['finalized'] == finalized, 'trace_finalized_mismatch')
        for key in Q.ACTIVITY:
            if key != 'finalized':
                Q.require(trace['activity'][key] == [gross[key + '_' + s] for s in SIDES], 'trace_activity_mismatch')
    Q.require(not changes and gross == accounting['final_gross_counters'], 'trace_final_gross')
    Q.require(pending == [accounting['final_pending_uncapped'][s] for s in SIDES], 'trace_final_pending')


def checked_metadata(value: dict) -> None:
    traces = {r['frame']: r for r in value['start_counter_trace']}
    for row in value['stable_snapshots'] + value['metadata']:
        index = ('1P', '2P').index(row['side'])
        frame = row['frame_idx']
        Q.require(frame in traces, 'trace_metadata_frame')
        args, trace = row['arguments'], traces[frame]
        Q.require(trace['scores'][index] == args['score'] and trace['states'][index] == args['bstate']['value'],
                  'trace_metadata_mismatch')
    Q.require(value['start_counter_trace'][-1]['game'] == value['observed_game_idx'], 'trace_game_mismatch')


def start_eligible(value: dict, boundaries: tuple) -> bool:
    if not FIELDS.intersection(value):
        return J.start_eligible(value, boundaries)
    Q.require(FIELDS.issubset(value), 'partial_start_schema')
    Q.require(value['start_parameters'] == dict(debounce_sec=Q.DEBOUNCE_SEC,
                                               visual_persist_sec=Q.VISUAL_PERSIST_SEC), 'start_parameters')
    decision = Q.decide(value, debounce_sec=Q.DEBOUNCE_SEC)
    Q.require(decision == value['start_decision'], 'saved_start_decision')
    checked_accounting(value)
    checked_metadata(value)
    return decision['eligible']


# 原buildのheader/盤面STABLE/canonical会計/境界時刻は無変更で再用する。
build = FunctionType(J.build.__code__, dict(vars(J), start_eligible=start_eligible), J.build.__name__)


def __getattr__(name: str) -> Any:
    return getattr(J, name)
