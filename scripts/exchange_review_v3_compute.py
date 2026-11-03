"""v3の方式F/quickと独立1秒・2秒探索。レビュー専用。"""
from __future__ import annotations

from typing import Any
from scripts.exchange_review_engine import fast
from scripts.exchange_review_v2 import recommendation, FAST_RATIO
from scripts.exchange_review_v2_budget import compute as one_second, BACK_RATIO
from scripts.exchange_review_v2b_compute import compute as two_seconds, check_prefix
from scripts.exchange_review_v3_input import Inputs, prepare_row, DISPLAY_TTL_SEC
from scripts.exchange_review_seed import PrefixSeeds

MS_PER_SEC = 1000


def compute_row(row: dict, prefix: PrefixSeeds, inputs: Inputs, native: Any,
                recursive: bool = True) -> dict:
    row, seed = prepare_row(row, prefix, inputs)
    event = dict(event_id=row['event_id'], game_idx=row['game_idx'], input_sec=row['t_sec'],
        fire_sec=row['fire_sec'], trigger_sec=row['trigger_sec'], receiver=row['receiver'],
        attacker=row['attacker'], deferred=row['deferred'], seed=seed, status='欠測', fps=row.get('fps', 60.),
        pending_input_sec=row.get('pending_input_sec'), board_audit=row.get('board_audit', {}))
    if seed.get('reason'):
        return event | dict(reason=seed['reason'])
    data = row['exchange_input']
    if len(data['colors']) != 4:
        return event | dict(reason='4色未確定')
    if seed.get('indices') and any(c not in data['colors'] for p in seed['tape'] for c in p):
        seed = dict(status='特定種と観測4色が不一致', input_sec=seed.get('input_sec'))
    if not seed.get('indices') and any(c not in data['colors'] for pair in data['known_receiver'] for c in pair):
        return event | dict(reason='ツモに試合4色以外の色あり')
    fg = fast(row, seed, native)
    # 相手が既にSTABLEなら、再帰にもその時点の確定盤を用いる。
    end = row.get('end_observation')
    if end and end['sides'][data['attacker']]['state'] == 'STABLE' and end['sides'][data['attacker']]['board'] is not None:
        from scripts.exchange_review_v3_input import sanitize
        other = end['sides'][data['attacker']]['board']
        clean, audit = sanitize(data | dict(attacker_board=other))
        fg['opponent_board'] = clean['attacker_board']
        event['board_audit']['attacker_board'] = audit['attacker_board']
    event.update(status='計算済み', seed=seed, exchange_input=data, foreground=fg,
        fast_ready_sec=row['t_sec'] + fg['compute_ms'] * FAST_RATIO / MS_PER_SEC,
        close_sec=row['t_sec'] + data['remaining_sec'] + DISPLAY_TTL_SEC)
    event['recommendations'] = dict(fast=recommendation(event, False))
    if not event['recommendations']['fast']:
        event['reason'] = '探索候補なし'
    if recursive and seed.get('indices'):
        add_recursive(event, native)
    return event


def add_recursive(event: dict, native: Any) -> None:
    event['background'] = one_second(event, native)
    event['background_ready_sec'] = max(event['fast_ready_sec'] + 1 / event['fps'],
        event['input_sec'] + event['background']['compute_ms'] * BACK_RATIO / MS_PER_SEC)
    event['recommendations']['back'] = recommendation(event, True)
    result = two_seconds(event, native)
    result['final_ready_sec'] = max(event['fast_ready_sec'] + 1 / event['fps'],
        event['input_sec'] + result['low_spec_ms'] / MS_PER_SEC)
    result['partial_ready_sec'] = event['input_sec'] + 1. if result['partial_tree'] else None
    if result['partial_tree']:
        result['checked_nodes'] = check_prefix(result['partial_tree'], result['trace_overhead_ms'])
    event['v2b'] = result
