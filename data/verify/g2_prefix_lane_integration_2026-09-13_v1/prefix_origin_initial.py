"""lane開始前に原経路が取得した起点・別名を、検証済みlegacy票から引き継ぐ。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

VERSION = 'prefix-lane-initial/v2'


def encode(serializer: Any, mode: Any, lane: Any) -> dict:
    aliases = [dict(object_id=identity[0], trigger_sec=identity[1], token=token)
        for identity, token in sorted(mode.origin_ids.items())]
    return dict(kind=VERSION, initial=serializer.encode(lane.initial), base=lane.base,
        prior_assumption=lane.assumption, origins=deepcopy(mode.origins),
        origin_aliases=aliases, quality_gate_clear=False)


def remember(parts: Any, mode: Any, receipt: dict, steps: dict, ledger: Any,
             initial_call: str) -> None:
    """旧transitionが通った原起点だけを再導出。保存init内の辞書を真としない。"""
    parser, require = parts.arrival_saved.V.P, parts.mode.B.require
    for origin in (receipt.get('basis_origin'), receipt.get('origin')):
        if origin is None:
            continue
        if 'operation_token' in origin:
            token = origin['operation_token']
            board = parser.basis_origin(origin, steps, ledger, receipt['applied_frame'], initial_call)
        else:
            token = receipt['arrival']['token']
            require(token in ledger.applied, 'prefix_initial_origin_not_applied')
            board = parser.origin(origin, steps, ledger, receipt['applied_frame'])
        actual = deepcopy(origin) | dict(grid=parts.mode.B.grid(board))
        previous = mode.origins.get(token)
        require(previous is None or previous == actual, 'prefix_initial_origin_changed')
        step = steps[origin['source_call_token']]
        triggers = {event['active_origin']['trigger_sec'] for event in step['events']
            if event.get('active_origin') is not None and event['active_origin']['object_id'] == origin['object_id']}
        require(len(triggers) == 1, 'prefix_initial_origin_trigger')
        identity = (origin['object_id'], next(iter(triggers)))
        require(identity not in mode.origin_ids or mode.origin_ids[identity] == token,
            'prefix_initial_origin_alias_changed')
        mode.origins[token], mode.origin_ids[identity] = actual, token
