"""連鎖終了通知は発火起点に割り当てず、非消費の別票に記録する。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any


def is_settled(origin: dict[str, Any]) -> bool:
    from src.event_physical_observer_v1 import _MECHANISM_KIND
    from src.chain_id_resolver import ObservationKind
    return _MECHANISM_KIND.get(origin.get('mechanism')) == ObservationKind.CHAIN_SETTLED


def partition(item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events, notices = [], []
    for event in item['events']:
        origin = event.get('active_origin')
        if origin is not None and is_settled(origin):
            notices.append(dict(stage=event['stage'], origin=deepcopy(origin)))
            events.append(dict(event, active_origin=None))
        else:
            events.append(event)
    return dict(item, events=events), notices


def packet(mode: Any, item: dict[str, Any], notices: list[dict[str, Any]]) -> dict[str, Any]:
    c, native = mode.connection, mode.native
    frame = item['scope']['frame_idx']
    require = mode.B.require
    require(c.binding.scope[-1] == '2P' and native.connection is c
            and native.last_frame == frame and item['token'] in native.seen_calls,
            'settled_notice_native_call')
    for notice in notices:
        origin = notice['origin']
        require(origin['before_board'] is not None, 'settled_notice_missing_before')
        key = (origin['object_id'], float(origin['trigger_sec']))
        grid = mode.B.grid(mode.B.Board.from_dict({'grid': origin['before_board']['grid']}))
        require(key not in mode.settled_ids or mode.settled_ids[key] == grid, 'settled_notice_mutated')
        mode.settled_ids[key] = grid
    return dict(kind='second_settled_notice/v1', scope=list(c.binding.scope),
        initial_call_token=c.binding.initial_call_token, source_call_token=item['token'], frame=frame,
        notices=notices, classification='CHAIN_SETTLED', physical_identity_certified=False,
        operation_token_assigned=False, next_consumed=False, probability_updated=False,
        quality_gate_clear=False)


def mode_class(original: Any, belief: Any, write: Any) -> type:
    class Mode(original):
        B = belief

        def __init__(self, connection: Any, state: dict[str, Any], stream: Any) -> None:
            super().__init__(connection, state, stream)
            self.settled_ids: dict[tuple[int, float], Any] = {}

        def capture_origin(self, item: dict[str, Any]) -> None:
            view, notices = partition(item)
            if notices:
                write(packet(self, item, notices))
            # Nativeには既に元itemが渡っている。発火選択だけを分類し、原J/active originは保持。
            super().capture_origin(view)
    return Mode
