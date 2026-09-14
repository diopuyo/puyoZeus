"""原journalの一回FIFO消費を読む。これだけで物理着地とは認定しない。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import belief as B

FRAME_STRIDE = 2  # 固定診断runの原更新間隔。


@dataclass(frozen=True)
class Consumption:
    source_call_token: str
    occurrence_token: str
    pair: tuple[int, int]
    consumed_frame: int | None = None
    physical_landing_certified: bool = field(default=False, init=False)
    accounting_permission: bool = field(default=False, init=False)


def extract(item: dict[str, Any]) -> Consumption | None:
    events = item['events']
    before = [(i, event) for i, event in enumerate(events) if event['stage'] == 'fifo_before']
    after = [(i, event) for i, event in enumerate(events) if event['stage'] == 'fifo_after']
    if not before and not after:
        return None
    B.require(len(before) == len(after) == 1 and before[0][0] < after[0][0], 'native_fifo_stage_pair')
    start, end = before[0][1], after[0][1]
    queue, tail = start['accounting']['pending_tsumo'], end['accounting']['pending_tsumo']
    tokens = start['fifo_occurrence_tokens']
    B.require(type(queue) is list and bool(queue) and type(tail) is list
              and type(tokens) is list and len(tokens) == len(queue), 'native_fifo_before')
    B.require(tail == queue[1:] and end['committed'] == queue[0], 'native_fifo_not_single_head_pop')
    token = end['enqueue_occurrence_token']
    B.require(type(token) is str and bool(token) and token == tokens[0], 'native_fifo_unowned_token')
    pair = tuple(end['committed'])
    B.require(len(pair) == 2 and all(type(c) is int and c in B.PIECE_COLORS for c in pair), 'native_fifo_pair')
    B.require(type(item['token']) is str and bool(item['token']), 'native_call_token')
    frame = item['scope']['frame_idx']
    B.require(type(frame) is int and frame >= 0, 'native_consumption_frame')
    return Consumption(item['token'], token, pair, frame)


class Recorder:
    """確率bindingに続く原Jだけを認証し、未反映の実消費を保持する。"""
    def __init__(self, connection: Any) -> None:
        B.require(connection.binding is not None, 'native_basis_missing')
        self.connection = connection
        self.last_frame = connection.registry.current(connection.binding).frame
        self.seen_calls: set[str] = set()
        self.seen_occurrences: set[str] = set()
        self.pending: list[Consumption] = []

    def observe(self, item: dict[str, Any], error: Any) -> Consumption | None:
        c, r = self.connection, self.connection.recovery
        current = c.registry.current(c.binding)
        B.require(error is None and r.error is None and not r.journal.errors, 'native_prior_error')
        B.require(item['frame'].f_code in r.journal.codes and item['pipe'] is r.pipe
                  and r.journal.active is None, 'native_actual_J')
        local = item['frame'].f_locals
        frame, clock = local['frame_idx'], local['time_sec']
        scope = r.journal.scope(r.pipe, current.scope[-1], frame, clock)
        # 原J開始後にTSUMO_FALLへ進入すればaction_revisionは正当に変わる。
        # reset世代と時計は一致必須。action版を開始時と終了時で無条件同一にしない。
        without_action = lambda v: dict(v, generation={k: x for k, x in v['generation'].items()
                                                       if k != 'action_revision'})
        B.require(without_action(item['scope']) == without_action(scope)
                  and item['epoch'] == r.journal.epoch(r.pipe, current.scope[-1]) == current.scope[2]
                  and r.evidence.scope(r.factory, r.pipe) == current.scope, 'native_live_scope')
        B.require(frame == self.last_frame + FRAME_STRIDE and frame <= current.deadline, 'native_frame_bound')
        B.require(item['token'] not in self.seen_calls, 'native_duplicate_call')
        event = extract(item)
        if event is not None:
            prefix = (current.scope[0] + ':' + current.scope[1] + ':reset:'
                      + str(current.scope[2]) + ':' + current.scope[-1] + ':enqueue:')
            B.require(event.occurrence_token.startswith(prefix), 'native_occurrence_scope')
            B.require(event.occurrence_token not in self.seen_occurrences, 'native_duplicate_consumption')
            self.seen_occurrences.add(event.occurrence_token)
            self.pending.append(event)
        self.last_frame = frame
        self.seen_calls.add(item['token'])
        return event
