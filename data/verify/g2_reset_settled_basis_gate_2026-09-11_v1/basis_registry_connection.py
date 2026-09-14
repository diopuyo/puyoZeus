"""同callで発行された確率基準だけを一意所有し、整数採録と別の保存票へ記録する。"""
from __future__ import annotations
import json
from types import MethodType
from typing import Any
import belief as B
import conditioning as C
import registry as R
import serialization as S
import actual_connection_v2 as A

KEY = '_g2_probabilistic_scope_registry'


class Connection:
    def __init__(self, recovery: Any, observer: A.Observer, deadline: int, stream: Any) -> None:
        B.require(type(observer) is A.Observer and observer.recovery is recovery, 'basis_observer_identity')
        B.require(type(deadline) is int and deadline > observer.gate.deadline, 'tracking_bound')
        self.recovery, self.observer, self.deadline, self.stream = recovery, observer, deadline, stream
        self.registry = R.Registry(recovery.factory)
        self.binding: R.Binding | None = None
        self.rows = 0

    def qualify(self, candidate: Any, item: Any) -> None:
        r, gate = self.recovery, self.observer.gate
        B.require(candidate is gate.candidate and gate.state == 'ISSUED', 'basis_candidate_identity')
        B.require(r.error is None and not r.journal.errors and self.observer.error is None, 'basis_prior_error')
        B.require(item['frame'].f_code in r.journal.codes and r.journal.active is None
                  and item['pipe'] is r.pipe, 'basis_actual_J')
        B.require(candidate.source_call_token == item['token'] and candidate.frame == item['scope']['frame_idx'],
                  'basis_source_call')
        B.require(candidate.scope == r.evidence.scope(r.factory, r.pipe)
                  and item['epoch'] == candidate.scope[2], 'basis_live_scope')

    def publish(self, item: Any) -> None:
        candidate = self.observer.gate.candidate
        if candidate is None or self.binding is not None:
            return
        self.qualify(candidate, item)
        board = B.Board.from_dict({'grid': candidate.raw_grid})
        probability = B.ProbabilisticBoard.from_board(board)
        for col, distribution in enumerate(candidate.hidden_probability):
            # 資格済みの実分布をそのままコピーする。normalizeによる補完はしない。
            probability.cell(0, col).probs = dict(distribution)
        value, mass, removed = C.establish_conditioned(candidate.scope, candidate.frame,
                                                      self.deadline, board, probability)
        packet = dict(kind='actual_settled_probabilistic_basis', source_call_token=item['token'],
            basis_deadline=self.observer.gate.deadline, tracking_deadline=self.deadline,
            retained_mass=mass, gravity_removed_worlds=removed, state=S.encode(value),
            integer_current_published=False, legacy_collector_append=False, quality_gate_clear=False)
        encoded = json.dumps(packet, ensure_ascii=False, allow_nan=False)
        self.binding = self.registry.bind(self.recovery.factory, value, item['token'])
        self.stream.write(encoded + '\n')
        self.stream.flush()
        self.rows += 1


def install(stack: Any, recovery: Any, state: dict[str, Any], observer: A.Observer,
            deadline: int) -> Connection:
    B.require(KEY not in vars(recovery.factory) and KEY not in state, 'duplicate_factory_registry')
    stream = stack.enter_context((state['output'] / 'PROBABILISTIC_BASIS.jsonl').open('x', encoding='utf-8'))
    value = Connection(recovery, observer, deadline, stream)
    setattr(recovery.factory, KEY, value.registry)
    state[KEY] = value.registry
    state['probabilistic_basis_connection'] = value
    original = recovery.complete
    def complete(self: Any, item: Any, result: Any, error: Any) -> Any:
        returned = original(item, result, error)
        if error is None:
            value.publish(item)
        return returned
    wrapper = MethodType(complete, recovery)
    def close() -> None:
        owned = vars(recovery).get('complete') is wrapper
        if owned:
            recovery.complete = original
        B.require(owned and getattr(recovery.factory, KEY) is value.registry
                  and state[KEY] is value.registry, 'basis_restore_binding')
        delattr(recovery.factory, KEY)
    stack.callback(close)
    recovery.complete = wrapper
    return value
