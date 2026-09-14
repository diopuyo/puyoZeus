"""原J全行からprefix laneと新receiptを再計算する。欠落sidecarは補完しない。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import lane_state as N
import prefix_live_adapter as A
import prefix_commit as C
import stable_evidence as E
import stable_decision as D
import prefix_origin_initial as I


def indexed(rows: list, key: str, require: Any) -> dict:
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'prefix_saved_duplicate_sidecar')
    return result


class Replay:
    def __init__(self, parts: Any, saved: dict, contexts: dict) -> None:
        self.parts, self.saved, self.contexts = parts, saved, contexts
        self.V = parts.arrival_saved.V
        self.R, self.L = self.V.R, self.V.L
        self.require = parts.mode.B.require
        self.decisions = indexed(saved['decisions'], 'source_call_token', self.require)
        self.qualifications = indexed(saved['qualifications'], 'journal_token', self.require)
        self.observations = indexed(saved['observations'], 'evidence_key', self.require)
        self.seen_decisions: set[str] = set()
        self.seen_qualifications: set[str] = set()
        self.seen_observations: set[str] = set()
        self.origin_index = 0
        self.lane: Any = None
        self.mode = SimpleNamespace(origins={}, origin_ids={}, arrival_ledger=None)
        self.core = SimpleNamespace(module=parts.mode, parts=parts, write=self.origin_write)

    def equal(self, actual: Any, expected: Any, reason: str) -> None:
        self.require(self.R.normalized(actual) == self.R.normalized(expected), reason)

    def start(self, current: Any, ledger: Any) -> None:
        self.require(self.lane is None and len(self.saved['initial']) == 1, 'prefix_saved_initial_count')
        self.lane = N.Lane(self.parts, current, ledger, A.ASSUMPTION)
        self.equal(self.saved['initial'][0], I.encode(self.parts.mode.BASE.V1.S, self.mode, self.lane), 'prefix_saved_initial')

    def origin_write(self, mode: Any, filename: str, actual: dict) -> None:
        self.require(filename == 'PREFIX_ORIGIN_ASSIGNMENTS.jsonl'
            and self.origin_index < len(self.saved['origins']), 'prefix_saved_origin_missing')
        self.equal(actual, self.saved['origins'][self.origin_index], 'prefix_saved_origin_assignment')
        self.origin_index += 1

    def origins(self, ledger: Any, step: dict) -> None:
        self.mode.arrival_ledger = ledger
        for event in step['events']:
            origin = event.get('active_origin')
            if origin is None:
                continue
            identity = (origin['object_id'], origin['trigger_sec'])
            if identity in self.mode.origin_ids:
                prior = self.mode.origins[self.mode.origin_ids[identity]]
                self.require(origin['before_board'] is not None
                    and tuple(map(tuple, origin['before_board']['grid'])) == prior['grid'], 'prefix_saved_origin_mutation')
            else:
                A.Live.origin(self.core, self.mode, self.lane, step, origin, identity)

    def qualification(self, ledger: Any, step: dict, required: bool) -> dict | None:
        call = step['token']
        decision = self.decisions.get(call)
        self.require(not required or decision is not None, 'prefix_saved_decision_missing')
        if decision is None:
            self.require(call not in self.qualifications, 'prefix_saved_qualification_without_decision')
            return None
        D.verify(self.parts.mode, decision, step, self.contexts[step['frame_idx']])
        self.seen_decisions.add(call)
        qualification = self.qualifications.get(call)
        self.require((qualification is not None) == (decision['reason'] is None), 'prefix_saved_qualification_missing')
        if qualification is not None:
            E.verify(self.parts, qualification, step, self.contexts[step['frame_idx']], ledger)
            value = qualification['stable_qualification']
            for key in ('state', 'match_active', 'effect_window', 'no_origin', 'grace_end', 'raw', 'cnn', 'sm', 'returned'):
                self.equal(decision[key], value[key], 'prefix_saved_decision_qualification_disagreement')
            self.seen_qualifications.add(call)
        return qualification

    def advance(self, ledger: Any, row: dict, step: dict, qualification: dict | None) -> tuple:
        call = step['token']
        if qualification is None:
            self.require(call not in self.observations and row.get('transition') is None,
                'prefix_saved_unqualified_transition')
            self.equal(row['reason'], 'arrival:' + self.decisions[call]['reason'], 'prefix_saved_reason')
            return self.lane.current, ledger, None
        self.require(call in self.observations, 'prefix_saved_observation_missing')
        observed = self.parts.mode.B.Board.from_dict({'grid': qualification['stable_qualification']['observed']})
        actual = self.lane.observe(ledger, step['frame_idx'], observed, call)
        self.equal(actual, self.observations[call], 'prefix_saved_observation_replay')
        self.seen_observations.add(call)
        prepared = self.lane.prepare(ledger, call)
        if prepared is None:
            self.require(row.get('transition') is None and row.get('provisional_update') is False,
                'prefix_saved_ambiguous_transition')
            self.equal(row['reason'], 'prefix_ambiguous_prepop_or_no_new_arrival', 'prefix_saved_reason')
            return self.lane.current, ledger, None
        self.equal(prepared.receipt, row.get('transition'), 'prefix_saved_receipt')
        self.require(row.get('provisional_update') is True, 'prefix_saved_update_flag')
        self.equal(row['reason'], C.VERSION + '_applied', 'prefix_saved_reason')
        self.lane.accepted(prepared, prepared.following)
        return prepared.following, prepared.ledger_after, row['transition']

    def finish(self) -> None:
        self.require(self.seen_decisions == set(self.decisions)
            and self.seen_qualifications == set(self.qualifications)
            and self.seen_observations == set(self.observations)
            and self.origin_index == len(self.saved['origins']), 'prefix_saved_unused_sidecar')
        self.require(self.lane is not None or not self.saved['initial'], 'prefix_saved_unused_initial')


def replay(parts: Any, saved: dict, contexts: dict, rows: list, packets: list, journal: list,
           initial: Any, prior: Any, initial_call: str, end: int, verify_enqueue: Any) -> tuple:
    state = Replay(parts, saved, contexts)
    V, R, L = state.V, state.R, state.L
    enqueues, steps = V.selected(journal, initial, end)
    L.require(callable(verify_enqueue) and len(rows) == len(packets) == len(steps), 'prefix_saved_rows')
    L.require([r['journal_token'] for r in rows] == [s['token'] for s in steps]
        and [p['completed_call_token'] for p in packets] == [s['token'] for s in steps], 'prefix_saved_order')
    originals = indexed([row for row in journal if row.get('kind') == 'step'], 'token', state.require)
    ledger = L.Ledger(initial.scope, initial.frame, initial.deadline, initial.frame)
    current, receipts, closed = initial, [], False
    for row, packet, enqueue, step in zip(rows, packets, enqueues, steps):
        ledger = R.arrival(ledger, packet, enqueue, step, verify_enqueue)
        ledger = R.acknowledge(ledger, row, step)
        qualification = state.qualification(ledger, step, state.lane is not None)
        if state.lane is not None:
            state.origins(ledger, step)
            current, ledger, receipt = state.advance(ledger, row, step, qualification)
            if receipt is not None:
                receipts.append(receipt)
        elif row.get('transition') is not None:
            L.require(qualification is not None, 'prefix_saved_basis_qualification')
            current, ledger, closed = V.P.transition(current, ledger, row, step, originals, prior, closed, initial_call)
            receipts.append(row['transition'])
            I.remember(parts, state.mode, row['transition'], originals, ledger, initial_call)
            if closed and ledger.applied:
                state.start(current, ledger)
        ledger = R.finished_row(ledger, row, step)
    state.finish()
    return current, ledger, receipts, closed
