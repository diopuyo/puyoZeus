"""元2P prefix再計算を保持し、終端handoff以後だけ新しい原票契約で再計算する。"""
from __future__ import annotations
from dataclasses import asdict
from types import SimpleNamespace as N
from typing import Any
import json
import second_arrival_state as A
import observed_terminal_drop as D
import warning_source as W

STRIDE = 2


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


class Replay:
    def __init__(self, parts: Any, services: Any, old: Any, source: dict[int, dict],
                 warnings: tuple[dict, ...], terminal: list[dict]) -> None:
        self.parts, self.services, self.old = parts, services, old
        self.source, self.warnings, self.terminal = source, warnings, terminal
        self.L, self.check = parts.mode.L, parts.mode.B.require
        self.ledger, self.lane, self.handoff = None, None, None
        self.used_terminal: set[int] = set()
        self.notice_ids: dict = {}
        self.pending_timeline: dict[int, tuple] = {old.current.frame: ()}

    def advance_source(self, frame: int) -> None:
        if self.ledger is None:
            initial = self.old.lane.initial
            self.ledger = self.L.Ledger(initial.scope, initial.frame, initial.deadline, initial.frame)
        frames = tuple(range(self.ledger.clock + STRIDE, frame + STRIDE, STRIDE))
        rows, events = [], []
        for tick in frames:
            step = self.old.original.step_for(self.old.steps, tick, list(self.ledger.scope))
            event = self.old.native.extract(dict(token=step['token'], scope=dict(frame_idx=tick), events=step['events']))
            rows.append(self.source[tick])
            if event is not None:
                events.append(event)
        self.ledger = A.advance(self.L, self.ledger, tuple(rows), tuple(events), frame)

    def warnings_verified(self, frame: int) -> tuple:
        selected = tuple(row for row in self.warnings if self.old.current.frame < row['frame'] <= frame)
        for row in selected:
            step = self.old.original.step_for(self.old.steps, row['frame'], list(self.ledger.scope))
            self.check(row['call_token'] == step['token'] and tuple(row['scope']) == self.ledger.scope,
                       'terminal_saved_warning_original_call')
            context = self.old.contexts[row['frame']]['sides'][self.ledger.scope[-1]]
            confirmed = context['sm']['context_after']['confirmed']
            present = None if confirmed is None else any(W.OJAMA in values
                for values in confirmed['grid'][self.old.engine.B.HIDDEN_ROWS:])
            self.check(row['confirmed_has_ojama'] == present, 'terminal_saved_confirmed_garbage')
            origins = any(event.get('active_origin') is not None for event in step['events'])
            self.check(row['active_origin'] == origins, 'terminal_saved_origin_flags')
        return selected

    def old_record(self, row: dict, step: dict) -> None:
        self.old.record(row, step)
        if row['kind'] == 'second_prefix_initial':
            self.advance_source(step['frame_idx'])
        elif row['kind'] == 'second_prefix_commit':
            receipt = row['receipt']
            self.check(receipt['before_prefix'] == len(self.ledger.applied), 'terminal_saved_old_applied_prefix')
            self.ledger = self.L.applied(self.ledger, tuple(receipt['applied_tokens']), step['frame_idx'])
            self.check(len(self.ledger.applied) <= len(self.ledger.acknowledgements), 'terminal_saved_old_without_ack')

    def prepare_handoff(self, row: dict, step: dict) -> tuple:
        frame, check = step['frame_idx'], self.check
        check(self.handoff is None and self.old.decisions[frame]['reason'] is None,
              'terminal_saved_unqualified_or_duplicate')
        observed = self.old.engine.B.Board.from_dict({'grid': self.old.decisions[frame]['raw']})
        self.old.sync()  # 元Adapter.progressと同じ順序で、最後の原消費をlaneへ接続する。
        try:
            self.old.lane.condition(frame, observed, self.old.origin_adapter.fired)
        except ValueError as error:
            check(str(error) == 'probabilistic_scope:consumed_prefix_zero_support', 'terminal_saved_wrong_original_failure')
        else:
            check(False, 'terminal_saved_original_had_support')
        evidence = W.select(check, self.warnings_verified(frame), self.ledger.scope, self.old.current.frame, frame)
        family, drop = D.candidate(self.parts, self.services.commit, self.old.current, self.ledger, observed, evidence)
        prepared = self.services.commit.prepare(self.parts, self.old.current, self.ledger, self.ledger.applied,
            (family,), frame, step['token'], 'warning-source:' + step['token'])
        receipt = prepared.receipt | dict(kind=D.VERSION, drop=drop,
            applied_without_ack_tokens=[token for token in prepared.ledger_after.applied
                if token not in {ack.token for ack in prepared.ledger_after.acknowledgements}])
        check(normalized(receipt) == row['terminal_receipt'] == self.old.receipts[frame],
              'terminal_saved_handoff_math_receipt')
        return prepared, receipt

    def accept_handoff(self, row: dict, step: dict) -> None:
        before, receipt = self.prepare_handoff(row, step)
        check = self.check
        check(len(self.terminal) >= 3 and self.terminal[0]['kind'] == 'prepared'
              and self.terminal[1]['kind'] == 'committed', 'terminal_saved_prepare_commit_order')
        first, second = self.terminal[:2]
        self.old.origin_adapter.mode.native = N(pending=self.old.pending)
        snapshot = self.old.origin_adapter.snapshot()
        check(first['old_prefix_snapshot'] == normalized(snapshot)
              and first['old_ledger'] == normalized(asdict(self.ledger)), 'terminal_saved_old_state')
        check(first['receipt'] == second['receipt'] == normalized(receipt), 'terminal_saved_receipt_consistency')
        tokens = tuple(ack.token for ack in self.ledger.acknowledgements if ack.token not in self.ledger.applied)
        check(tuple(event.occurrence_token for event in self.old.pending) == tokens
              and second['removed_private_native_ack_tokens'] == list(tokens), 'terminal_saved_real_ack_only')
        self.old.pending.clear()
        self.ledger = before.ledger_after
        check(second['ledger'] == normalized(asdict(self.ledger)), 'terminal_saved_committed_ledger')
        self.old.current = before.following
        self.lane = self.services.lane.Lane(self.parts, before.following, self.ledger,
            '予告陽性と観測後の終端30着弾に条件付け。較正済み保証ではない。')
        check(row['lane'] == normalized(self.old.lane.snapshot(self.old.S)) and row['pending'] == []
              and row['historical_lane_retained'] is True, 'terminal_saved_historical_lane')
        self.handoff = step['frame_idx']
        self.used_terminal.update((0, 1))

    def verify_notice(self, row: dict, step: dict, notices: list) -> None:
        connection = N(binding=N(scope=tuple(self.old.current.scope),
            initial_call_token=self.old.packet['initial']['source_call_token']))
        mode = N(B=self.old.engine.B, connection=connection, settled_ids=self.notice_ids,
            native=N(connection=connection, last_frame=step['frame_idx'], seen_calls={step['token']}))
        item = dict(scope=dict(frame_idx=step['frame_idx']), token=step['token'])
        expected = self.services.notice.packet(mode, item, notices)
        self.check(row['notice'] == normalized(expected), 'terminal_saved_settled_notice')

    def terminal_frame(self, frame: int) -> None:
        check = self.check
        check(tuple(a.token for a in self.ledger.arrivals) == self.lane.base, 'terminal_saved_new_arrival')
        rows = [(i, row) for i, row in enumerate(self.terminal) if row.get('frame') == frame]
        step = self.old.steps[frame]
        view, notices = self.services.notice.partition(dict(events=step['events']))
        check(all(event.get('active_origin') is None for event in view['events']), 'terminal_saved_new_fire')
        observation_count, notice_count = 0, 0
        for index, row in rows:
            check(index not in self.used_terminal, 'terminal_saved_reused_record')
            self.used_terminal.add(index)
            if row['kind'] == 'settled_notice':
                check(bool(notices), 'terminal_saved_false_notice')
                self.verify_notice(row, step, notices)
                notice_count += 1
                continue
            if row['kind'] == 'late_ack':
                check(bool(self.old.pending) and row['events'] == normalized([asdict(e) for e in self.old.pending])
                    and row['ledger'] == normalized(asdict(self.ledger))
                    and row['physically_reapplied'] is False, 'terminal_saved_late_ack')
                check(all(event.occurrence_token in self.ledger.applied
                    and event.occurrence_token in {ack.token for ack in self.ledger.acknowledgements}
                    for event in self.old.pending), 'terminal_saved_unapplied_ack')
                self.old.pending.clear()
                continue
            check(row['kind'] in ('stable_terminal', 'nonstable_terminal'), 'terminal_saved_unknown_record')
            observation_count += 1
            decision = self.old.decisions[frame]
            if row['kind'] == 'stable_terminal':
                check(decision['reason'] is None, 'terminal_saved_false_stable')
                board = self.old.engine.B.Board.from_dict({'grid': decision['raw']})
                evidence = self.lane.observe(self.ledger, frame, board, decision['source_call_token'])
                check(row['evidence'] == normalized(evidence) and self.lane.prepare(self.ledger,
                    decision['source_call_token']) is None, 'terminal_saved_second_application')
            else:
                check(row['reason'] == decision['reason'] and row['reason'] is not None, 'terminal_saved_hold_reason')
        check(observation_count == 1 and notice_count == int(bool(notices))
              and not self.old.pending, 'terminal_saved_frame_coverage')

    def finish(self, prefix_finish: dict, end: int) -> dict:
        check, packet = self.check, self.old.packet
        check(self.handoff is not None and self.terminal[-1]['kind'] == 'finish', 'terminal_saved_finish_missing')
        finish = self.terminal[-1]
        self.used_terminal.add(len(self.terminal)-1)
        check(self.used_terminal == set(range(len(self.terminal))), 'terminal_saved_unconsumed_records')
        check(finish['committed'] is True and finish['ledger'] == normalized(asdict(self.ledger))
            and finish['current'] == finish['planned_state'] == self.old.S.encode(self.old.current),
            'terminal_saved_final_state')
        unacked = [a.token for a in self.ledger.arrivals if a.token not in {ack.token for ack in self.ledger.acknowledgements}]
        check(finish['unacknowledged'] == unacked and self.ledger.clock == end, 'terminal_saved_final_ack')
        check(prefix_finish == normalized(dict(kind='second_prefix_finish',
            **self.old.origin_adapter.snapshot())), 'terminal_saved_old_finish')
        check(packet['closed'] and packet['error'] is None and packet['retired'] is None
            and packet['current'] == finish['current'], 'terminal_saved_packet_finish')
        return dict(handoff_frame=self.handoff, end=end, arrivals=len(self.ledger.arrivals),
            applied=len(self.ledger.applied), acknowledged=len(self.ledger.acknowledgements),
            unacknowledged_terminal_tokens=unacked, historical_prefix_verified=True,
            conditional_terminal_replayed=True, quality_gate_clear=False)

    def run(self, records: list[dict], end: int) -> dict:
        check, initial = self.check, self.old.current
        check(records[-1]['kind'] == 'second_prefix_finish', 'terminal_saved_prefix_finish')
        check(all(initial.frame < frame <= end and (frame-initial.frame) % STRIDE == 0
                  for frame in self.old.receipts), 'terminal_saved_receipt_bounds')
        first = self.old.original.step_for(self.old.steps, initial.frame, list(initial.scope))
        check(first['token'] == self.old.packet['initial']['source_call_token'], 'terminal_saved_initial_J')
        groups, previous = {}, initial.frame
        for row in records[:-1]:
            frame = row.get('frame', row.get('scope', {}).get('frame_idx', row.get('receipt', {}).get('applied_frame')))
            check(type(frame) is int and initial.frame < frame <= end and frame >= previous
                  and (frame-initial.frame) % STRIDE == 0, 'terminal_saved_order')
            groups.setdefault(frame, []).append(row)
            previous = frame
        for frame in range(initial.frame + STRIDE, end + STRIDE, STRIDE):
            step = self.old.original.step_for(self.old.steps, frame, list(initial.scope))
            event = self.old.native.extract(dict(token=step['token'], scope=dict(frame_idx=frame), events=step['events']))
            if event is not None:
                check(event.occurrence_token not in self.old.seen, 'terminal_saved_duplicate_ack')
                self.old.seen.add(event.occurrence_token)
                self.old.pending.append(event)
            if self.ledger is not None:
                self.advance_source(frame)
            for row in groups.get(frame, ()):
                if row['kind'] == 'second_prefix_handoff':
                    self.accept_handoff(row, step)
                elif self.handoff is None:
                    self.old_record(row, step)
                else:
                    check(row['kind'] == 'prefix-stable-decision/v1', 'terminal_saved_old_lane_resumed')
                    self.old.record(row, step)
            receipt = self.old.receipts.get(frame)
            if receipt is not None and receipt.get('kind') not in ('second_consumed_prefix/v1', D.VERSION):
                self.old.ordinary(receipt, step)
            elif receipt is not None:
                check(frame in self.old.commits or frame == self.handoff, 'terminal_saved_missing_commit')
            if self.handoff is not None and frame > self.handoff:
                self.terminal_frame(frame)
            self.pending_timeline[frame] = tuple(event.occurrence_token for event in self.old.pending)
        return self.finish(records[-1], end)
