"""観測後の終端着弾を一回反映し、遅い原ACKを別に保持する私有接続。"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
import observed_terminal_drop as D
import warning_source as W

VERSION = D.VERSION


class Terminal:
    def __init__(self, binding: Any, services: Any, prepared: Any,
                 drop: dict, path: Path) -> None:
        self.binding, self.services, self.prepared = binding, services, prepared
        self.mode = binding.mode
        self.lane = services.lane.Lane(services.parts, prepared.following,
            prepared.ledger_after, '予告陽性と観測後の終端30着弾に条件付け。較正済み保証ではない。')
        self.receipt = prepared.receipt | dict(kind=VERSION, drop=drop,
            applied_without_ack_tokens=[token for token in prepared.ledger_after.applied
                if token not in {a.token for a in prepared.ledger_after.acknowledgements}])
        self.stream = path.open('x', encoding='utf-8')
        self.committed, self.closed = False, False

    def write(self, kind: str, **values: Any) -> None:
        self.binding.L.require(not self.closed, 'terminal_recovery_closed')
        self.stream.write(json.dumps(dict(kind=kind, **values,
            original_fifo_changed=False, quality_gate_clear=False), allow_nan=False) + '\n')
        self.stream.flush()

    def commit(self) -> None:
        mode, before, check = self.mode, self.prepared, self.binding.L.require
        check(not self.committed and self.binding.ledger is before.ledger_before,
              'terminal_recovery_commit_once')
        tokens = tuple(a.token for a in before.ledger_before.acknowledgements
                       if a.token not in before.ledger_before.applied)
        check(tuple(e.occurrence_token for e in mode.native.pending) == tokens,
              'terminal_recovery_native_ack_prefix')
        self.write('prepared', receipt=self.receipt,
            old_prefix_snapshot=mode.prefix.snapshot(), old_ledger=asdict(self.binding.ledger))
        connection = mode.connection
        following, receipt = connection.registry.transition(connection.recovery.factory,
            connection.binding, before.expected, before.receipt['source_call_token'],
            lambda current: (before.following, self.receipt))
        check(following is before.following and receipt is self.receipt, 'terminal_recovery_registry_result')
        self.binding.ledger = before.ledger_after
        del mode.native.pending[:len(tokens)]
        mode.applied.append(self.receipt)
        self.committed = True
        self.write('committed', receipt=self.receipt, ledger=asdict(self.binding.ledger),
                   removed_private_native_ack_tokens=tokens)
        mode.prefix.write(dict(kind='second_prefix_handoff', frame=following.frame,
            source_call_token=before.receipt['source_call_token'], terminal_receipt=self.receipt,
            historical_lane_retained=True, **mode.prefix.snapshot()))

    def capture(self, item: dict) -> None:
        check, ledger = self.binding.L.require, self.binding.ledger
        check(self.committed and not self.closed, 'terminal_recovery_not_committed')
        check(tuple(a.token for a in ledger.arrivals) == self.lane.base,
              'terminal_recovery_new_arrival_before_reset')
        view, notices = self.services.notice.partition(item)
        check(all(event.get('active_origin') is None for event in view['events']),
              'terminal_recovery_new_fire_before_reset')
        pending = self.mode.native.pending
        known = {ack.token for ack in ledger.acknowledgements}
        check(all(event.occurrence_token in ledger.applied and event.occurrence_token in known
                  for event in pending), 'terminal_recovery_unapplied_ack')
        if notices:
            self.save_notice(item, notices)
        if pending:
            self.write('late_ack', frame=ledger.clock, events=[asdict(event) for event in pending],
                       ledger=asdict(ledger), physically_reapplied=False)
            pending.clear()

    def save_notice(self, item: dict, notices: list) -> None:
        """旧Session所有streamを閉じず、同一の通知票を旧検収先と終端票へ保存する。"""
        stream = getattr(self.binding, 'notice_stream', None)
        self.binding.L.require(stream is not None, 'terminal_recovery_notice_stream_missing')
        packet = self.services.notice.packet(self.mode, item, notices)
        stream.write(json.dumps(packet, allow_nan=False) + '\n')
        stream.flush()
        self.write('settled_notice', frame=self.binding.ledger.clock, notice=packet)

    def progress(self, item: dict, result: Any, row: dict) -> dict:
        mode, check = self.mode, self.binding.L.require
        check(mode.connection.registry.current(mode.connection.binding) is self.lane.current,
              'terminal_recovery_current_owner')
        observed, reason = mode.stable(item, result)
        frame = item['scope']['frame_idx']
        if reason is None:
            evidence = self.lane.observe(self.binding.ledger, frame, observed, item['token'])
            check(self.lane.prepare(self.binding.ledger, item['token']) is None,
                  'terminal_recovery_unexpected_second_application')
            self.write('stable_terminal', frame=frame, evidence=evidence)
        else:
            self.write('nonstable_terminal', frame=frame, reason=reason)
        return row | dict(reason='terminal_current:'+str(reason or 'unchanged'),
                          provisional_update=False, pending_occurrences=[])

    def close(self) -> None:
        if self.closed:
            return
        try:
            current = self.mode.connection.registry.current(self.mode.connection.binding)
            self.write('finish', committed=self.committed, ledger=asdict(self.binding.ledger),
                current=self.mode.prefix.serializer.encode(current),
                planned_state=self.mode.prefix.serializer.encode(self.prepared.following),
                unacknowledged=[a.token for a in self.binding.ledger.arrivals
                    if a.token not in {ack.token for ack in self.binding.ledger.acknowledgements}])
        finally:
            self.stream.close()
            self.closed = True


def recover(binding: Any, services: Any, item: dict, result: Any, row: dict,
            original_error: BaseException, path: Path) -> dict:
    mode, check = binding.mode, binding.L.require
    try:
        cached = mode._second_stable_cache
        check(cached[0] is item and cached[1] is result and cached[3] is None,
              'terminal_recovery_same_qualified_call')
        current = mode.connection.registry.current(mode.connection.binding)
        check(mode.prefix.lane.current is current and mode.prefix.lane.last_frame < binding.ledger.clock
              and getattr(binding, 'terminal', None) is None, 'terminal_recovery_unchanged_failure')
        evidence = W.select(check, tuple(binding.warning.rows), current.scope,
                            current.frame, binding.ledger.clock)
        family, drop = D.candidate(services.parts, services.commit, current, binding.ledger,
                                    cached[2], evidence)
        prepared = services.commit.prepare(services.parts, current, binding.ledger,
            binding.ledger.applied, (family,), binding.ledger.clock, item['token'],
            'warning-source:' + item['token'])
        terminal = Terminal(binding, services, prepared, drop, path)
        binding.terminal = terminal  # commit失敗時も所有側closeで準備票を保存・解放する。
        terminal.commit()
        return row | dict(reason=VERSION, transition=terminal.receipt,
                          provisional_update=True, pending_occurrences=[])
    except BaseException as error:
        binding.failure = error
        binding.save(dict(kind='terminal_recovery_rejected', original_error=repr(original_error),
                          recovery_error=repr(error), frame=item['scope']['frame_idx']))
        raise original_error from error
