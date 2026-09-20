"""終了前の元公開検査を保持し、終了後全updateの非干渉を別に検査する。"""
from __future__ import annotations
from typing import Any
import terminal_boundary as C


def check(original: Any, mode: Any, consumer: list, history: list, journal: list,
          recovery: list, tracking: list, **kwargs: Any) -> dict:
    capture = mode.arrival_capture
    if getattr(capture, 'terminal_receipt', None) is None:
        return original.check(consumer, history, journal, recovery, tracking, **kwargs)
    ledger = mode.arrival_ledger
    C.require(kwargs['end_frame'] == ledger.clock == mode.native.last_frame
        and capture.terminal_last == ledger.deadline and capture.pending is None,
        'terminal_publication_end_owner')
    tail = [row for row in consumer if row['frame_idx'] > ledger.clock]
    frames = list(range(ledger.clock+C.STRIDE, ledger.deadline+C.STRIDE, C.STRIDE))
    C.require(frames and [row['frame_idx'] for row in tail] == frames
        and capture.terminal_rows == len(frames), 'terminal_publication_coverage')
    for row in tail:
        C.require(row['same_result_identity'] and row['comparison_completed']
            and not row['changed_sides'] and row['full_before'] == row['full_after']
            and row['tickets_this_update'] == row['released_this_update'] == 0,
            'terminal_publication_noninterference')
    prefix = [row for row in consumer if row['frame_idx'] <= ledger.clock]
    result = original.check(prefix, history, journal, recovery, tracking, **kwargs)
    return result | dict(terminal_publication_updates=len(tail), terminal_end=ledger.deadline,
        terminal_original_journal_verified=False, quality_gate_clear=False)
