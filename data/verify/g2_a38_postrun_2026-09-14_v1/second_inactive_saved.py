"""元2P数学consumerの最後のlive状態と、同run終了tailの全原Jを別々に検査する。"""
from __future__ import annotations
from types import SimpleNamespace as N
from typing import Any
import first_terminal_saved as V
import terminal_boundary as C


def derive(original: Any, packets: list, status: dict) -> type:
    class Replay(original.Replay):
        def run(self, records: list, end: int) -> dict:
            finish = self.terminal[-1]
            C.require(finish['kind'] == 'finish', 'second_inactive_saved_finish')
            last = finish['ledger']['clock']
            if last == end:
                C.require(not packets and not status, 'second_inactive_saved_unexpected_tail')
                return super().run(records, end)
            C.require(type(last) is int and last < end, 'second_inactive_saved_clock')
            result = super().run(records, last)  # 原finish/全会計/current/旧prefix検査を維持する。
            C.require(self.ledger.scope[-1] == '2P' and self.ledger.deadline == end and not self.old.pending,
                'second_inactive_saved_deadline_or_pending')
            raw = sorted([*self.source.values(), *self.old.steps.values()], key=lambda row: row['row_index'])
            adapter = N(L=self.L, R=N(N=self.old.native))
            tail = V.verify_records(adapter, self.ledger, raw, packets, self.old.contexts)
            C.require(status['closed'] is True and status['error'] is None and status['original_body'] is None
                and status['rows'] == len(packets) and status['last_frame'] == end
                and status['receipt'] == packets[0]['terminal_receipt'], 'second_inactive_saved_status')
            for frame in range(last+C.STRIDE, end+C.STRIDE, C.STRIDE):
                self.pending_timeline[frame] = ()  # 原Jの無消費検査後。未ACKは到来台帳へ保持。
            return result | dict(end=end, completed_live_prefix_end=last, inactive_terminal=tail)
    return Replay
