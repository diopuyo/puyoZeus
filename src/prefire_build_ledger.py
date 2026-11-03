"""v4.8と同じ相殺・着弾器を、観測台帳のコピーにだけ適用する。"""
from __future__ import annotations
from typing import Any
from src.b3_score_ledger import ScoreLedgerV2, _Batch, MODE_TIMING
from src.prefire_pending_v47 import fork_ledger, attack


class BuildLedger:
    def __init__(self, pending: list, core: Any = None, origin: float = 0.) -> None:
        self.core = fork_ledger(core) if core is not None else ScoreLedgerV2(
            mode=MODE_TIMING, newest_first_offset=True)
        self.landed = [0, 0]
        if core is None:
            for side, amount in enumerate(pending):
                if amount:
                    self.core.batches[side].append(_Batch(int(amount),1-side,-1,True,float('-inf')))
        else:
            for batches in self.core.batches:
                for batch in batches:
                    if batch.confirmed_t is not None:
                        batch.confirmed_t -= origin
                    batch.gen_t -= origin

    def send(self, side: int, amount: int, key: tuple, ready: float) -> None:
        count = len(self.core.batches[1-side])
        attack(self.core, side, amount, ready, key[1])
        if len(self.core.batches[1-side]) > count:
            batch = self.core.batches[1-side][-1]
            batch.confirmed, batch.confirmed_t = True, ready

    def pending(self, side: int, stamp: float = float('inf')) -> int:
        if stamp == float('inf'):
            return self.core._pending(side)
        return sum(b.n for b in self.core.batches[side] if b.confirmed
                   and b.confirmed_t is not None and b.confirmed_t <= stamp)

    def drop(self, side: int, stamp: float) -> int:
        count = self.core._land(side, stamp, stamp)
        self.landed[side] += count
        return count
