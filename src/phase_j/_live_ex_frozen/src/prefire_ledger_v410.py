"""v4.10試作の時刻付き会計。将来の仮説と観測済みの事実を混同しない。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable
from src.phase_j._live_ex_frozen.src.scoring import OJAMA_MAX_DROP_PER_TURN

SIDES = (0, 1)
PAIR_CANCEL_COST = 2


@dataclass
class Batch:
    """受け側の予告。確定時刻までは着弾できない。"""
    side: int
    amount: int
    source: str
    generated_at: float
    confirmed_at: float | None = None


@dataclass
class Ledger:
    """生成 = 相殺の両側 + 着弾 + 残予告を各更新で検査する。"""
    batches: list[Batch] = field(default_factory=list)
    initial: int = 0
    generated: list[int] = field(default_factory=lambda: [0, 0])
    cancelled: int = 0
    dropped: list[int] = field(default_factory=lambda: [0, 0])
    trace: list[dict] = field(default_factory=list)

    @classmethod
    def from_batches(cls, batches: Iterable[Batch]) -> Ledger:
        items = [replace(b) for b in batches]
        if any(b.amount < 0 or b.side not in SIDES for b in items):
            raise ValueError('予告の側・個数が不正')
        return cls(items, sum(b.amount for b in items))

    def copy(self) -> Ledger:
        return Ledger([replace(b) for b in self.batches], self.initial,
                      list(self.generated), self.cancelled, list(self.dropped), list(self.trace))

    def pending(self, side: int) -> int:
        return sum(b.amount for b in self.batches if b.side == side)

    def _consume(self, side: int, amount: int, cutoff: float | None = None) -> int:
        """相殺は新しい予告から。着弾は確定時刻の条件を満たすものだけ。"""
        left = amount
        for batch in reversed(self.batches):
            if batch.side != side:
                continue
            if cutoff is not None and (batch.confirmed_at is None or batch.confirmed_at > cutoff):
                continue
            take = min(left, batch.amount)
            batch.amount -= take
            left -= take
        self.batches = [b for b in self.batches if b.amount]
        return amount-left

    def generate(self, amounts: tuple[int, int], stamp: float, sources: tuple[str, str]) -> None:
        """同時刻の両側を原子的に相殺する。側の処理順で結果を変えない。"""
        if any(n < 0 for n in amounts):
            raise ValueError('負の生成')
        own = [self._consume(i, amounts[i]) for i in SIDES]
        left = [amounts[i]-own[i] for i in SIDES]
        cross = min(left)
        self.cancelled += sum(own)+cross
        for side in SIDES:
            self.generated[side] += amounts[side]
            surplus = left[side]-cross
            if surplus:
                self.batches.append(Batch(1-side, surplus, sources[side], stamp))
        self.trace.append(dict(kind='generate', t=stamp, amounts=list(amounts),
                               cancelled=sum(own)+cross, sent=[n-cross for n in left]))
        self.check()

    def confirm(self, source: str, stamp: float) -> None:
        for batch in self.batches:
            if batch.source == source and batch.confirmed_at is None:
                batch.confirmed_at = stamp
        self.trace.append(dict(kind='confirm', source=source, t=stamp))

    def land(self, side: int, stamp: float, trigger: float | None = None) -> int:
        """連鎖終了では発火前に確定した分だけ、非発火の設置では現在の確定分を落とす。"""
        cutoff = stamp if trigger is None else trigger
        count = self._consume(side, OJAMA_MAX_DROP_PER_TURN, cutoff)
        self.dropped[side] += count
        self.trace.append(dict(kind='land', side=side, t=stamp, count=count, cutoff=cutoff))
        self.check()
        return count

    def check(self) -> None:
        accounted = PAIR_CANCEL_COST*self.cancelled+sum(self.dropped)+sum(b.amount for b in self.batches)
        if self.initial+sum(self.generated) != accounted:
            raise AssertionError('おじゃま保存則違反')

    def audit(self) -> dict:
        self.check()
        return dict(initial=self.initial, generated=list(self.generated), cancelled=self.cancelled,
                    dropped=list(self.dropped), pending=[self.pending(i) for i in SIDES],
                    residual=0, events=list(self.trace))
