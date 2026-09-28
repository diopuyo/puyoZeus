"""撃ち合いの区切りに依存しない未消化予告。改訂時は履歴を再計算する。"""
from __future__ import annotations

AttackKey = tuple[int, int]
AttackTotals = dict[AttackKey, tuple[int, bool]]
Chunk = tuple[AttackKey, int]


def consume(chunks: list[Chunk], amount: int) -> int:
    """FIFOで既存予告だけを消費し、余剰の着地を将来の攻撃へ繰り越さない。"""
    while chunks and amount > 0:
        key, pending = chunks[0]
        used = min(pending, amount)
        amount -= used
        if used == pending:
            chunks.pop(0)
        else:
            chunks[0] = (key, pending-used)
    return amount


class PendingLedger:
    """同一連鎖の累積通知は差分化し、下方改訂は過去の相殺も含めて巻き戻す。"""

    def __init__(self) -> None:
        self.game: int | None = None
        self.events: list[tuple[str, AttackKey, int]] = []
        self.totals: AttackTotals = {}
        self.dropped = (0, 0)
        self.chunks: list[list[Chunk]] = [[], []]

    def observe(self, game: int, totals: AttackTotals, dropped: tuple[int, int]) -> None:
        """試合境界だけで全消去し、交換終了では台帳を維持する。"""
        if game != self.game:
            self.game, self.events, self.totals = game, [], {}
            self.dropped, self.chunks = dropped, [[], []]
        changed = totals != self.totals
        for key, (amount, _) in totals.items():
            if amount != self.totals.get(key, (0, False))[0]:
                self.events.append(('send', key, max(0, int(amount))))
        for side, (before, after) in enumerate(zip(self.dropped, dropped)):
            if after > before:
                self.events.append(('drop', (side, 0), after-before))
                changed = True
        self.totals, self.dropped = dict(totals), dropped
        if changed:
            self._rebuild()

    def _rebuild(self) -> None:
        """最新の総量を上限に履歴を再生し、過去の過大予測を差分で二重控除しない。"""
        self.chunks, credited = [[], []], {}
        for kind, key, amount in self.events:
            side = key[0]
            if kind == 'drop':
                consume(self.chunks[side], amount)
                continue
            cumulative = min(amount, self.totals.get(key, (0, False))[0])
            delta = max(0, cumulative-credited.get(key, 0))
            credited[key] = max(cumulative, credited.get(key, 0))
            surplus = consume(self.chunks[side], delta)
            if surplus:
                self.chunks[1-side].append((key, surplus))

    @property
    def pending(self) -> list[int]:
        """各受け側の未消化予告を上限なしで返す。"""
        return [sum(n for _, n in chunks) for chunks in self.chunks]

    def verified(self, receiver: int) -> bool:
        """生き残った予告の送り側根拠を確認し、未裏付け予測を死亡根拠にしない。"""
        return all(self.totals[key][1] for key, _ in self.chunks[receiver])
