"""手に持つ組と NEXT/NEXT2 を、記録の NEXT の揺れとずれから切り離して決める (2026-10-01、Phase 4)。

記録の NEXT の意味 (exev logs/next_shift/RESULT.md、5記録の実測):
- 同じ確定盤面が続く区間 (STABLE 区間) の先頭約0.25秒は、next = 手に持つ組 P_k、dnext = P_{k+1} (繰り上がり前)。
- その後 NEXT が繰り上がり、next = P_{k+1}、dnext = P_{k+2} になる。P_k は画面から消える。
- 連鎖後・おじゃま後は、区間の先頭から既に繰り上がっていることが多い。
- 未読 (0 や 9) と、A→B→A のちらつきがある。

規則 (状態は外部 wrapper のここにだけ持つ):
1. 読みは STABLE_FRAMES フレーム続けて同じで、4色すべて実色のときだけ採用する (未読・ちらつきでは変えない)。
2. 区間の手に持つ組 = 1つ前の区間の最後の採用読みの next。1つ前の区間に採用読みがなければ不明。
3. 今の区間の最新の採用読みが繰り上がり後なら (P_k, next, dnext) の3組、繰り上がり前なら (P_k, dnext) の2組。
   繰り上がり後と判断する条件: 区間内で採用読みが2つ以上ある / 読みの next が P_k と違う /
   採用した時刻が区間先頭から PROMOTION_SEC 以上後。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STABLE_FRAMES = 3            # 採用に要する連続フレーム数 (30fps で 0.1 秒。採点前に固定)
PROMOTION_SEC = 0.3          # 繰り上がりが済む区間先頭からの時間 (RESULT.md: 0.3秒以降は 78〜83% が繰り上がり後)
PLAYABLE = (1, 2, 3, 4, 5)
PAIR = 2
UNKNOWN_PAIR = (0, 0)


def _valid(reading: tuple[int, ...]) -> bool:
    """4色すべて実色の読みか (0・9 は未読の番兵)。"""
    return len(reading) == 2 * PAIR and all(int(c) in PLAYABLE for c in reading)


@dataclass
class _Interval:
    """1つの STABLE 区間の観測。"""

    board: bytes
    start: float
    adopted: list[tuple[float, tuple[int, ...]]] = field(default_factory=list)
    in_hand: tuple[int, int] | None = None

    def promoted(self) -> bool:
        """最新の採用読みが繰り上がり後か (モジュール docstring の規則3)。"""
        if not self.adopted:
            return False
        when, reading = self.adopted[-1]
        return (len(self.adopted) >= 2 or (self.in_hand is not None and tuple(reading[:PAIR]) != self.in_hand)
                or when - self.start >= PROMOTION_SEC)

    def next_in_hand(self) -> tuple[int, int] | None:
        """次の区間で手に持つ組 (規則2) = この区間の最後の採用読みの next。

        区間の終わりには通常 NEXT が繰り上がり済みなので next = 次に置く組。繰り上がり前と判断して dnext を
        使う版より、5記録の入力の正しさが高かった (手に持つ組 79.2% → 89.4%、scripts/prefire_stable_queue_accuracy)。
        """
        if not self.adopted:
            return None
        return tuple(self.adopted[-1][1][:PAIR])


@dataclass
class SideQueue:
    """片側の区間と読みの追跡。observe に STABLE 履歴を順に渡す。"""

    interval: _Interval | None = None
    run: tuple[int, ...] | None = None
    run_count: int = 0

    def observe(self, t_sec: float, board: bytes, queue: np.ndarray) -> None:
        """STABLE 履歴の1件を取り込む。"""
        reading = tuple(int(v) for v in queue)
        if self.interval is None or board != self.interval.board:
            previous = self.interval
            self.interval = _Interval(board, t_sec)
            self.interval.in_hand = previous.next_in_hand() if previous is not None else None
            self.run, self.run_count = None, 0
        self.run_count = self.run_count + 1 if reading == self.run else 1
        self.run = reading
        adopted = self.interval.adopted
        if self.run_count == STABLE_FRAMES and _valid(reading) and (not adopted or adopted[-1][1] != reading):
            adopted.append((t_sec, reading))

    def known(self) -> tuple[int, ...]:
        """(手に持つ組, NEXT, NEXT2) の6色。不明な組は 0。手に持つ組か読みが無ければ全て 0。"""
        interval = self.interval
        if interval is None or interval.in_hand is None or not interval.adopted:
            return (0,) * (3 * PAIR)
        reading = interval.adopted[-1][1]
        if interval.promoted():
            return (*interval.in_hand, *reading)
        return (*interval.in_hand, *reading[PAIR:], *UNKNOWN_PAIR)

    def reading(self) -> np.ndarray | None:
        """S3 の入力に渡す NEXT/NEXT2 (最新の採用読み。本番の history[-1].queue のちらつきを除いたもの)。"""
        if self.interval is None or not self.interval.adopted:
            return None
        return np.asarray(self.interval.adopted[-1][1])


class StableQueues:
    """両側の SideQueue を overlay の履歴から増分で更新する (試合が変われば作り直す)。"""

    def __init__(self) -> None:
        self.sides = [SideQueue(), SideQueue()]
        self._seen = [0, 0]
        self._game: int | None = None

    def update(self, overlay) -> None:
        """overlay._history の新しい項目だけを取り込む。"""
        if overlay._game != self._game or any(len(h) < n for h, n in zip(overlay._history, self._seen)):
            self.sides, self._seen, self._game = [SideQueue(), SideQueue()], [0, 0], overlay._game
        for idx, history in enumerate(overlay._history):
            for item in history[self._seen[idx]:]:
                self.sides[idx].observe(item.t_sec, np.asarray(item.board._grid, dtype=np.int8).tobytes(), item.queue)
            self._seen[idx] = len(history)
