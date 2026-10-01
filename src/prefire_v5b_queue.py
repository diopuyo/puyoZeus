"""5B専用の組単位の安定読み。盤面境界で画面NEXTの観測履歴を捨てない。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.prefire_stable_queue import PLAYABLE, STABLE_FRAMES, UNKNOWN_PAIR, PROMOTION_SEC

KNOWN_SLOTS = 6


@dataclass
class PairReading:
    """画面の1組を追跡する。別の組の欠測には依存しない。"""

    run: tuple[int, ...] = UNKNOWN_PAIR
    count: int = 0
    accepted: tuple[int, ...] | None = None

    def observe(self, pair: tuple[int, ...]) -> None:
        """新しい安定読みでのみ更新する。未読は観測済みの組を消去しない。"""
        self.count = self.count + 1 if pair == self.run else 1
        self.run = pair
        if self.count >= STABLE_FRAMES and all(c in PLAYABLE for c in pair):
            self.accepted = pair


@dataclass
class SideQueueV5B:
    """STABLE確定盤面の手持ちと、画面上の2組を因果的に対応付ける。"""

    board: bytes | None = None
    hand: tuple[int, ...] | None = None
    pairs: list[PairReading] = field(default_factory=lambda: [PairReading(), PairReading()])
    shifted: bool = False
    last_next: tuple[int, ...] | None = None
    start: float = 0.0
    adopted: tuple | None = None
    adoptions: int = 0

    def observe(self, t_sec: float, board: bytes, queue: np.ndarray) -> None:
        """盤面境界では手持ちだけ更新し、継続中の画面読みの安定証拠は保つ。"""
        changed = self.board != board
        if changed:
            # exevの実測と5Aと同じ、前区間で最後に観測したNEXTを手持ちへ送る。
            self.hand = self.pairs[0].accepted if self.board is not None else None
            self.board, self.shifted, self.last_next = board, False, None
            self.start, self.adopted, self.adoptions = t_sec, None, 0
            # 新区間で未読なら前区間のNEXTを現在のNEXTへ再利用しない。
            # run/countだけを維持し、現在も同じ読みなら下で直ちに再確認する。
            for reading in self.pairs:
                reading.accepted = None
        old_next = self.pairs[0].accepted
        old_second = self.pairs[1].accepted
        for reading, offset in zip(self.pairs, (0, 2)):
            reading.observe(tuple(int(c) for c in queue[offset:offset+2]))
        first = self.pairs[0].accepted
        reading = tuple(p.accepted for p in self.pairs)
        if all(reading) and reading != self.adopted:
            self.adopted, self.adoptions = reading, self.adoptions + 1
            if self.adoptions >= 2 or t_sec - self.start >= PROMOTION_SEC:
                self.shifted = True
        if self.hand is not None and first is not None and first != self.hand:
            self.shifted = True
        if not changed and first is not None and old_second == first and old_next != first:
            self.shifted = True
        if first is not None:
            self.last_next = first

    def known(self) -> tuple[int, ...]:
        """観測していない組は0のまま。繰上げの時刻規則は新しい採用読みだけに適用する。"""
        first, second = (p.accepted or UNKNOWN_PAIR for p in self.pairs)
        if self.hand is None:
            return (0,) * KNOWN_SLOTS
        tail = (*first, *second) if self.shifted else (*second, *UNKNOWN_PAIR)
        return (*self.hand, *tail)

    def reading(self) -> np.ndarray:
        """側特徴向けにも、安定した組だけを渡す。"""
        return np.asarray([c for p in self.pairs for c in (p.accepted or UNKNOWN_PAIR)])


class StableQueuesV5B:
    """左右を独立更新し、試合境界で読みの証拠を破棄する。"""

    def __init__(self) -> None:
        self.sides = [SideQueueV5B(), SideQueueV5B()]
        self._seen = [0, 0]
        self._game: int | None = None

    def update(self, overlay: Any) -> None:
        """最新フレームまでのSTABLE履歴だけを消費する。"""
        if overlay._game != self._game or any(len(h) < n for h, n in zip(overlay._history, self._seen)):
            self.sides = [SideQueueV5B(), SideQueueV5B()]
            self._seen, self._game = [0, 0], overlay._game
        for index, history in enumerate(overlay._history):
            for item in history[self._seen[index]:]:
                self.sides[index].observe(item.t_sec, np.asarray(item.board._grid, dtype=np.int8).tobytes(), item.queue)
            self._seen[index] = len(history)
