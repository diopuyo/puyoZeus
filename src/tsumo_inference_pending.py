"""上段の新しい対を、確定盤を変えず評価盤だけで仮扱いにする。"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import Board, HIDDEN_ROWS
from src.tsumo_rollback import (
    Cell, PAIR_SIZE, TOP_ROWS, SAFE_STATES, COLOR_MIN, COLOR_MAX, _additions,
)

MAX_PENDING_SEC = 1.0
TIME_EPSILON_SEC = 1e-9
COLOR_UNKNOWN = 10
ROTATION_CONFIRM_FRAMES = 2
ADJACENT_CELL_DISTANCE = 1
CONTINUATION_STATES = frozenset(("ojama_fall", "chain", "gravity_settle", "effect"))


def _is_visible_swap(cells: tuple[Cell, ...], observed: Board) -> bool:
    if len(cells) != PAIR_SIZE:
        return False
    (r0, c0, _), (r1, c1, _) = cells
    if abs(r0 - r1) + abs(c0 - c1) != ADJACENT_CELL_DISTANCE:
        return False
    colors = [v for _, _, v in cells]
    actual = [observed.get(r, c) for r, c, _ in cells]
    return (all(r >= HIDDEN_ROWS for r, _, _ in cells)
            and colors != actual and sorted(colors) == sorted(actual))


@dataclass
class ProvisionalPair:
    cells: tuple[Cell, ...]
    started_sec: float
    discarded: bool = False
    replaced: bool = False
    rotation_frames: int = 0
    rotated: bool = False


class InferencePendingPair:
    """NEXT通知までは除き、消えた対を期限後に復活させない。"""

    def __init__(self) -> None:
        self.previous: Board | None = None
        self.pairs: list[ProvisionalPair] = []
        self.last_sec: float | None = None

    def reset(self) -> None:
        self.previous = None
        self.pairs.clear()
        self.last_sec = None

    def _advance(self, confirmed: Board, observed: Board,
                 time_sec: float, next_shift: bool) -> None:
        remaining = []
        for pair in self.pairs:
            if pair.rotated:
                # 置き換わった升を追跡から外し、同色の新しいぷよを再び除外しない。
                pair.cells = tuple((r, c, v) for r, c, v in pair.cells
                                   if confirmed.get(r, c) == v)
            if any(confirmed.get(r, c) not in (0, v, COLOR_UNKNOWN) for r, c, v in pair.cells):
                pair.replaced = True
            if not any(confirmed.get(r, c) == v for r, c, v in pair.cells):
                continue
            visible = all(observed.get(r, c) == v for r, c, v in pair.cells)
            swapped = _is_visible_swap(pair.cells, observed)
            pair.rotation_frames = pair.rotation_frames + 1 if swapped else 0
            pair.rotated |= pair.rotation_frames >= ROTATION_CONFIRM_FRAMES
            if any(observed.get(r, c) not in (v, COLOR_UNKNOWN) for r, c, v in pair.cells):
                pair.discarded = True
            elif visible and not pair.rotated:
                # 見え直した対も最初の期限を使い、観測の揺れで待機を延ばさない。
                pair.discarded = False
            expired = time_sec - pair.started_sec >= MAX_PENDING_SEC - TIME_EPSILON_SEC
            if pair.rotated:
                pair.discarded = True
            if not pair.discarded and (next_shift or expired):
                continue
            remaining.append(pair)
        self.pairs = remaining

    def _track_addition(self, confirmed: Board, observed: Board,
                        time_sec: float, next_shift: bool) -> None:
        # 同じコマのNEXTは既存の仮の対だけを確かめる。今回現れた対には流用しない。
        if self.previous is None:
            return
        cells = _additions(self.previous, confirmed)
        if len(cells) != PAIR_SIZE or not any(r < TOP_ROWS for r, _, _ in cells):
            return
        discarded = any(observed.get(r, c) not in (v, COLOR_UNKNOWN) for r, c, v in cells)
        self.pairs.append(ProvisionalPair(cells, time_sec, discarded))

    def project(
        self, confirmed: Board | None, observed: Board, time_sec: float,
        *, published: Board | None, next_shift: bool, active: bool,
        state: str = 'stable',
    ) -> Board | None:
        """変更が必要な時だけコピーを返す。Noneは従来の評価盤を使う意味。"""
        if state not in SAFE_STATES:
            if state not in CONTINUATION_STATES:
                self.reset()
                return None
            # 回転を確認した元の対、またはおじゃま中の片割れだけを除く。
            self.pairs = [p for p in self.pairs if p.discarded and (p.rotated or not p.replaced)
                          and confirmed is not None
                          and (p.rotated or (state == 'ojama_fall'
                               and sum(confirmed.get(r, c) == v for r, c, v in p.cells) == 1))]
            self.previous = None
        if not active or (self.last_sec is not None and time_sec < self.last_sec):
            self.reset()
        self.last_sec = time_sec
        if not active or confirmed is None:
            self.reset()
            return None
        self._advance(confirmed, observed, time_sec, next_shift)
        self._track_addition(confirmed, observed, time_sec, next_shift)
        self.previous = confirmed.copy()
        if published is None:
            return None
        result = None
        for pair in self.pairs:
            for r, c, color in pair.cells:
                # 公開時の色補正で同じ操作対が除外をすり抜けないよう、元盤で同一性を確かめる。
                if confirmed.get(r, c) == color and COLOR_MIN <= published.get(r, c) <= COLOR_MAX:
                    if result is None:
                        result = published.copy()
                    result.set(r, c, 0)
        return result
