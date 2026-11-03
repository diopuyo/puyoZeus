"""画面の式から現在段を読む表示専用状態。B3の学習状態には接続しない。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.scoring import chain_power, color_bonus, connection_bonus

MIN_GROUP = 4
MAX_COLORS = 4
MAX_CHAIN = 19
BOARD_CELLS = 78
SCORE_FACTOR = 10
CACHE_SIZE = 4096
CONFIRM_GAP_SEC = 0.5
END_QUIET_SEC = 0.5
QUIET_STATES = frozenset(('STABLE', 'TSUMO_FALL', 'OJAMA_FALL'))
SIDES = ('p1', 'p2')


@lru_cache(maxsize=BOARD_CELLS + 1)
def _bonuses(erased: int) -> frozenset[tuple[int, int]]:
    """消去数の分割で可能な連結ボーナスと群数を列挙する。"""
    if erased == 0:
        return frozenset({(0, 0)})
    return frozenset((bonus + connection_bonus(size), groups + 1)
        for size in range(MIN_GROUP, erased + 1)
        for bonus, groups in _bonuses(erased - size))


@lru_cache(maxsize=CACHE_SIZE)
def formula_stage(left: int, right: int) -> int | None:
    """画面式の段が一意の場合だけ返す。観測件数や予測長では補完しない。"""
    if left % SCORE_FACTOR or not MIN_GROUP * SCORE_FACTOR <= left <= BOARD_CELLS * SCORE_FACTOR:
        return None
    bonuses = {bonus + color_bonus(colors) for bonus, groups in _bonuses(left // SCORE_FACTOR)
               for colors in range(1, min(groups, MAX_COLORS) + 1)}
    candidates = [n for n in range(1, MAX_CHAIN + 1)
                  if any(max(1, chain_power(n) + bonus) == right for bonus in bonuses)]
    return candidates[0] if len(candidates) == 1 else None


@dataclass
class ScreenStage:
    """式の連続確認と欠読保持。内部fire_completeではリセットしない。"""
    value: int | None = None
    observed_at: float | None = None
    candidate: tuple[int, int] | None = None
    candidate_t: float = -float('inf')
    quiet_since: float | None = None

    def update(self, raw: dict, t: float) -> None:
        formula = raw.get('formula') or {}
        left, right = formula.get('left'), formula.get('right')
        key = (left, right) if formula.get('valid') and left and right else None
        number = formula_stage(*key) if key else None
        if number is not None:
            if key == self.candidate and 0 < t - self.candidate_t <= CONFIRM_GAP_SEC:
                self.value, self.observed_at = number, t
            self.quiet_since = None
        elif raw.get('state') in QUIET_STATES and key is None:
            if self.quiet_since is None:
                self.quiet_since = t
            if t - self.quiet_since >= END_QUIET_SEC:
                self.value, self.observed_at = None, None
        else:
            self.quiet_since = None
        self.candidate, self.candidate_t = key, t


class ChainDisplay:
    """公開時に最新観測を追加し、遅延した推論結果による巻戻しを防ぐ。"""

    def __init__(self) -> None:
        self.sides = [ScreenStage(), ScreenStage()]
        self.frame: int | None = None
        self.t_sec: float | None = None

    def observe(self, frame: dict) -> None:
        self.frame, self.t_sec = int(frame['frame']), float(frame['t_sec'])
        if not frame['match_active'] or frame.get('match_end_locked', False):
            self.sides = [ScreenStage(), ScreenStage()]
            return
        for tracker, raw in zip(self.sides, frame['sides']):
            tracker.update(raw, self.t_sec)

    def decorate(self, contents: dict) -> dict:
        result = dict(contents, chain_display_frame=self.frame, chain_display_t_sec=self.t_sec)
        for side, tracker in zip(SIDES, self.sides):
            result[side] = dict(contents[side], chain_display_step=tracker.value,
                                chain_display_observed_at_sec=tracker.observed_at)
        return result
