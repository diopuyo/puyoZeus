"""STABLE中に繰り返し見えた可視盤を、物理予測の追加候補として保持する。"""
from __future__ import annotations
from collections import defaultdict, deque
import numpy as np
from src.light_likelihood import MIN_FRAMES, MIN_WINDOW_SECONDS

HISTORY_SECONDS = 4.
VISIBLE_START = 1
WINDOW_TIME_EPSILON = 1e-7


def supported(board: np.ndarray) -> bool:
    """可視の色の下に空がある盤は、落下中なので追加しない。"""
    occupied = board[VISIBLE_START:] != 0
    return not np.any(occupied[:-1] & ~occupied[1:])


class ObservedBases:
    """確定盤を書き換えず、繰り返し見えた支持のある実観測を返す。"""
    def __init__(self) -> None:
        self.records: deque[tuple[float, np.ndarray]] = deque()

    def remember(self, t: float, raw: np.ndarray | None) -> None:
        while self.records and self.records[0][0] < t-HISTORY_SECONDS:
            self.records.popleft()
        if raw is None or (self.records and t <= self.records[-1][0]):
            return
        board = raw.copy()
        board[0] = 0
        if supported(board):
            self.records.append((t, board))

    def variants(self, t: float, base: np.ndarray,
                 allow_color_conflicts: bool = False) -> list[np.ndarray]:
        """色の衝突は通常拒否。最初の式に合う通常候補が0件の時だけ許す。

        許可時も6回・0.25秒・過去4秒・未来でない・支えの条件は同じ。
        既知色から空への違いも衝突に含め、観測盤全体を独立の候補とする。
        """
        result = [base.copy()]
        groups: dict[bytes, list[tuple[float, np.ndarray]]] = defaultdict(list)
        for stamp, board in self.records:
            consistent = allow_color_conflicts or np.all(board[base!=0] == base[base!=0])
            if t-HISTORY_SECONDS <= stamp <= t and consistent:
                groups[board.tobytes()].append((stamp,board))
        for records in groups.values():
            if len(records) < MIN_FRAMES or records[-1][0]-records[0][0]+WINDOW_TIME_EPSILON < MIN_WINDOW_SECONDS:
                continue
            board = records[0][1]
            if not np.array_equal(board,base):
                result.append(board.copy())
        return result
