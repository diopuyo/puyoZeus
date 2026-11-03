"""E21: 窒息セルを持つ確定盤面では新発火を消去実測まで保留する。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import numpy as np
from src.board import COLOR_EMPTY, COLOR_UNKNOWN, DEATH_COL, DEATH_ROW
from src.board_state_machine import BoardState
from src.chain import MIN_ERASE_COUNT
from src.scoring import calculate_chain_score

SIDES = ("1P", "2P")
COLOR_MIN, COLOR_MAX = 1, 5


def occupied(board: Any) -> bool:
    """未知色を占有の証拠にせず、隠し段を窒息判定へ含めない。"""
    return board is not None and int(board._grid[DEATH_ROW, DEATH_COL]) not in (COLOR_EMPTY, COLOR_UNKNOWN)


def observed_erasure(before: Any, after: Any) -> bool:
    """独立したSTABLE確定盤面で、同色が消去最小数以上減ったかを確認する。"""
    if before is None or after is None:
        return False
    if np.any(before._grid == COLOR_UNKNOWN) or np.any(after._grid == COLOR_UNKNOWN):
        return False
    return any(np.count_nonzero(before._grid == c)-np.count_nonzero(after._grid == c)
               >= MIN_ERASE_COUNT for c in range(COLOR_MIN, COLOR_MAX+1))


class DeathCandidateGate:
    """発火検出器自身の減少通知を、独立した消去の証拠として再利用しない。"""

    def __init__(self, simulator: Any) -> None:
        self.simulator = simulator
        self.audit: list[dict] = []
        self.reset()

    def reset(self) -> None:
        """試合を跨いで保留通知を再送しない。監査記録だけは保存する。"""
        self.boards: list[Any] = [None, None]
        self.board_sec = [float('-inf'), float('-inf')]
        self.candidates: list[dict | None] = [None, None]
        self.pending: list[dict[tuple, Any]] = [{}, {}]
        self.accepted: list[set[float]] = [set(), set()]
        self.dead: set[str] = set()

    def observe(self, sides: tuple, dead: tuple, stamp: float, game: int) -> None:
        """候補は最新のSTABLE確定盤面だけで開始・解除する。"""
        self.dead.update(dead)
        for idx, side in enumerate(sides):
            candidate = self.candidates[idx]
            if SIDES[idx] in self.dead:
                if candidate is not None:
                    candidate.update(end_sec=stamp, outcome="confirmed_death")
                    self.candidates[idx] = None
                self.pending[idx].clear()
                continue
            board = side.confirmed_board
            if side.state != BoardState.STABLE or board is None:
                continue
            self.boards[idx] = board.copy()
            self.board_sec[idx] = stamp
            if not occupied(board):
                if candidate is not None:
                    candidate.update(end_sec=stamp, outcome="cleared")
                    self.candidates[idx] = None
            elif candidate is None:
                candidate = dict(game=game, side=SIDES[idx], start_sec=stamp,
                                 outcome="unresolved", held=[], accepted=[])
                self.audit.append(candidate)
                self.candidates[idx] = candidate

    def notifications(self, idx: int, event: Any, stamp: float) -> list[Any]:
        """元の発火時刻と通知順を保ち、解除時は保留分を通常処理へ返す。"""
        if SIDES[idx] in self.dead:
            return []
        pending, candidate = self.pending[idx], self.candidates[idx]
        if candidate is None:
            ready = list(pending.values())
            pending.clear()
            if event is not None and self._key(event) not in {self._key(e) for e in ready}:
                ready.append(event)
            return ready
        if event is not None and event.trigger_sec in self.accepted[idx]:
            return [event]
        if event is not None and self._key(event) not in pending:
            pending[self._key(event)] = deepcopy(event)
            candidate["held"].append(dict(t_sec=stamp, trigger_sec=event.trigger_sec,
                                          mechanism=event.mechanism))
        ready = []
        for key, saved in list(pending.items()):
            prediction, score, erased = self._evidence(idx, saved)
            if prediction > 0 and erased:
                ready.append(saved)
                self.accepted[idx].add(saved.trigger_sec)
                candidate["accepted"].append(dict(t_sec=stamp, trigger_sec=saved.trigger_sec,
                    predicted_chain_count=prediction, predicted_final_score=score))
                del pending[key]
        return ready

    def _evidence(self, idx: int, event: Any) -> tuple[int, int, bool]:
        """予測は起点盤面、消去実測は別時点の確定盤面から取る。"""
        before = getattr(event, "before_board", None)
        if before is None or self.board_sec[idx] <= event.trigger_sec:
            return 0, 0, False
        result = self.simulator.simulate(before)
        return result.chain_count, calculate_chain_score(result).total_score, observed_erasure(before, self.boards[idx])

    @staticmethod
    def _key(event: Any) -> tuple:
        """同じ継続通知の再保留だけを除く。"""
        return event.trigger_sec, event.mechanism, event.chain_count, event.total_score
