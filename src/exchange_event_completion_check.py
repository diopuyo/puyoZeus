"""連鎖終了直後・次の着手前の最初の確定盤面だけを因果的に照合する。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from src.board import COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA
from src.board_state_machine import BoardState

SIDES = ("1P", "2P")


def colored_layout(grid: Any) -> np.ndarray:
    """おじゃまだけを空白へ置換し、色ぷよの座標を動かさない。"""
    array = np.asarray(grid).copy()
    array[array == COLOR_OJAMA] = COLOR_EMPTY
    return array


@dataclass
class CompletionSample:
    """終了候補ごとに最初の盤面と着手の締切を保持する。"""
    end: float
    board: np.ndarray | None = None
    stamp: float | None = None
    cutoff: float | None = None
    checked: bool = False
    mismatch: bool = False


class CompletionVerifier:
    """終了撤回時には照合も撤回し、後続着手の盤面へ取り替えない。"""
    def __init__(self) -> None:
        self.samples: dict[int, CompletionSample] = {}

    def check(self, overlay: Any, result: Any, chain: Any) -> bool:
        """終了静穏確認後に、同側・同連鎖の保存済み最初の盤面を比較する。"""
        identity, end = chain.chain_id, chain.end_signal_sec
        if end is None:
            self.samples.pop(identity, None)
            return False
        sample = self.samples.get(identity)
        if sample is None or sample.end != end:
            sample = self.samples[identity] = CompletionSample(end)
        idx = SIDES.index(chain.side)
        side, history = (result.p1, result.p2)[idx], overlay._history[idx]
        stamp = getattr(overlay, "_completion_stamp", end)
        self._capture(sample, side, history, chain, stamp)
        if sample.checked or chain.end_confirmed is not True or sample.board is None:
            return sample.mismatch
        if chain.predicted_final_board is None:
            return False
        sample.checked = True
        if not np.any(sample.board == COLOR_UNKNOWN):
            sample.mismatch = not np.array_equal(colored_layout(sample.board),
                                                colored_layout(chain.predicted_final_board))
        return sample.mismatch

    def _capture(self, sample: CompletionSample, side: Any, history: list,
                 chain: Any, stamp: float) -> None:
        """次ツモ落下・操作加点より前に見えた最初のSTABLEだけを保存する。"""
        cutoffs = [v for v in (sample.cutoff, chain.post_end_drop_sec) if v is not None]
        if side.state == BoardState.TSUMO_FALL and stamp >= sample.end:
            cutoffs.append(stamp)
        if cutoffs:
            sample.cutoff = min(cutoffs)
        if sample.board is not None or side.state != BoardState.STABLE or not history:
            return
        first = next((h for h in history if h.t_sec > sample.end), None)
        if first is None or (sample.cutoff is not None and first.t_sec >= sample.cutoff):
            return
        sample.board, sample.stamp = first.board._grid.copy(), first.t_sec
