"""置き完了合図で一度だけ行う可視セル照合。判定は履歴を変更しない。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from src.board import Board, HIDDEN_ROWS, COLOR_UNKNOWN
from src.prefire_snapshot_reader import grounded

MAX_DIFFERENCE_CELLS = 6
FORMULA_LOOKBACK_FRAMES = 4


@dataclass(frozen=True)
class Observation:
    """融合・物理補正前の独立読取りと原フレーム番号。"""

    frame: int
    stamp: float
    cnn: np.ndarray
    hsv: np.ndarray
    quality: str = ''
    erasing: bool = False

    def agreement(self) -> np.ndarray:
        """不可視段と不明セルを一致数から除く。"""
        mask = (self.cnn == self.hsv) & (self.cnn != COLOR_UNKNOWN)
        mask[:HIDDEN_ROWS] = False
        return mask


def select_observation(history: list[Observation], signal: str, frame: int) -> Observation | None:
    """式だけは直前1〜4枚の最多一致を選ぶ。同数なら新しい画像。"""
    offset = 0 if signal == 'next' else 1
    window = FORMULA_LOOKBACK_FRAMES if signal == 'formula' else offset
    candidates = [o for o in history if frame-window <= o.frame <= frame-offset]
    return max(candidates, key=lambda o: (int(o.agreement().sum()), o.frame), default=None)


def reconcile(board: Board | None, current: Observation | None,
              previous: Observation | None, max_cells: int = MAX_DIFFERENCE_CELLS
              ) -> tuple[Board | None, dict]:
    """全差分の上限・連続一致・品質・接地を確認し、原盤面を変えず返す。"""
    audit: dict = dict(reason='', differences=[], corrections=[], cell_reasons=[])
    if board is None or current is None:
        audit['reason'] = 'missing_board' if board is None else 'missing_frame'
        return None, audit
    audit.update(quality=current.quality, erasing=current.erasing,
                 previous_quality=previous.quality if previous else None,
                 previous_erasing=previous.erasing if previous else None)
    same = current.agreement()
    diff = same & (current.cnn != board._grid)
    cells = np.argwhere(diff)
    audit['differences'] = [dict(row=int(r), col=int(c), before=int(board._grid[r,c]),
                               after=int(current.cnn[r,c])) for r, c in cells]
    reason = current.quality or ('erasing' if current.erasing else '')
    if len(cells) > max_cells:
        reason = 'difference_limit'
    if previous is None or previous.frame != current.frame-1:
        reason = reason or 'nonconsecutive'
    if reason:
        audit['reason'] = reason
        return None, audit
    valid = diff & previous.agreement() & (current.cnn == previous.cnn)
    if previous.quality or previous.erasing:
        valid[:] = False
    proposed = board.copy()
    proposed._grid[valid] = current.cnn[valid]
    if not grounded(proposed._grid[HIDDEN_ROWS:]):
        audit['reason'] = 'floating'
        return None, audit
    for cell in audit['differences']:
        accepted = bool(valid[cell['row'], cell['col']])
        audit['cell_reasons'].append(dict(**cell, reason='corrected' if accepted else 'not_two_frames'))
        if accepted:
            audit['corrections'].append(cell)
    audit['reason'] = 'corrected' if valid.any() else 'no_eligible_difference'
    return (proposed if valid.any() else None), audit
