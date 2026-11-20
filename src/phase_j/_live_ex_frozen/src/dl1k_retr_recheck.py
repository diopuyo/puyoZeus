"""初段の式の表示中に遅れて読めた盤を確認する。既定OFF。"""
from __future__ import annotations

from dataclasses import replace
import os
from typing import Any

ENV = 'PUYO_RETR_OWN_CHAIN_LEN'
REPEAT_WINDOW_SEC = 0.25
FIRST_STEP_COUNT = 1


def observe(tracker: Any, now: float, formula: dict, raw: Any) -> None:
    """同じ初段得点と同じ盤を二度観測した時だけ予測を取り直す。"""
    if os.environ.get(ENV, '0') != '1':
        return
    from .fire_board import clean, step_scores
    fc = tracker.fc
    if fc is None or len(fc.observed) != FIRST_STEP_COUNT:
        return
    if not formula.get('valid') or formula.get('product') != fc.observed[0]:
        return
    board = clean(raw)
    if board is None:
        return
    groups, final = tracker.trace(board)
    scores = step_scores(groups)
    if not scores or scores[0] != fc.observed[0] or len(scores) <= len(fc.steps):
        return
    key = (fc.chain_id, tuple(map(int, board.ravel())))
    previous = getattr(tracker, '_retr_len_candidate', None)
    tracker._retr_len_candidate = (key, now)
    if previous is None or previous[0] != key or not 0 < now-previous[1] <= REPEAT_WINDOW_SEC:
        return
    tracker._pred = list(scores)
    tracker.fc = replace(fc, steps=tuple(scores), final_board=final,
        source='observed_first_step_recheck', refinds=fc.refinds+1,
        exp_total=None, upside=None, n_comb=0, combos=(), comb_mass=None, comb_ms=None)
