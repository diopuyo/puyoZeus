"""A1: 予測不能な発火を、時刻以前の式と途中観測盤面だけで補う。"""
from __future__ import annotations

from typing import Any
from src.board_state_machine import BoardState
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.exchange_midchain_completion import remaining, compact
from src.exchange_event_tracker import valid_nonnegative


class CompletionScoreFallback:
    """連鎖ごとに最新の式を保持し、独立した完走予測が無い場合だけ補完する。"""

    def __init__(self) -> None:
        self.entries: dict[tuple, dict] = {}
        self.audit: list[dict] = []

    def reset(self) -> None:
        """過去試合の参照を解放し、採点用の原票だけを残す。"""
        self.entries.clear()

    def observe(self, overlay: Any, result: Any, stamp: float) -> None:
        """式は通知時刻以前だけを受理し、落下後の実観測を起点にする。"""
        for idx, side in enumerate((result.p1, result.p2)):
            chain = overlay.tracker.latest_chain(f'{idx+1}P')
            if chain is None or chain.end_signal_sec is not None:
                continue
            key = (overlay._game, chain.side, chain.chain_id)
            entry = self.entries.setdefault(key, {})
            event = side.chain_event
            if (event is not None and event.mechanism == CHAIN_MECHANISM_FORMULA_READ
                    and chain.trigger_sec <= event.trigger_sec <= stamp and event.chain_count > 0
                    and valid_nonnegative(event.total_score)):
                entry['formula'] = (event.chain_count, event.total_score)
            current = (chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board)
            if current != entry.get('published') and (current[0] or current[1]):
                continue
            board = getattr(side, 'midchain_board', None)
            if (side.state != BoardState.GRAVITY_SETTLE or board is None
                    or 'formula' not in entry or not compact(board)):
                continue
            count, score = entry['formula']
            identity = (count, score, board._grid.tobytes())
            if identity == entry.get('identity'):
                continue
            value = remaining(board, count, score, overlay._landing_projection.simulator)
            if value is None:
                value = dict(score=score, count=count, board=board._grid.tolist())
            predicted = (value['score'], value['count'], value['board'])
            chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board = predicted
            entry.update(published=predicted, identity=identity)
            self.audit.append(dict(game=key[0], side=key[1], chain_id=key[2], t_sec=stamp,
                observed_count=count, observed_score=score, predicted_score=value['score'],
                predicted_count=value['count'], method='observed_formula_plus_remaining'))
