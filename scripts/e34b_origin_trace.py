"""評価器へ渡した起点を変更せず、D1残差比較用に保存する。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.e33b_score_trace import ScoreTrace
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.exchange_event_overlay import ExchangeEventOverlay


class OriginTrace(ScoreTrace):
    """OFFの通知起点とONの保持起点を、同じ連鎖キーで記録する。"""

    def __init__(self, source: str, out: Path) -> None:
        super().__init__(source, out)
        self.out = out
        self.origins: list[dict] = []
        self.install_origins()

    def install_origins(self) -> None:
        """seed直前の実入力を保存し、現在層の盤面を書き換えない。"""
        original = ExchangeEventOverlay._predict_completion
        def traced(overlay: Any, chain: Any, event: Any, idx: int) -> None:
            saved = next((s for s in reversed(overlay._history[idx]) if s.t_sec < chain.trigger_sec), None)
            board = getattr(event, 'before_board', None)
            board = board if board is not None else saved.board if saved else None
            origin = board._grid.tolist() if board is not None else None
            original(overlay, chain, event, idx)
            if overlay._origin_guard is not None:
                origin = overlay._origin_guard.audit[-1]['board']
            self.origins.append(dict(game=overlay._game, side=chain.side, chain_id=chain.chain_id,
                trigger_sec=chain.trigger_sec, board=origin,
                mechanism=getattr(event, 'mechanism', None), count=getattr(event, 'chain_count', None)))
        ExchangeEventOverlay._predict_completion = traced

    def summary(self) -> dict:
        """得点原票と起点原票を別々に確定する。"""
        save_json(self.out/'origin_audit.json', self.origins)
        return super().summary()
