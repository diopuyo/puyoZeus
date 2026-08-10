"""推論盤面生成 (Phase B-5).

BoardStateMachine の state ごとに「現 frame で表示すべき推論盤面」を生成する。

- STABLE: ctx.confirmed_board をそのまま返す
- CHAIN: ChainSimulator.simulate を CHAIN 開始時に 1 度走らせ、各 frame では
        進行率 (時刻ベース) に応じた段の board_after を返す
- TSUMO_FALL / OJAMA_FALL / EFFECT: confirmed_board を hold
  (B-5 段階の MVP。完全な落下推論は B-7 以降で拡張)
- MENU: None

新方針 (project_recognition_strategy_pivot) では、推論盤面が「真値」、
CNN 出力は drift detector の照合用とする。drift detector (B-6) は本
モジュールが返す盤面を baseline として CNN 出力と比較する。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.board import Board
from src.board_state_machine import BoardState, NON_STABLE_STATES, StateContext
from src.chain import ChainResult, ChainSimulator
from src.production_config import GHOST_CHAIN_RULE_ENABLED


# ============================
# 内部 state
# ============================


@dataclass
class _ChainPlayback:
    """CHAIN state 中の playback 情報。"""

    chain_event: object
    chain_result: ChainResult
    trigger_sec: float
    end_sec: float


# ============================
# Generator
# ============================


class InferenceBoardGenerator:
    """state context + chain_event + 時刻 から推論盤面を生成する.

    Usage:
        gen = InferenceBoardGenerator()
        for frame_idx, signals in enumerate(stream):
            ctx = sm.update(frame_idx, signals)
            board = gen.generate(
                ctx, signals.chain_event, signals.time_sec,
            )
            if board is not None:
                # board が「真値」、CNN 出力は drift detector に渡す
                ...
    """

    def __init__(self, simulator: ChainSimulator | None = None) -> None:
        # 幽霊連鎖ルール (2026-08-10 本番ON採用): production_config.py が単一情報源。
        self._sim = simulator or ChainSimulator(
            exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        )
        self._chain_playback: _ChainPlayback | None = None

    def reset(self) -> None:
        """playback 状態を初期化."""
        self._chain_playback = None

    @property
    def chain_playback(self) -> _ChainPlayback | None:
        return self._chain_playback

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def generate(
        self,
        ctx: StateContext,
        chain_event: object | None = None,
        time_sec: float | None = None,
    ) -> Board | None:
        """現 frame の推論盤面を返す.

        Args:
            ctx: BoardStateMachine.context (現 state 含む)
            chain_event: CHAIN state 入りで simulate に使う ChainEvent
            time_sec: CHAIN 進行率計算に使う現時刻 (省略時 ctx.time_sec)
        """
        t = time_sec if time_sec is not None else ctx.time_sec

        # CHAIN state 開始時に simulate を 1 度だけ走らせる
        if (
            ctx.state == BoardState.CHAIN
            and self._chain_playback is None
            and ctx.confirmed_board is not None
            and chain_event is not None
        ):
            self._start_chain_playback(ctx.confirmed_board, chain_event)

        # CHAIN 終了で playback クリア
        if ctx.state != BoardState.CHAIN and self._chain_playback is not None:
            self._chain_playback = None

        # state ごとに分岐
        if ctx.state == BoardState.STABLE:
            return ctx.confirmed_board

        if ctx.state == BoardState.CHAIN:
            board = self._chain_board_at(t)
            return board if board is not None else ctx.confirmed_board

        if ctx.state in NON_STABLE_STATES:
            # TSUMO_FALL / OJAMA_FALL / EFFECT: 直近 STABLE を hold
            return ctx.confirmed_board

        return None  # MENU

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _start_chain_playback(
        self, baseline: Board, chain_event: object,
    ) -> None:
        trigger_sec = getattr(chain_event, "trigger_sec", None)
        end_sec = getattr(chain_event, "end_sec", None)
        if trigger_sec is None or end_sec is None:
            return
        result = self._sim.simulate(baseline)
        if result.chain_count == 0:
            return  # 連鎖なし、起動しない
        self._chain_playback = _ChainPlayback(
            chain_event=chain_event,
            chain_result=result,
            trigger_sec=float(trigger_sec),
            end_sec=float(end_sec),
        )

    def _chain_board_at(self, time_sec: float) -> Board | None:
        pb = self._chain_playback
        if pb is None:
            return None
        duration = max(0.001, pb.end_sec - pb.trigger_sec)
        progress = (time_sec - pb.trigger_sec) / duration
        progress = max(0.0, min(1.0, progress))
        n_steps = pb.chain_result.chain_count
        if n_steps == 0:
            return None

        # progress を 0..n_steps にマップ
        # idx = floor(progress * n_steps)
        # idx == 0 で 1 段目消去後、idx == n_steps-1 で最終段消去後
        # progress >= 1.0 (= 連鎖終了後) なら final_board
        idx = int(progress * n_steps)
        if idx >= n_steps:
            return pb.chain_result.final_board
        return pb.chain_result.steps[idx].board_after


__all__ = [
    "InferenceBoardGenerator",
]
