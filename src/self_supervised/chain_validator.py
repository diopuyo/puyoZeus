"""ChainValidator: chain detect の自己整合性検査.

ロジック:
    1. score delta 整合: state==CHAIN 終了直後の score jump =
       ChainSimulator.simulate(before_board) の total_score と一致するはず。
       一致 → ChainEvent が ground truth (chain_count, ojama 等)。
       不一致 → ChainTracker または ScoreOcr のいずれかが misread。
    2. CHAIN 持続時間: 1 連鎖あたり ~80-86 frame (60fps 想定)。
       chain_count × ~84 と CHAIN state 期間がほぼ一致すべき。
    3. board cell delta: chain 前後で消去 cell 数 = total_erased。

擬似ラベル形式:
    component="chain"
    input_data={
        "before_board_grid": List[List[int]],  # シリアライズ用
        "score_jump": int,
        "duration_frames": int,
    }
    label={"chain_count": int, "total_erased": int, "ojama_sent": int}
    confidence=0.95+
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, Board
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import calculate_chain_score
from src.self_supervised.cross_validator import CrossValidator
from src.self_supervised.pseudo_label import (
    COMPONENT_CHAIN,
    PseudoLabelSample,
)


# 1 連鎖あたり期待 frame 数 (60fps 想定)
EXPECTED_FRAMES_PER_CHAIN: float = 84.0
# 持続時間が期待 ± この比率以内なら OK
DURATION_TOLERANCE: float = 0.40
# score 一致なら高信頼
SCORE_MATCH_CONFIDENCE: float = 0.95
SCORE_MISMATCH_CONFIDENCE: float = 0.70


@dataclass
class _ChainTracking:
    """1 side 分の連鎖追跡 state."""

    in_chain: bool = False
    start_frame_idx: int = -1
    start_score: int | None = None
    before_board: Board | None = None
    chain_event: Any | None = None  # ChainEvent (循環 import 回避で Any)


class ChainValidator(CrossValidator):
    """chain detect の自己整合性検査."""

    def __init__(self) -> None:
        super().__init__()
        self._track_1p = _ChainTracking()
        self._track_2p = _ChainTracking()
        # 幽霊連鎖ルール (2026-08-10 本番ON採用): production_config.py が単一情報源。
        self._sim = ChainSimulator(
            exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        )

    def reset(self) -> None:
        super().reset()
        self._track_1p = _ChainTracking()
        self._track_2p = _ChainTracking()

    def update(
        self,
        frame_idx: int,
        t_sec: float,
        pipeline_result: Any,
        frame_bgr: np.ndarray | None,
    ) -> None:
        """1 frame の更新."""
        if not getattr(pipeline_result, "is_match_active", False):
            return
        self._update_side("1P", frame_idx, t_sec, pipeline_result.p1)
        self._update_side("2P", frame_idx, t_sec, pipeline_result.p2)

    def _update_side(
        self,
        side: str,
        frame_idx: int,
        t_sec: float,
        side_result: Any,
    ) -> None:
        """1 side の連鎖 lifecycle 監視."""
        track = self._track_1p if side == "1P" else self._track_2p
        state = side_result.state
        score = getattr(side_result, "score", None)
        chain_event = getattr(side_result, "chain_event", None)
        if state == BoardState.CHAIN:
            if not track.in_chain:
                # CHAIN 開始
                track.in_chain = True
                track.start_frame_idx = frame_idx
                track.start_score = score
                track.before_board = (
                    chain_event.before_board
                    if chain_event is not None else None
                )
                track.chain_event = chain_event
            else:
                # CHAIN 継続中、最新 chain_event があれば更新
                if chain_event is not None:
                    track.chain_event = chain_event
                    if track.before_board is None:
                        track.before_board = chain_event.before_board
        elif track.in_chain and state != BoardState.CHAIN:
            # CHAIN 終了
            self._emit_chain_consistency(
                side, frame_idx, t_sec, track, score,
            )
            # state リセット
            if side == "1P":
                self._track_1p = _ChainTracking()
            else:
                self._track_2p = _ChainTracking()

    def _emit_chain_consistency(
        self,
        side: str,
        end_frame_idx: int,
        t_sec: float,
        track: _ChainTracking,
        end_score: int | None,
    ) -> None:
        """CHAIN 終了時の整合性 emit."""
        if track.before_board is None or track.chain_event is None:
            return
        if track.start_score is None or end_score is None:
            return
        # ChainSimulator で再シミュレーション
        try:
            cr = self._sim.simulate(track.before_board)
        except Exception:
            return
        if cr.chain_count <= 0:
            return
        try:
            score_result = calculate_chain_score(cr)
        except Exception:
            return
        expected_score_delta = int(score_result.total_score)
        actual_delta = int(end_score - track.start_score)
        duration_frames = end_frame_idx - track.start_frame_idx
        expected_duration = EXPECTED_FRAMES_PER_CHAIN * cr.chain_count
        duration_ok = (
            duration_frames >= expected_duration * (1.0 - DURATION_TOLERANCE)
            and duration_frames <= expected_duration * (1.0 + DURATION_TOLERANCE)
        )
        score_ok = abs(actual_delta - expected_score_delta) <= max(
            10, int(0.05 * expected_score_delta),
        )
        match = score_ok and duration_ok
        # 高信頼: tracker の chain_count = simulate の chain_count、
        # かつ score / duration 整合
        ce_chain_count = int(track.chain_event.chain_count)
        sim_chain_count = int(cr.chain_count)
        chain_count_match = ce_chain_count == sim_chain_count
        sample = PseudoLabelSample(
            component=COMPONENT_CHAIN,
            timestamp=t_sec,
            input_data={
                "before_board_grid": _board_to_list(track.before_board),
                "score_jump": int(actual_delta),
                "duration_frames": int(duration_frames),
                "side": side,
            },
            label={
                "chain_count": sim_chain_count,
                "total_erased": int(cr.total_erased),
                "expected_score_delta": int(expected_score_delta),
            },
            confidence=(
                SCORE_MATCH_CONFIDENCE
                if (match and chain_count_match)
                else SCORE_MISMATCH_CONFIDENCE
            ),
            metadata={
                "side": side,
                "start_frame_idx": int(track.start_frame_idx),
                "end_frame_idx": int(end_frame_idx),
                "chain_event_count": ce_chain_count,
                "simulator_count": sim_chain_count,
                "score_match": bool(score_ok),
                "duration_match": bool(duration_ok),
                "chain_count_match": bool(chain_count_match),
                "source": "chain_consistency",
            },
        )
        self._emit(sample)


# ============================
# helpers
# ============================


def _board_to_list(board: Board) -> list[list[int]]:
    """Board を 2D int リストへ変換 (JSON 化用)."""
    return [
        [int(board.get(r, c)) for c in range(BOARD_COLS)]
        for r in range(BOARD_ROWS)
    ]


__all__ = [
    "DURATION_TOLERANCE",
    "EXPECTED_FRAMES_PER_CHAIN",
    "SCORE_MATCH_CONFIDENCE",
    "SCORE_MISMATCH_CONFIDENCE",
    "ChainValidator",
]
