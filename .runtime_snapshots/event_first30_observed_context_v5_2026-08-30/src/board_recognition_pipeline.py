"""連続フレームを処理する時系列認識パイプライン (Phase U)。

ImageReader の単一フレーム認識を以下の時系列処理でラップする:

    Frame → AnimationFilter (連鎖中スキップ)
         ↓
         ImageReader.read_both_boards
         ↓
         TemporalSmoother (過去 N フレーム多数決)
         ↓
         StatefulBoardTracker (物理ルール違反棄却)
         ↓
         (オプション) AdaptiveBackgroundFingerprint 更新
         ↓
         確定盤面

利用例:
    pipeline = BoardRecognitionPipeline(reader)
    for frame in video_frames:
        board_1p, board_2p = pipeline.read(frame)
"""
from __future__ import annotations

import cv2
import numpy as np

from src.adaptive_background import AdaptiveBackgroundFingerprint
from src.animation_filter import AnimationFilter
from src.background_fingerprint import (
    BackgroundFingerprint,
    capture_pair_robust,
)
from src.board import Board
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    ImageReader,
)
from src.stateful_board_tracker import StatefulBoardTracker
from src.temporal_smoother import TemporalSmoother


DEFAULT_TEMPORAL_WINDOW: int = 5


class BoardRecognitionPipeline:
    """時系列を考慮した盤面認識パイプライン。

    各レイヤーは個別に有効化可能 (use_anim_filter / use_smoother / use_tracker /
    use_adaptive_bg)。デフォルトは全部 ON。
    """

    def __init__(
        self,
        reader: ImageReader,
        use_anim_filter: bool = True,
        use_smoother: bool = True,
        use_tracker: bool = True,
        use_adaptive_bg: bool = True,
        temporal_window: int = DEFAULT_TEMPORAL_WINDOW,
    ) -> None:
        self._reader = reader
        self._anim_filter = AnimationFilter() if use_anim_filter else None
        self._smoother_1p = (
            TemporalSmoother(window_size=temporal_window) if use_smoother else None
        )
        self._smoother_2p = (
            TemporalSmoother(window_size=temporal_window) if use_smoother else None
        )
        self._tracker_1p = StatefulBoardTracker() if use_tracker else None
        self._tracker_2p = StatefulBoardTracker() if use_tracker else None
        self._use_adaptive_bg = bool(use_adaptive_bg)
        self._adaptive_p1: "AdaptiveBackgroundFingerprint | None" = None
        self._adaptive_p2: "AdaptiveBackgroundFingerprint | None" = None
        self._last_b1: Board | None = None
        self._last_b2: Board | None = None
        self._frame_count = 0

    def initialize_background(
        self, frames: list[np.ndarray],
    ) -> None:
        """試合開始フレーム群から背景 FP を取得 (キャラ背景プロファイル)。"""
        if not frames:
            return
        p1_t = (
            DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
            DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
        )
        p2_t = (
            DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
            DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
        )
        fp1, fp2 = capture_pair_robust(frames, p1_t, p2_t)
        if self._use_adaptive_bg:
            self._adaptive_p1 = AdaptiveBackgroundFingerprint(fp1)
            self._adaptive_p2 = AdaptiveBackgroundFingerprint(fp2)
            self._reader.set_background_fingerprints(
                self._adaptive_p1, self._adaptive_p2,
            )
        else:
            self._reader.set_background_fingerprints(fp1, fp2)

    def read(self, frame: np.ndarray) -> tuple[Board, Board]:
        """1 フレームを処理して両盤面を返す。

        連続呼び出しで内部状態 (前盤面、tracker) が更新される。
        """
        if frame is None or frame.size == 0:
            return Board(), Board()
        # 1080p 化
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )

        # 連鎖アニメ・閃光検出: AnimationFilter は **毎フレーム** 呼んで
        # 内部の _prev_frame / _prev_stats を更新する。判定結果は
        # _last_b1 が確定している場合のみ使用する。
        if self._anim_filter is not None:
            p1_anim = self._anim_filter.is_animation(
                frame,
                (DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
                 DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height),
            )
            p2_anim = self._anim_filter.is_animation(
                frame,
                (DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
                 DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height),
            )
            if (
                self._last_b1 is not None
                and (p1_anim.is_animation or p2_anim.is_animation)
            ):
                return self._last_b1, self._last_b2

        # adaptive_bg を毎フレーム更新
        if self._adaptive_p1 is not None and self._adaptive_p2 is not None:
            self._adaptive_p1.update(
                frame,
                DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
                DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
            )
            self._adaptive_p2.update(
                frame,
                DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
                DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
            )

        # 通常認識
        try:
            board_1p, board_2p = self._reader.read_both_boards(frame)
        except Exception:
            board_1p = Board()
            board_2p = Board()

        # 時系列スムージング
        if self._smoother_1p is not None and self._smoother_2p is not None:
            board_1p = self._smoother_1p.update(board_1p)
            board_2p = self._smoother_2p.update(board_2p)

        # ステートフル追跡 (物理ルール違反棄却)
        if self._tracker_1p is not None and self._tracker_2p is not None:
            board_1p = self._tracker_1p.update(board_1p)
            board_2p = self._tracker_2p.update(board_2p)

        self._last_b1 = board_1p
        self._last_b2 = board_2p
        self._frame_count += 1
        return board_1p, board_2p

    def reset(self) -> None:
        """全レイヤーの内部状態をリセット。"""
        if self._anim_filter is not None:
            self._anim_filter.reset()
        if self._smoother_1p is not None:
            self._smoother_1p.reset()
        if self._smoother_2p is not None:
            self._smoother_2p.reset()
        if self._tracker_1p is not None:
            self._tracker_1p.reset()
        if self._tracker_2p is not None:
            self._tracker_2p.reset()
        self._last_b1 = None
        self._last_b2 = None
        self._frame_count = 0


__all__ = [
    "BoardRecognitionPipeline",
    "DEFAULT_TEMPORAL_WINDOW",
]
