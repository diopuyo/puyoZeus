"""CellColorValidator: cell 色 CNN の自己教師あり学習用 validator.

ユーザー指摘 (Phase I 観察):
    - 9 秒: 1P で 2 手置いたが認識せず (cell が EMPTY のまま)
    - 13 秒: 1P col 1, 2 の紫を 2 秒間 黄色 / 青に誤認
    これらは cell 色 CNN (cnn_phase_b_v1.pt) の認識誤りで、Phase I では
    fine-tune していなかった。

ロジック:
    1. STABLE 中に各 cell の (color, frame_idx, t_sec, cell_patch) 履歴を
       deque に蓄積する。
    2. SETTLE_FRAMES_REQUIRED frame 連続で同色 (= settled) を観測したら
       「正解色」として確定。
    3. settle 直前 LOOKBACK_FRAMES 内に異なる色を出していた frame があれば、
       そこは誤認 (transient mis-classification) として擬似ラベル化。
    4. transient frame の cell_patch + settled_color を pseudo-label として emit。

擬似ラベル形式:
    component="cell"
    input_data={"patch": ndarray (cell_patch BGR), "side": "1P"/"2P",
                "row": int, "col": int}
    label=settled_color (int, COLOR_*)
    confidence=0.90 (settle 多数決ベース、十分高信頼)
    metadata={frame_idx, settled_at_frame, predicted_color, settled_color, ...}
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)
from src.board_state_machine import BoardState
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion
from src.self_supervised.cross_validator import CrossValidator
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)


# ============================
# 定数
# ============================

# settle 確定に必要な「同色連続観測 frame 数」(STABLE 中の連続観測)
SETTLE_FRAMES_REQUIRED: int = 5

# settle 確定後、過去どこまで遡って transient 誤認を抽出するか (frame 数)
LOOKBACK_FRAMES: int = 10

# 履歴保持上限 (cell ごと)
HISTORY_WINDOW: int = 64

# settle 由来の擬似ラベル信頼度
SETTLE_CONFIDENCE: float = 0.90

# 擬似ラベル対象色 (EMPTY 含む全 7 クラス)
# COLOR_UNKNOWN は確定できないので学習対象から外す
ALLOWED_LABEL_COLORS: frozenset[int] = frozenset({
    0, 1, 2, 3, 4, 5, 9,  # EMPTY, RED, BLUE, GREEN, YELLOW, PURPLE, OJAMA
})

# 入力フレームの想定解像度 (1080p)
TARGET_FRAME_HEIGHT: int = 1080
TARGET_FRAME_WIDTH: int = 1920


# ============================
# 内部データ
# ============================


@dataclass(frozen=True)
class _CellSample:
    """1 cell の 1 frame 観測."""

    color: int
    frame_idx: int
    t_sec: float
    patch: np.ndarray  # BGR cell patch (uint8)


# ============================
# Validator 本体
# ============================


class CellColorValidator(CrossValidator):
    """settle pattern による cell 色擬似ラベル抽出.

    Args:
        settle_frames_required: 確定に必要な連続同色観測数
        lookback_frames: settle 確定時、過去何 frame 分から誤認 frame を抽出するか
        history_window: 各 cell の履歴 deque maxlen
    """

    def __init__(
        self,
        settle_frames_required: int = SETTLE_FRAMES_REQUIRED,
        lookback_frames: int = LOOKBACK_FRAMES,
        history_window: int = HISTORY_WINDOW,
    ) -> None:
        super().__init__()
        if settle_frames_required < 2:
            raise ValueError("settle_frames_required must be >= 2")
        if lookback_frames < 1:
            raise ValueError("lookback_frames must be >= 1")
        if history_window < settle_frames_required + lookback_frames:
            raise ValueError(
                "history_window must accommodate settle + lookback windows",
            )
        self._settle_n: int = int(settle_frames_required)
        self._lookback_n: int = int(lookback_frames)
        self._window: int = int(history_window)
        # 各 cell (side, row, col) ごとに deque
        self._history: dict[
            tuple[str, int, int], deque[_CellSample],
        ] = {}
        # 既に擬似ラベル化した frame_idx を再 emit しない (per-cell)
        self._emitted: set[tuple[str, int, int, int]] = set()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._history.clear()
        self._emitted.clear()

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
        if frame_bgr is None or frame_bgr.ndim != 3:
            return
        # 1080p に揃える (ROI は 1920x1080 ハードコード)
        frame_1080 = self._ensure_1080p(frame_bgr)
        if frame_1080 is None:
            return
        self._update_side(
            "1P", DEFAULT_P1_REGION, pipeline_result.p1, frame_1080,
            frame_idx, t_sec,
        )
        self._update_side(
            "2P", DEFAULT_P2_REGION, pipeline_result.p2, frame_1080,
            frame_idx, t_sec,
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_1080p(frame: np.ndarray) -> np.ndarray | None:
        """frame を 1080p (1920x1080) に揃える. 16:9 でなければ None."""
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return None
        if (h, w) == (TARGET_FRAME_HEIGHT, TARGET_FRAME_WIDTH):
            return frame
        # 16:9 検査 (近似許容)
        if abs(w * 9 - h * 16) > max(w, h):
            return None
        return cv2.resize(
            frame, (TARGET_FRAME_WIDTH, TARGET_FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    def _update_side(
        self,
        side: str,
        region: BoardRegion,
        side_result: Any,
        frame_1080: np.ndarray,
        frame_idx: int,
        t_sec: float,
    ) -> None:
        """1 side 分の cell 履歴更新 + settle 検出."""
        # STABLE のみ採用 (TSUMO/CHAIN/EFFECT は puyo 動的変化中で誤教師化リスク)
        state = getattr(side_result, "state", None)
        confirmed: Board | None = getattr(side_result, "confirmed_board", None)
        if state != BoardState.STABLE or confirmed is None:
            return
        # 各可視 cell について履歴更新 + settle 判定
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                color = int(confirmed.get(row, col))
                if color == COLOR_UNKNOWN:
                    continue  # 不確定セル (隠し段) は学習対象外
                patch = self._extract_cell_patch(
                    frame_1080, region, row, col,
                )
                if patch is None:
                    continue
                self._record_and_check(
                    side, row, col, color, frame_idx, t_sec, patch,
                )

    @staticmethod
    def _extract_cell_patch(
        frame_1080: np.ndarray,
        region: BoardRegion,
        row: int,
        col: int,
    ) -> np.ndarray | None:
        """1080p frame から cell patch を切り出し (image_reader 互換).

        ImageReader.cell_sample_rect と同じロジック (CELL_SAMPLE_RATIO=0.5)。
        """
        x1, y1, x2, y2 = region.cell_sample_rect(row, col)
        h, w = frame_1080.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(y1 + 1, min(int(y2), h))
        patch = frame_1080[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch.copy()

    def _record_and_check(
        self,
        side: str,
        row: int,
        col: int,
        color: int,
        frame_idx: int,
        t_sec: float,
        patch: np.ndarray,
    ) -> None:
        """履歴更新 + settle 判定 + 誤認 frame 抽出."""
        key = (side, row, col)
        if key not in self._history:
            self._history[key] = deque(maxlen=self._window)
        self._history[key].append(_CellSample(
            color=color, frame_idx=frame_idx, t_sec=t_sec, patch=patch,
        ))
        # settle 判定: 直近 N 件が全て同色か
        settled_color = self._detect_settle(self._history[key])
        if settled_color is None:
            return
        # settle 確定 → 過去 lookback frame で異なる色だった frame を抽出
        self._emit_transient_misreads(
            side, row, col, settled_color, t_sec,
        )

    def _detect_settle(
        self, hist: deque[_CellSample],
    ) -> int | None:
        """直近 settle_frames_required 件が同色なら、その色を返す.

        該当しない / 履歴不足は None。settle 色は ALLOWED_LABEL_COLORS のみ。
        """
        if len(hist) < self._settle_n:
            return None
        recent = list(hist)[-self._settle_n:]
        first_color = recent[0].color
        if first_color not in ALLOWED_LABEL_COLORS:
            return None
        for s in recent[1:]:
            if s.color != first_color:
                return None
        return first_color

    def _emit_transient_misreads(
        self,
        side: str,
        row: int,
        col: int,
        settled_color: int,
        t_sec: float,
    ) -> None:
        """settle 色と異なる過去 frame を擬似ラベル化."""
        key = (side, row, col)
        hist = self._history[key]
        # 直近 settle_n 件は settle frame 自身なので除外
        # その手前 lookback_n 件を検査
        n_total = len(hist)
        settle_start = n_total - self._settle_n
        check_start = max(0, settle_start - self._lookback_n)
        candidates = list(hist)[check_start:settle_start]
        for cand in candidates:
            if cand.color == settled_color:
                continue
            if cand.color == COLOR_UNKNOWN:
                continue
            # 重複防止 (same cell, same frame_idx, same predicted color)
            dedup_key = (side, row, col, cand.frame_idx)
            if dedup_key in self._emitted:
                continue
            sample = PseudoLabelSample(
                component=COMPONENT_CELL,
                timestamp=float(cand.t_sec),
                input_data={
                    "patch": cand.patch.copy(),
                    "side": side,
                    "row": int(row),
                    "col": int(col),
                },
                label=int(settled_color),
                confidence=SETTLE_CONFIDENCE,
                metadata={
                    "frame_idx": int(cand.frame_idx),
                    "predicted_color": int(cand.color),
                    "settled_color": int(settled_color),
                    "settled_at_t_sec": float(t_sec),
                    "source": "settle_pattern",
                },
            )
            self._emit(sample)
            self._emitted.add(dedup_key)


__all__ = [
    "ALLOWED_LABEL_COLORS",
    "CellColorValidator",
    "HISTORY_WINDOW",
    "LOOKBACK_FRAMES",
    "SETTLE_CONFIDENCE",
    "SETTLE_FRAMES_REQUIRED",
]
