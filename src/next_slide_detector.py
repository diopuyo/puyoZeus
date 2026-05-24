"""NEXT window slide motion detector + tsumo placement self-consistency.

Phase I R-1 + R-7 統合:
    - R-7: NEXT ROI (next/dnext) の pixel-wise abs diff を見て
      「next 1 → next 0 にスライドした」signal を返す。これは
      「ツモが盤面に置かれた」直接 signal で、TsumoPhaseDetector の
      フレーム内 puyo count delta だけでは捕まえきれない置きを補強する。
    - R-1: 着地後 (TSUMO_FALL → STABLE 復帰時) に、cnn_board と直前
      STABLE 盤面の色 count delta が、落下中ツモ (next_queue[-2]) と
      整合しているかを validate するヘルパも提供。

設計:
    - state を持つ最小限の wrapper (slide motion 検出のため running stats)。
    - frame は呼び出し側 (RecognitionPipeline) が提供。`update(prev, curr)`
      の inputs は (BGR uint8, 1080x1920) を期待。解像度違いは silent skip。
    - threshold は固定 default + 適応 median (running) の OR で判定する。
      固定 threshold は最初の数 frame でも発火可能、median は安定後の
      動的補正に使う。
    - 1P / 2P 個別 ROI に対してそれぞれ instance を持つ想定 (= state は
      side ごとに分離)。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
)
from src.next_detector import (
    ROI_1P_DNEXT_BOT,
    ROI_1P_DNEXT_TOP,
    ROI_1P_NEXT_BOT,
    ROI_1P_NEXT_TOP,
    ROI_2P_DNEXT_BOT,
    ROI_2P_DNEXT_TOP,
    ROI_2P_NEXT_BOT,
    ROI_2P_NEXT_TOP,
)


# ============================
# 定数
# ============================

# pixel 単位の絶対差分 (gray scale 換算後) の固定 threshold。
# 通常時の next ROI は (ほぼ) 静止しており diff は数値的に小さい。
# スライド motion 中はぷよが画面内を移動するので大きな diff となる。
DEFAULT_DIFF_THRESHOLD: float = 8.0

# 適応 threshold: 直近 N frame の median を取り、その K 倍を閾値にする。
# K は経験値、median 比 3〜4 倍を「異常」と見なす。
ADAPTIVE_HISTORY_LEN: int = 30
ADAPTIVE_K: float = 4.0
# 適応 threshold が極端に小さくなりすぎないよう floor を設ける。
ADAPTIVE_FLOOR: float = 4.0

# slide motion を観測した後、即座にもう一度発火するのを抑制する cooldown。
# 1 ツモ落下サイクルあたり 1 回だけ発火させる狙い。
DEFAULT_COOLDOWN_FRAMES: int = 6

# 着地直後の color count delta tolerance:
# next_pair の色 count が想定 +1 / +2 と一致しているか確認する。
# 連鎖発火で消える等の特殊ケースを許容するため tolerance を 1 で許す。
COLOR_DELTA_TOLERANCE: int = 1


# ============================
# 結果 dataclass
# ============================


@dataclass(frozen=True)
class SlideMotionResult:
    """next ROI スライド検出結果.

    Attributes:
        slide_motion: スライド motion (= ツモ移動) を検出したか。
        diff_score: pixel 平均絶対差分 (gray scale)。0.0 = 完全静止。
        threshold_used: 判定に使った閾値 (debug 用)。
    """

    slide_motion: bool
    diff_score: float
    threshold_used: float


@dataclass(frozen=True)
class PlacementValidationResult:
    """tsumo 着地時の自己整合性チェック結果.

    Attributes:
        consistent: 色 count delta が落下ペアと一致するか。
        delta_total: cnn_board.count_puyos() - baseline.count_puyos()。
        details: 色別 delta dict (debug 用)。
    """

    consistent: bool
    delta_total: int
    details: dict[int, int] = field(default_factory=dict)


# ============================
# Slide motion detector
# ============================


class NextSlideDetector:
    """next ROI のスライド motion を検出する stateful detector.

    1 frame ずつ (prev, curr) を渡せば、直近 frame との差分から
    「ぷよ tile がスライドしたかどうか」を返す。state machine 側で
    TSUMO_FALL → STABLE 遷移の補強 signal として使う。

    side 別 (1P / 2P) に instance を持つこと。共通化すると ROI 平均で
    片側のスライドが他側で薄まる。
    """

    def __init__(
        self,
        side: str = "1P",
        diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
        adaptive_history_len: int = ADAPTIVE_HISTORY_LEN,
        adaptive_k: float = ADAPTIVE_K,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ) -> None:
        if side not in ("1P", "2P"):
            raise ValueError(f"side must be '1P' or '2P', got {side!r}")
        self._side = side
        self._diff_threshold = float(diff_threshold)
        self._adaptive_k = float(adaptive_k)
        self._cooldown_frames = int(cooldown_frames)
        self._diff_history: deque[float] = deque(
            maxlen=max(1, adaptive_history_len),
        )
        self._cooldown: int = 0
        self._last_diff_score: float = 0.0

    @property
    def side(self) -> str:
        return self._side

    def reset(self) -> None:
        """internal state をクリア (試合切替等で呼ぶ)。"""
        self._diff_history.clear()
        self._cooldown = 0
        self._last_diff_score = 0.0

    def _get_rois(self) -> tuple[tuple[int, int, int, int], ...]:
        """side に応じた next/dnext ROI 4 つを返す."""
        if self._side == "1P":
            return (
                ROI_1P_NEXT_TOP, ROI_1P_NEXT_BOT,
                ROI_1P_DNEXT_TOP, ROI_1P_DNEXT_BOT,
            )
        return (
            ROI_2P_NEXT_TOP, ROI_2P_NEXT_BOT,
            ROI_2P_DNEXT_TOP, ROI_2P_DNEXT_BOT,
        )

    @staticmethod
    def _extract_roi_gray(
        frame: np.ndarray, roi: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """frame から ROI を切り出して gray scale で返す.

        invalid frame (None / 解像度不足) なら None。
        """
        if frame is None or frame.ndim != 3:
            return None
        h, w = frame.shape[:2]
        y1, y2, x1, x2 = roi
        if y2 > h or x2 > w or y1 < 0 or x1 < 0:
            return None
        if y2 - y1 <= 0 or x2 - x1 <= 0:
            return None
        patch = frame[y1:y2, x1:x2]
        # 高速化のため簡易 gray (B+G+R)/3。BGR vs RGB は問わない。
        return patch.astype(np.float32).mean(axis=2)

    def _compute_diff_score(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray,
    ) -> float:
        """全 NEXT ROI 共通の平均絶対差分 (gray) を計算."""
        diffs: list[float] = []
        for roi in self._get_rois():
            prev_gray = self._extract_roi_gray(prev_frame, roi)
            curr_gray = self._extract_roi_gray(curr_frame, roi)
            if prev_gray is None or curr_gray is None:
                continue
            if prev_gray.shape != curr_gray.shape:
                continue
            diffs.append(float(np.abs(prev_gray - curr_gray).mean()))
        if not diffs:
            return 0.0
        # 全 ROI の平均ではなく最大を採用: スライド初期は next_top のみが
        # 動き、他 ROI はまだ静止のことが多いため、平均だと信号が薄まる。
        return max(diffs)

    def _adaptive_threshold(self) -> float:
        """直近 history から適応 threshold を計算 (median * K, floor あり)."""
        if not self._diff_history:
            return self._diff_threshold
        med = float(np.median(np.asarray(self._diff_history, dtype=np.float32)))
        adaptive = max(med * self._adaptive_k, ADAPTIVE_FLOOR)
        # 固定 threshold とのうち大きい方 (= より保守的) を採用
        return max(adaptive, self._diff_threshold)

    def update(
        self,
        prev_frame: np.ndarray | None,
        curr_frame: np.ndarray | None,
    ) -> SlideMotionResult:
        """1 frame 分の (prev, curr) 比較を実行.

        Args:
            prev_frame: 直前 frame (BGR uint8, 1080x1920 想定)。
            curr_frame: 現 frame。
                どちらかが None なら slide_motion=False で skip。

        Returns:
            SlideMotionResult.
        """
        # cooldown 消化
        if self._cooldown > 0:
            self._cooldown -= 1

        if prev_frame is None or curr_frame is None:
            return SlideMotionResult(
                slide_motion=False,
                diff_score=0.0,
                threshold_used=self._diff_threshold,
            )

        diff_score = self._compute_diff_score(prev_frame, curr_frame)
        self._last_diff_score = diff_score
        threshold = self._adaptive_threshold()

        # スライド検出: cooldown 中は発火しない
        slide = (
            diff_score >= threshold
            and self._cooldown == 0
        )
        if slide:
            self._cooldown = self._cooldown_frames

        # history は「静止中の median」を取りたいので、発火していない
        # frame の値だけを履歴に積む。
        if not slide:
            self._diff_history.append(diff_score)

        return SlideMotionResult(
            slide_motion=slide,
            diff_score=diff_score,
            threshold_used=threshold,
        )

    @property
    def last_diff_score(self) -> float:
        return self._last_diff_score


# ============================
# Self-consistency validator (R-1)
# ============================


def _color_counts(board: Board) -> dict[int, int]:
    """Board の色別 puyo count を辞書で返す (空・UNKNOWN 除外)."""
    counts: dict[int, int] = {}
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(board.get(r, c))
            if v in (COLOR_EMPTY, COLOR_UNKNOWN):
                continue
            counts[v] = counts.get(v, 0) + 1
    return counts


def validate_tsumo_placement(
    baseline: Board | None,
    landed: Board,
    falling_pair: tuple[int, int] | None,
    *,
    tolerance: int = COLOR_DELTA_TOLERANCE,
) -> PlacementValidationResult:
    """着地後の盤面が落下ペアと色 count delta が整合するか check.

    Args:
        baseline: TSUMO_FALL 開始前の確定盤面。
        landed: 着地後の CNN 盤面。
        falling_pair: 落下中ツモ (top_color, bot_color)。next_queue[-2] 由来。
        tolerance: ±許容差 (連鎖発火等の特殊ケース緩和)。

    Returns:
        PlacementValidationResult.
        baseline=None / falling_pair=None のいずれかなら consistent=False。
    """
    if baseline is None or falling_pair is None:
        return PlacementValidationResult(
            consistent=False, delta_total=0, details={},
        )
    base_counts = _color_counts(baseline)
    land_counts = _color_counts(landed)

    delta: dict[int, int] = {}
    all_colors: set[int] = set(base_counts.keys()) | set(land_counts.keys())
    for color in all_colors:
        d = land_counts.get(color, 0) - base_counts.get(color, 0)
        if d != 0:
            delta[color] = d

    delta_total = sum(delta.values())

    # 落下ペアから期待 delta を計算
    expected: dict[int, int] = {}
    for c in falling_pair:
        ci = int(c)
        if ci in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA):
            continue
        expected[ci] = expected.get(ci, 0) + 1

    # 期待 vs 実測 を tolerance で比較。
    # 個別の差分 |actual-expected| を全色で集計し、合計が tolerance 以内なら
    # 整合とみなす。色違いが 1 つでもあれば各色 |1| が累積し tolerance 超えで
    # 落ちる設計 (= 強い整合性チェック)。
    all_color_set: set[int] = (
        set(expected.keys()) | set(delta.keys())
    )
    diff_sum = 0
    for color in all_color_set:
        exp_n = expected.get(color, 0)
        act_n = delta.get(color, 0)
        diff_sum += abs(act_n - exp_n)
    consistent = diff_sum <= tolerance
    # 全体 puyo 数が +1 か +2 でない場合も不整合
    if consistent and not (1 - tolerance <= delta_total <= 2 + tolerance):
        consistent = False

    return PlacementValidationResult(
        consistent=consistent,
        delta_total=delta_total,
        details=delta,
    )


__all__ = [
    "ADAPTIVE_FLOOR",
    "ADAPTIVE_HISTORY_LEN",
    "ADAPTIVE_K",
    "COLOR_DELTA_TOLERANCE",
    "DEFAULT_COOLDOWN_FRAMES",
    "DEFAULT_DIFF_THRESHOLD",
    "NextSlideDetector",
    "PlacementValidationResult",
    "SlideMotionResult",
    "validate_tsumo_placement",
]
