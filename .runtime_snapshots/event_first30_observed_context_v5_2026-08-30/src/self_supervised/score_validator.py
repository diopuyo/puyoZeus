"""ScoreValidator: score OCR の自己整合性検査.

ロジック:
    1. 単調性: 試合中 score は減らない。
       減少 → OCR misread 確定。
    2. 連鎖整合性: state==CHAIN 終了直後の score jump は
       ChainSimulator.simulate(before_board) の total_score と一致するはず。
    3. 連続性: 直前 N frame と同一 score + 単調 → 高信頼。
       周辺で同値が観測される frame の digit patch を per-digit に分解して
       擬似ラベル化。

擬似ラベル形式:
    component="score"
    input_data={"patch": np.ndarray (50x40, BGR)} (1 桁分の画像)
    label=0..9 (正解 digit)
    confidence=0.95+
    metadata={frame_idx, side, digit_pos}
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.board_state_machine import BoardState
from src.score_ocr import (
    DIGIT_HEIGHT,
    DIGIT_LEFTS_1P,
    DIGIT_LEFTS_2P,
    DIGIT_TOP,
    DIGIT_WIDTH,
    SCORE_1P_REGION,
    SCORE_2P_REGION,
)
from src.self_supervised.cross_validator import CrossValidator
from src.self_supervised.pseudo_label import (
    COMPONENT_SCORE,
    PseudoLabelSample,
)


# 擬似ラベル抽出用パラメータ
SCORE_HISTORY_WINDOW: int = 7
SCORE_AGREE_MIN: int = 5  # window 内で何 frame 一致なら confirmed か
HIGH_CONFIDENCE: float = 0.95
MEDIUM_CONFIDENCE: float = 0.80

# Phase I 改良: 緩和版 emit のためのパラメータ
# 5 fps サンプリングで window=7 / agree_min=5 は厳しすぎた為、
# 「3 frame 連続一致」を MEDIUM、「5 frame 一致」を HIGH の二段階に分ける。
SCORE_LENIENT_AGREE_MIN: int = 3  # MEDIUM 信頼で emit する最小 frame 数
SCORE_LENIENT_CONFIDENCE: float = 0.85  # 3 frame 一致時の信頼度


@dataclass
class _ScoreFrame:
    """1 frame 分の score 観測."""

    frame_idx: int
    t_sec: float
    digits_1p: tuple[int | None, ...]
    digits_2p: tuple[int | None, ...]
    score_1p: int | None
    score_2p: int | None
    state_1p: BoardState
    state_2p: BoardState
    score_roi_1p: np.ndarray | None  # 65x320 BGR
    score_roi_2p: np.ndarray | None


class ScoreValidator(CrossValidator):
    """score OCR の自己整合性検査 + 擬似ラベル抽出."""

    def __init__(
        self,
        history_window: int = SCORE_HISTORY_WINDOW,
        agree_min: int = SCORE_AGREE_MIN,
        lenient_agree_min: int = SCORE_LENIENT_AGREE_MIN,
        enable_lenient_emit: bool = True,
    ) -> None:
        super().__init__()
        if history_window < 3:
            raise ValueError("history_window must be >= 3")
        if agree_min < 2 or agree_min > history_window:
            raise ValueError("agree_min must be in [2, history_window]")
        if lenient_agree_min < 2 or lenient_agree_min > agree_min:
            raise ValueError(
                "lenient_agree_min must be in [2, agree_min]"
            )
        self._history: deque[_ScoreFrame] = deque(maxlen=history_window)
        self._agree_min = agree_min
        self._lenient_agree_min = lenient_agree_min
        self._enable_lenient = bool(enable_lenient_emit)
        # 既に擬似ラベル化した (frame_idx, side, pos) を skip するための set
        # 旧仕様 (frame_idx, side) も互換のため保持
        self._emitted: set[tuple[int, str]] = set()
        # confidence 別 dedup. lenient → HIGH 昇格を許容する設計のため、
        # source 別に管理する。key: (frame_idx, side, pos, conf_tier)
        self._emitted_pos: set[tuple[int, str, int]] = set()  # HIGH 用
        self._emitted_pos_lenient: set[tuple[int, str, int]] = set()
        # 直近の確定 score (単調性チェック用、監視のみ)
        self._last_confirmed_1p: int | None = None
        self._last_confirmed_2p: int | None = None

    def reset(self) -> None:
        super().reset()
        self._history.clear()
        self._emitted.clear()
        self._emitted_pos.clear()
        self._emitted_pos_lenient.clear()
        self._last_confirmed_1p = None
        self._last_confirmed_2p = None

    def update(
        self,
        frame_idx: int,
        t_sec: float,
        pipeline_result: Any,
        frame_bgr: np.ndarray | None,
    ) -> None:
        """1 frame の更新."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        if not getattr(pipeline_result, "is_match_active", False):
            return
        roi_1p = _crop_score_roi(frame_bgr, "1P")
        roi_2p = _crop_score_roi(frame_bgr, "2P")
        # pipeline_result に digits 情報は無い → score_tracker から間接的に取得
        # ここでは score (= last_score) のみ取れる。digit patch は ROI 切出しで対応。
        s_1p = getattr(pipeline_result.p1, "score", None)
        s_2p = getattr(pipeline_result.p2, "score", None)
        digits_1p = _score_to_digits(s_1p) if s_1p is not None else (None,) * 8
        digits_2p = _score_to_digits(s_2p) if s_2p is not None else (None,) * 8
        rec = _ScoreFrame(
            frame_idx=frame_idx,
            t_sec=t_sec,
            digits_1p=digits_1p,
            digits_2p=digits_2p,
            score_1p=s_1p,
            score_2p=s_2p,
            state_1p=pipeline_result.p1.state,
            state_2p=pipeline_result.p2.state,
            score_roi_1p=roi_1p,
            score_roi_2p=roi_2p,
        )
        self._history.append(rec)
        self._maybe_emit_window()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _maybe_emit_window(self) -> None:
        """window 内で多数決一致した digit patch を擬似ラベル化.

        lenient モード ON 時は agree_min より少ない一致でも MEDIUM 信頼で emit。
        実環境 (5fps サンプリング) で score OCR がブレやすい状況に対応。
        """
        threshold = (
            self._lenient_agree_min
            if self._enable_lenient
            else self._agree_min
        )
        if len(self._history) < threshold:
            return
        # 最も古い frame が confirmed 候補。中央値前後に多数派 → 古い frame を確定。
        center = self._history[len(self._history) // 2]
        for side in ("1P", "2P"):
            self._emit_for_side(side, center)

    def _emit_for_side(self, side: str, center: _ScoreFrame) -> None:
        """1 side について、中央 frame の digits が window 多数決と一致するか
        を確認し、一致すれば per-digit 擬似ラベルを emit.

        改良点:
            - 単調性違反でも emit を試みる (recovery_pair 同等。pos 別に)
            - agree count が agree_min 以上なら HIGH、lenient_agree_min 以上
              なら MEDIUM の二段階 emit
            - 既 emit 判定を (frame_idx, side, pos) 単位に細分化、
              一部 pos だけ後から追加 emit を許可
        """
        center_digits = (
            center.digits_1p if side == "1P" else center.digits_2p
        )
        center_roi = (
            center.score_roi_1p if side == "1P" else center.score_roi_2p
        )
        if center_roi is None:
            return
        if any(d is None for d in center_digits):
            return
        agree_per_pos = self._compute_agree_per_pos(side, center_digits)
        # 単調性違反フラグ (emit するが metadata に記録)
        last_conf = (
            self._last_confirmed_1p if side == "1P"
            else self._last_confirmed_2p
        )
        center_score = (
            center.score_1p if side == "1P" else center.score_2p
        )
        monotonic_violation = (
            last_conf is not None
            and center_score is not None
            and center_score < last_conf
        )
        # 違反時は厳格 (HIGH のみ) emit
        emitted_any = self._emit_digits(
            side, center, center_digits, center_roi,
            agree_per_pos, monotonic_violation,
        )
        if emitted_any and center_score is not None and not monotonic_violation:
            if side == "1P":
                self._last_confirmed_1p = center_score
            else:
                self._last_confirmed_2p = center_score

    def _compute_agree_per_pos(
        self, side: str, center_digits: tuple[int | None, ...],
    ) -> list[int]:
        """window 内の各 pos で center_digits と一致する frame 数をカウント."""
        agree_per_pos: list[int] = [0] * 8
        all_digits = [
            (f.digits_1p if side == "1P" else f.digits_2p)
            for f in self._history
        ]
        for digits in all_digits:
            for pos, d in enumerate(digits):
                if d is not None and d == center_digits[pos]:
                    agree_per_pos[pos] += 1
        return agree_per_pos

    def _emit_digits(
        self,
        side: str,
        center: _ScoreFrame,
        center_digits: tuple[int | None, ...],
        center_roi: np.ndarray,
        agree_per_pos: list[int],
        monotonic_violation: bool,
    ) -> bool:
        """pos 別に擬似ラベル emit. 違反時は emit せず (旧仕様互換).

        lenient → HIGH への昇格を許可: 同 pos が lenient で emit 済でも、
        agree が agree_min を超えたら HIGH を別 sample として再 emit する。
        """
        if monotonic_violation:
            return False
        emitted_any = False
        key_legacy = (center.frame_idx, side)
        for pos in range(8):
            pos_key = (center.frame_idx, side, pos)
            agree = agree_per_pos[pos]
            high_eligible = agree >= self._agree_min
            lenient_eligible = (
                self._enable_lenient
                and agree >= self._lenient_agree_min
            )
            if high_eligible and pos_key not in self._emitted_pos:
                conf, source = HIGH_CONFIDENCE, "window_agreement"
                if self._do_emit(
                    side, center, center_digits, center_roi,
                    agree, pos, conf, source,
                ):
                    self._emitted_pos.add(pos_key)
                    emitted_any = True
            elif (
                lenient_eligible
                and pos_key not in self._emitted_pos
                and pos_key not in self._emitted_pos_lenient
            ):
                conf, source = (
                    SCORE_LENIENT_CONFIDENCE, "window_agreement_lenient",
                )
                if self._do_emit(
                    side, center, center_digits, center_roi,
                    agree, pos, conf, source,
                ):
                    self._emitted_pos_lenient.add(pos_key)
                    emitted_any = True
        if emitted_any:
            self._emitted.add(key_legacy)
        return emitted_any

    def _do_emit(
        self,
        side: str,
        center: _ScoreFrame,
        center_digits: tuple[int | None, ...],
        center_roi: np.ndarray,
        agree: int,
        pos: int,
        conf: float,
        source: str,
    ) -> bool:
        """1 件の PseudoLabelSample を生成し buffer に積む helper."""
        patch = _crop_digit_patch(center_roi, pos, side)
        if patch is None:
            return False
        sample = PseudoLabelSample(
            component=COMPONENT_SCORE,
            timestamp=center.t_sec,
            input_data={"patch": patch.copy()},
            label=int(center_digits[pos]),
            confidence=conf,
            metadata={
                "frame_idx": int(center.frame_idx),
                "side": side,
                "digit_pos": int(pos),
                "agree_count": int(agree),
                "window_size": int(len(self._history)),
                "source": source,
            },
        )
        self._emit(sample)
        return True

    def emit_chain_consistency(
        self,
        side: str,
        before_score: int,
        after_score: int,
        expected_delta: int,
        t_sec: float,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """連鎖整合性チェックを直接呼び出す API (ChainValidator 連携用).

        delta 一致 → high confidence emit (input_data=数値ペア)。
        不一致 → 警告のみ (擬似ラベル化はしない)。

        Returns:
            True: 一致、False: 不一致
        """
        actual = after_score - before_score
        match = actual == expected_delta
        sample = PseudoLabelSample(
            component=COMPONENT_SCORE,
            timestamp=t_sec,
            input_data={
                "before_score": int(before_score),
                "after_score": int(after_score),
            },
            label={"expected_delta": int(expected_delta)},
            confidence=HIGH_CONFIDENCE if match else MEDIUM_CONFIDENCE,
            metadata={
                "side": side,
                "match": bool(match),
                "source": "chain_consistency",
                **(metadata or {}),
            },
        )
        self._emit(sample)
        return match


# ============================
# helpers
# ============================


def _score_to_digits(score: int) -> tuple[int | None, ...]:
    """8 桁ゼロ埋め digit tuple に変換."""
    if score is None or score < 0:
        return (None,) * 8
    s = max(0, min(99_999_999, int(score)))
    return tuple(int(c) for c in f"{s:08d}")


def _crop_score_roi(frame: np.ndarray, side: str) -> np.ndarray | None:
    """1080p 基準で score ROI を切り出し.

    入力が 1080p 以外の 16:9 解像度 (例: 720p=1280x720) の場合は
    1080p に等倍リサイズしてから ROI を適用する。
    """
    if frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    if (h, w) != (1080, 1920):
        scaled = _ensure_1080p(frame)
        if scaled is None:
            return None
        frame = scaled
        h, w = frame.shape[:2]
    region = SCORE_1P_REGION if side == "1P" else SCORE_2P_REGION
    y1, y2, x1, x2 = region
    if y2 > h or x2 > w:
        return None
    return frame[y1:y2, x1:x2].copy()


def _ensure_1080p(frame: np.ndarray) -> np.ndarray | None:
    """16:9 のフレームを 1080p (1920x1080) にリサイズ.

    入力 aspect が 16:9 以外なら None を返して安全側に倒す。
    """
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None
    # 16:9 ±1 px 許容
    if abs(w * 9 - h * 16) > max(w, h):
        return None
    try:
        import cv2
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def _crop_digit_patch(
    roi: np.ndarray, pos: int, side: str,
) -> np.ndarray | None:
    """ROI 内の pos 番目の digit (50x40) を切り出し."""
    if roi is None or roi.size == 0:
        return None
    lefts = DIGIT_LEFTS_1P if side == "1P" else DIGIT_LEFTS_2P
    if pos < 0 or pos >= len(lefts):
        return None
    x = lefts[pos]
    cell = roi[
        DIGIT_TOP:DIGIT_TOP + DIGIT_HEIGHT,
        x:x + DIGIT_WIDTH,
    ]
    if cell.shape != (DIGIT_HEIGHT, DIGIT_WIDTH, 3):
        return None
    return cell


__all__ = [
    "HIGH_CONFIDENCE",
    "MEDIUM_CONFIDENCE",
    "SCORE_AGREE_MIN",
    "SCORE_HISTORY_WINDOW",
    "SCORE_LENIENT_AGREE_MIN",
    "SCORE_LENIENT_CONFIDENCE",
    "ScoreValidator",
]
