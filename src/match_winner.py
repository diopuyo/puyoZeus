"""
試合の勝敗判定モジュール。

原理:
    画面中央下部の "数値★WIN★数値" パネルの数値が試合終了で増える側 = 勝者。
    OCR 不要、数値画像の差分で判定する。

    試合 N の終了 → 試合 N+1 の開始の間に、勝者側の数値が +1 される。
    そのため:
        - frame_at_match_N_start の数値画像
        - frame_at_match_N+1_start の数値画像
    を比較し、変化した側が試合 N の勝者。

    最終試合は frame_after_panel_end も比較対象として使える。

使い方:
    detector = MatchWinnerDetector.load_default()
    winner = detector.detect_winner(cap, t_at_match_start, t_at_next_match_start)
    # winner: "1P" / "2P" / None
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.win_panel import (
    NUMBER_LEFT_X,
    NUMBER_RIGHT_X,
    NUMBER_Y,
    WinPanelDetector,
)

# 数値画像の指紋ハッシュ計算: 16×16 二値化
SIGNATURE_SIZE: int = 16
# 同一数値とみなすハミング距離上限（256 ビット中）。これ未満は「変化なし」
DIGIT_SAME_HAMMING: int = 5
# 異なる数値と確信するハミング距離下限。これ以上で「変化あり」確定
DIGIT_DIFF_HAMMING: int = 10
# 片側が変化大、もう片側が変化小（=「明らかに非対称」）と見なす差分倍率
DIGIT_ASYMMETRY_RATIO: float = 2.5
# 非対称判定で大きい側に最低限求めるハミング距離（DIGIT_DIFF より緩い）
DIGIT_ASYMMETRY_MIN: int = 8

WinnerSide = Literal["1P", "2P"]


@dataclass(frozen=True)
class WinnerDetectionResult:
    """勝敗判定の詳細結果。"""
    winner: WinnerSide | None        # "1P" / "2P" / None (判定不能)
    left_changed: bool               # 左側 (1P) 数値が変化したか
    right_changed: bool              # 右側 (2P) 数値が変化したか
    left_hamming: int                # 左側ハミング距離
    right_hamming: int               # 右側ハミング距離


def digit_signature(patch: np.ndarray) -> np.ndarray:
    """16×16 大津二値で 256 ビット指紋を返す。"""
    if patch is None or patch.size == 0:
        return np.zeros(SIGNATURE_SIZE * SIGNATURE_SIZE, dtype=np.uint8)
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    small = cv2.resize(
        gray, (SIGNATURE_SIZE, SIGNATURE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.flatten().astype(np.uint8)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """ビット指紋のハミング距離。"""
    if a.size != b.size:
        return SIGNATURE_SIZE * SIGNATURE_SIZE
    return int(np.sum(a != b))


def extract_digit_patches(
    frame: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """フレームから左右の数値領域を切り出す（パネル存在判定なし）。"""
    if frame is None or frame.ndim != 3:
        return None, None
    h, w = frame.shape[:2]
    if (h, w) != (1080, 1920):
        return None, None
    left = frame[NUMBER_Y[0]:NUMBER_Y[1], NUMBER_LEFT_X[0]:NUMBER_LEFT_X[1]].copy()
    right = frame[NUMBER_Y[0]:NUMBER_Y[1], NUMBER_RIGHT_X[0]:NUMBER_RIGHT_X[1]].copy()
    return left, right


def compare_digit_pairs(
    left_a: np.ndarray | None,
    right_a: np.ndarray | None,
    left_b: np.ndarray | None,
    right_b: np.ndarray | None,
) -> WinnerDetectionResult:
    """
    2 時点の左右数値画像を比較して勝者を判定する。

    どちらか片側だけ「明確に変わった」ら、その側が勝者。
    両方変わった or 両方同じ → 判定不能 (None)。
    """
    if any(p is None for p in (left_a, right_a, left_b, right_b)):
        return WinnerDetectionResult(
            winner=None, left_changed=False, right_changed=False,
            left_hamming=0, right_hamming=0,
        )
    sig_la = digit_signature(left_a)
    sig_ra = digit_signature(right_a)
    sig_lb = digit_signature(left_b)
    sig_rb = digit_signature(right_b)
    dl = hamming_distance(sig_la, sig_lb)
    dr = hamming_distance(sig_ra, sig_rb)

    # 厳格判定: 片側だけ変化大（>= DIFF）かつもう片側は変化小（<= SAME）
    left_changed_strict = dl >= DIGIT_DIFF_HAMMING and dr <= DIGIT_SAME_HAMMING
    right_changed_strict = dr >= DIGIT_DIFF_HAMMING and dl <= DIGIT_SAME_HAMMING

    # 非対称判定: 一方が他方の RATIO 倍以上で、大きい方が ASYMMETRY_MIN 以上
    # （DIGIT_DIFF より緩く、わずかな変化でも非対称なら勝者として採用）
    asymmetric_left = (
        dl >= DIGIT_ASYMMETRY_MIN
        and dl >= dr * DIGIT_ASYMMETRY_RATIO
    )
    asymmetric_right = (
        dr >= DIGIT_ASYMMETRY_MIN
        and dr >= dl * DIGIT_ASYMMETRY_RATIO
    )
    # dr=0 または dl=0 のケースは厳格条件で吸収済み

    winner: WinnerSide | None
    if left_changed_strict or asymmetric_left:
        winner = "1P"
    elif right_changed_strict or asymmetric_right:
        winner = "2P"
    else:
        winner = None
    return WinnerDetectionResult(
        winner=winner,
        left_changed=dl >= DIGIT_DIFF_HAMMING,
        right_changed=dr >= DIGIT_DIFF_HAMMING,
        left_hamming=dl,
        right_hamming=dr,
    )


class MatchWinnerDetector:
    """動画の VideoCapture から試合の勝敗を判定する。"""

    def __init__(
        self,
        panel_detector: WinPanelDetector | None = None,
    ) -> None:
        self._panel_detector = panel_detector or WinPanelDetector.load_default()

    @classmethod
    def load_default(cls) -> "MatchWinnerDetector":
        return cls(panel_detector=WinPanelDetector.load_default())

    def _read_frame(
        self,
        cap: cv2.VideoCapture,
        t_sec: float,
    ) -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        return frame

    def _find_panel_visible_time(
        self,
        cap: cv2.VideoCapture,
        around_sec: float,
        scan_back_max: float = 30.0,
        step: float = 0.3,
    ) -> float | None:
        """指定時刻周辺でパネル可視時刻を逆方向に探す。"""
        t = around_sec
        end = around_sec - scan_back_max
        while t >= end:
            frame = self._read_frame(cap, t)
            if frame is not None:
                if self._panel_detector.detect(frame).present:
                    return t
            t -= step
        return None

    def detect_winner(
        self,
        cap: cv2.VideoCapture,
        match_start_sec: float,
        next_match_start_sec: float,
        offset_before: float = 1.0,
        offset_after: float = 1.0,
    ) -> WinnerDetectionResult:
        """
        2 試合連続の数値画像差分で勝敗判定。

        Args:
            cap: 動画キャプチャ
            match_start_sec: 試合 N の開始時刻（数値読み取り時刻）
            next_match_start_sec: 試合 N+1 の開始時刻 (= 試合 N 勝敗反映後)
            offset_before: match_start_sec からの時間オフセット（数値が安定する位置）
            offset_after: next_match_start_sec からの時間オフセット
        """
        t_a = match_start_sec + offset_before
        t_b = next_match_start_sec + offset_after
        frame_a = self._read_frame(cap, t_a)
        frame_b = self._read_frame(cap, t_b)
        left_a, right_a = extract_digit_patches(frame_a) if frame_a is not None else (None, None)
        left_b, right_b = extract_digit_patches(frame_b) if frame_b is not None else (None, None)
        return compare_digit_pairs(left_a, right_a, left_b, right_b)

    def detect_all_winners(
        self,
        cap: cv2.VideoCapture,
        match_starts: list[float],
        last_observable_sec: float,
        offset_before: float = 1.0,
    ) -> list[WinnerDetectionResult]:
        """
        試合開始時刻リストから全試合の勝敗を判定する。

        試合 N の判定: match_starts[N] と match_starts[N+1] の数値比較。
        最終試合は match_starts[-1] と last_observable_sec の比較。

        Args:
            cap: 動画キャプチャ
            match_starts: 各試合の開始秒
            last_observable_sec: 最終試合の判定用、パネルがまだ見える最後の時刻
            offset_before: 数値読取りの時間オフセット
        """
        if not match_starts:
            return []
        # 最終試合用: パネルが可視な時刻を探索（パネル消失前の数値読取）
        panel_visible = self._find_panel_visible_time(cap, last_observable_sec)
        last_compare_t = panel_visible if panel_visible is not None else last_observable_sec
        boundaries = list(match_starts) + [last_compare_t]
        results: list[WinnerDetectionResult] = []
        for i in range(len(match_starts)):
            r = self.detect_winner(
                cap=cap,
                match_start_sec=boundaries[i],
                next_match_start_sec=boundaries[i + 1],
                offset_before=offset_before,
                offset_after=offset_before if i < len(match_starts) - 1 else 0.0,
            )
            results.append(r)
        return results
