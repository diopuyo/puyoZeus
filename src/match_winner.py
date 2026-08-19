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
# --- 端点探索定数 (2026-08-19 追加: 最初/最後の試合の勝敗ラベル系統的欠損の修正) ---
# 最終試合の遡り探索: 近傍は細かく、遠方は粗く探す 2 段構え。
# 旧実装は上限 30 秒でリザルト画面 (勝敗数値が最後に映る時刻) に届かず、
# 実測 47/50 動画で最終試合のラベルが欠損していた。
PANEL_SCAN_BACK_FINE_SEC: float = 30.0    # 細step で探す近傍範囲 (旧上限と同値)
PANEL_SCAN_BACK_FINE_STEP: float = 0.3    # 近傍の探索刻み (旧実装と同値)
PANEL_SCAN_BACK_MAX_SEC: float = 900.0    # 遡り上限 (アウトロが数分続く動画に対応)
PANEL_SCAN_BACK_COARSE_STEP: float = 1.0  # 遠方の探索刻み (リザルト画面は数秒続くため十分)
# game 0 起点の前方探索: 動画冒頭のイントロ (パネル非表示) を飛ばして
# 最初にパネルが映る時刻を起点にする。実測 50/50 動画で t=start+1 秒は
# イントロ映像でパネル不可視 → game 0 のラベルが全滅していた。
PANEL_SCAN_FORWARD_MAX_SEC: float = 900.0  # 前方探索の上限 (イントロは数分に及び得る)
PANEL_SCAN_FORWARD_STEP: float = 1.0       # 前方探索の刻み (試合中は常時表示のため十分)
# 前方探索は次の試合開始読取時刻のこの秒数手前で打ち切る
# (game 1 の領域まで踏み込むと比較基準が壊れるため)
PANEL_SNAP_NEXT_BOUNDARY_MARGIN_SEC: float = 2.0
# 最終試合の終点読取: 勝利演出画面は星エフェクトが数字に重なり、単一フレーム
# 比較では「左右両方変化」で判定不能になる (実測 8/11 動画)。複数時刻の
# フレームをピクセル中央値で合成すると、動くエフェクトは消え静止した数字が
# 残る (静止信号 vs 移動ノイズの物理的弁別、2026-08-19 追加)。
LAST_END_MEDIAN_OFFSETS: tuple[float, ...] = (0.0, -0.5, -1.0, -1.5, -2.0)
# 最終試合限定の二次判別 (NCC、2026-08-19 追加): 勝利演出はパネル全体が
# 発光し、二値指紋ハミングは左右両側とも大きくなる (照明変化に弱い)。
# TM_CCOEFF_NORMED (平均差し引き=輝度・コントラストシフトに不変) なら
# 「静止した数字 (敗者側)」は発光下でも高相関を保つ。実測分離ギャップ
# (真値既知の11動画、scripts/_diag_lastgame_ncc_2026-08-19.py):
#   静止側 (敗者、数字不変) の NCC 最小値 = 0.842
#   変化側 (勝者) / 多試合スパンの NCC 最大値 = 0.756
# 0.756 < 0.80 < 0.842 に収まるラウンド値を採用 (シーン逆算でなく分離
# ギャップから固定)。ギャップ下限 0.10 は「両側とも静止 (=試合が実際には
# 終わっていない)」の誤発火防止 (真の単一増分ケースの実測最小ギャップ
# 0.139 より下、両側静止なら差はほぼ 0)。
LAST_END_NCC_STATIC_MIN: float = 0.80
LAST_END_NCC_GAP_MIN: float = 0.10
# パネル系統が「物理的に読取不能」だったことを示す番兵値。呼び出し側
# (scripts/collect_boards_lean.py の 2 系統一致判定) はこの値を受けたとき、
# score 単独へ緩和するのではなく窒息判定 (_winner_by_survival) のみへ
# フォールバックする (score 単独緩和は断片化試合で 44.8% 誤ラベルの実測あり)。
PANEL_UNAVAILABLE: str = "PANEL_UNAVAILABLE"
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
    # パネルが物理的に画面に映らず読取自体が不能だったか (2026-08-19 追加、
    # 既定 False で後方互換)。True のとき winner は必ず None であり、
    # 「比較したが曖昧だった」(False) と区別できる。
    panel_unavailable: bool = False


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


def digit_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """同サイズ数値パッチ間の正規化相互相関 (TM_CCOEFF_NORMED、2026-08-19)。

    平均差し引き正規化により輝度・コントラストの一様なシフトに不変。
    勝利演出のパネル発光下でも「同じ数字」なら高相関を保つ。
    """
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return 0.0
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32) if b.ndim == 3 else b.astype(np.float32)
    v = cv2.matchTemplate(ga, gb, cv2.TM_CCOEFF_NORMED)
    return float(v[0, 0])


def last_winner_by_ncc(
    left_a: np.ndarray | None,
    right_a: np.ndarray | None,
    left_b: np.ndarray | None,
    right_b: np.ndarray | None,
) -> WinnerSide | None:
    """最終試合限定の二次判別: NCC で「静止側=敗者」を特定する (2026-08-19)。

    条件 (定数の根拠は LAST_END_NCC_* のコメント参照):
    - 高い側の NCC >= LAST_END_NCC_STATIC_MIN (敗者の数字が本当に静止)
    - 両側の NCC 差 >= LAST_END_NCC_GAP_MIN (両側静止=試合未終了の除外)
    どちらかを満たさなければ None (多試合スパン等は両側低 NCC で不発)。
    """
    if any(p is None for p in (left_a, right_a, left_b, right_b)):
        return None
    ncc_l = digit_ncc(left_a, left_b)
    ncc_r = digit_ncc(right_a, right_b)
    hi = max(ncc_l, ncc_r)
    if hi < LAST_END_NCC_STATIC_MIN or abs(ncc_l - ncc_r) < LAST_END_NCC_GAP_MIN:
        return None
    # 低い側 = 数字が変わった側 = 勝者
    return "1P" if ncc_l < ncc_r else "2P"


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
        scan_back_max: float = PANEL_SCAN_BACK_MAX_SEC,
        step: float = PANEL_SCAN_BACK_FINE_STEP,
        coarse_after_sec: float = PANEL_SCAN_BACK_FINE_SEC,
        coarse_step: float = PANEL_SCAN_BACK_COARSE_STEP,
        not_before_sec: float | None = None,
    ) -> float | None:
        """指定時刻周辺でパネル可視時刻を逆方向に探す。

        2026-08-19 変更: 上限を 30 秒 → PANEL_SCAN_BACK_MAX_SEC に延長
        (最終試合のリザルト画面まで届かず 47/50 動画でラベル欠損した修正)。
        近傍 coarse_after_sec 秒は従来通り細かい step、それより遠方は
        coarse_step の粗い刻みで探索コストを抑える。

        Args:
            not_before_sec: これより過去は探索しない (最終試合の開始時刻を
                渡すことで、意味のない過去への遡りを打ち切る)。
        """
        t = around_sec
        end = around_sec - scan_back_max
        if not_before_sec is not None:
            end = max(end, not_before_sec)
        fine_end = around_sec - coarse_after_sec
        while t >= end:
            frame = self._read_frame(cap, t)
            if frame is not None:
                if self._panel_detector.detect(frame).present:
                    return t
            t -= step if t > fine_end else coarse_step
        return None

    def _find_panel_visible_time_forward(
        self,
        cap: cv2.VideoCapture,
        from_sec: float,
        until_sec: float,
        step: float = PANEL_SCAN_FORWARD_STEP,
    ) -> float | None:
        """from_sec から前方 (未来方向) にパネル可視時刻を探す (2026-08-19 追加)。

        game 0 の読取起点補正用: 動画冒頭のイントロ区間を飛ばし、最初に
        WIN★パネルが映る時刻を返す。見つからなければ None。
        """
        t = from_sec
        while t <= until_sec:
            frame = self._read_frame(cap, t)
            if frame is not None:
                if self._panel_detector.detect(frame).present:
                    return t
            t += step
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

    def _resolve_read_times(
        self,
        cap: cv2.VideoCapture,
        match_starts: list[float],
        last_observable_sec: float,
        offset_before: float,
    ) -> list[float | None]:
        """各比較時点の実読取時刻を決める (2026-08-19 追加)。

        戻り値は len(match_starts)+1 要素。[i] は試合 i 開始時点の読取時刻、
        末尾は最終試合終了後 (勝敗数値反映後) の読取時刻。None は
        「パネルが物理的に映る時刻が見つからず読取不能」を意味する。
        """
        reads: list[float | None] = [t + offset_before for t in match_starts]
        # game 0 の起点補正: 動画冒頭はイントロ映像でパネルが映らないことが
        # 多い (実測 50/50 動画で欠損)。読取予定時刻にパネルが映っていなければ
        # 最初にパネルが映る時刻まで前方に補正する。映っていれば従来と同一
        # 時刻を使う (挙動不変)。
        t0 = reads[0]
        frame0 = self._read_frame(cap, t0)
        if frame0 is None or not self._panel_detector.detect(frame0).present:
            limit = t0 + PANEL_SCAN_FORWARD_MAX_SEC
            if len(reads) >= 2:
                # 次の試合の読取時刻より手前で打ち切る (game 1 の数値を
                # game 0 の基準にすると誤った勝者が出るため)
                limit = min(
                    limit, reads[1] - PANEL_SNAP_NEXT_BOUNDARY_MARGIN_SEC,
                )
            reads[0] = self._find_panel_visible_time_forward(
                cap, t0 + PANEL_SCAN_FORWARD_STEP, limit,
            )
        # 最終試合の終点: パネルが映る最後の時刻へ遡る (上限延長、2026-08-19)。
        # 最終試合の開始読取時刻より前しか見つからない場合は読取不能扱い
        # (前試合中の数値と比較すると誤った勝者が出るため)。
        last_read_floor = match_starts[-1] + offset_before
        # 遡り範囲は最終試合の開始読取時刻まで (アウトロが PANEL_SCAN_BACK_
        # MAX_SEC を超える動画でも floor が物理的な下限になるため、上限は
        # floor までの全域に広げる。c109 実測: アウトロ 900 秒超で欠損)
        scan_back = max(
            PANEL_SCAN_BACK_MAX_SEC, last_observable_sec - last_read_floor,
        )
        t_end = self._find_panel_visible_time(
            cap, last_observable_sec,
            scan_back_max=scan_back, not_before_sec=last_read_floor,
        )
        if t_end is not None and t_end < last_read_floor:
            t_end = None
        reads.append(t_end)
        return reads

    def _median_digit_patches(
        self,
        cap: cv2.VideoCapture,
        t_center: float,
        floor_sec: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """t_center 周辺の複数フレームをピクセル中央値合成した数値領域を返す。

        勝利演出の星エフェクト (移動ノイズ) を中央値で除去し、静止した数字
        だけを残す (2026-08-19 追加、最終試合の終点読取専用)。パネル可視の
        フレームのみを使い、1 枚も取れなければ (None, None)。
        """
        lefts: list[np.ndarray] = []
        rights: list[np.ndarray] = []
        for off in LAST_END_MEDIAN_OFFSETS:
            t = t_center + off
            if t < floor_sec:
                continue
            frame = self._read_frame(cap, t)
            if frame is None or not self._panel_detector.detect(frame).present:
                continue
            left, right = extract_digit_patches(frame)
            if left is not None and right is not None:
                lefts.append(left)
                rights.append(right)
        if not lefts:
            return None, None
        left_med = np.median(np.stack(lefts), axis=0).astype(np.uint8)
        right_med = np.median(np.stack(rights), axis=0).astype(np.uint8)
        return left_med, right_med

    def _detect_last_winner(
        self,
        cap: cv2.VideoCapture,
        t_a: float,
        t_b: float,
        floor_sec: float,
    ) -> WinnerDetectionResult:
        """最終試合の勝敗判定 (終点=勝利演出画面のエフェクト耐性つき)。

        始点 t_a は試合中の綺麗な画面 (単一フレーム)、終点 t_b 側は
        _median_digit_patches でエフェクト除去した合成画像を使う。
        一次判別 (二値指紋ハミング) が判定不能のときのみ、輝度不変の
        NCC 二次判別 (last_winner_by_ncc) を試す (勝利演出のパネル発光で
        ハミングが左右両側とも大きくなる実測 5/11 動画の救済、2026-08-19)。
        """
        frame_a = self._read_frame(cap, t_a)
        left_a, right_a = (
            extract_digit_patches(frame_a) if frame_a is not None else (None, None)
        )
        left_b, right_b = self._median_digit_patches(cap, t_b, floor_sec)
        result = compare_digit_pairs(left_a, right_a, left_b, right_b)
        if result.winner is not None:
            return result
        ncc_winner = last_winner_by_ncc(left_a, right_a, left_b, right_b)
        if ncc_winner is None:
            return result
        return WinnerDetectionResult(
            winner=ncc_winner,
            left_changed=ncc_winner == "1P",
            right_changed=ncc_winner == "2P",
            left_hamming=result.left_hamming,
            right_hamming=result.right_hamming,
        )

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
        最終試合は match_starts[-1] と「パネルが最後に映る時刻」の比較。

        2026-08-19 変更 (最初/最後の試合のラベル系統的欠損の修正):
        - game 0 の読取起点はパネル不可視なら前方補正する (イントロ対策)
        - 最終試合の遡り探索を延長 (30 秒上限 → 最終試合開始まで全域)
        - 最終試合の終点は複数フレーム中央値合成で勝利演出エフェクトを除去
        - 読取時刻が見つからない試合はパネル画像の比較自体を行わず
          panel_unavailable=True の結果を返す (旧実装は不可視フレームの
          切り出し画像同士を比較しており、原理的に無意味だった)

        Args:
            cap: 動画キャプチャ
            match_starts: 各試合の開始秒
            last_observable_sec: 最終試合の判定用、パネルがまだ見える最後の時刻
            offset_before: 数値読取りの時間オフセット
        """
        if not match_starts:
            return []
        reads = self._resolve_read_times(
            cap, match_starts, last_observable_sec, offset_before,
        )
        last_read_floor = match_starts[-1] + offset_before
        results: list[WinnerDetectionResult] = []
        for i in range(len(match_starts)):
            t_a = reads[i]
            t_b = reads[i + 1]
            if t_a is None or t_b is None:
                results.append(WinnerDetectionResult(
                    winner=None, left_changed=False, right_changed=False,
                    left_hamming=0, right_hamming=0, panel_unavailable=True,
                ))
                continue
            if i == len(match_starts) - 1:
                results.append(
                    self._detect_last_winner(cap, t_a, t_b, last_read_floor),
                )
                continue
            r = self.detect_winner(
                cap=cap,
                match_start_sec=t_a,
                next_match_start_sec=t_b,
                offset_before=0.0,
                offset_after=0.0,
            )
            results.append(r)
        return results
