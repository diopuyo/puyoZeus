"""
ネクスト・ダブルネクスト puyo 検出（1P 側）。

画面中央上部の 1P プレビュー枠から 4 セル（next pair × 2 + dnext pair × 2）を
切り出して色分類する。

色分類は CNN + HSV ハイブリッドで青背景バイアスを抑制する:
    1. 中心ピクセルから HSV 主要色を計算（背景の青/水色 は除外）
    2. CNN 予測との合議: HSV が確信できる色を提示すれば CNN を上書き

ROI 座標（1920×1080 前提、video_01 / video_02 で共通確認済）:
    NEXT 大 (top):    y=160-235, x=710-785
    NEXT 大 (bot):    y=240-315, x=710-785
    DNEXT 小 (top):   y=335-385, x=775-835
    DNEXT 小 (bot):   y=395-445, x=775-835

NEXT pair はサイズ大、DNEXT pair はやや小さく右にオフセット（L 字配置）。

使い方:
    from src.next_detector import NextDetector
    det = NextDetector.load_default()
    result = det.detect(frame)
    print(result.next_pair)    # (top_color, bot_color)
    print(result.dnext_pair)   # (top_color, bot_color)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.board import (
    COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_YELLOW,
)

# 1P 側 ROI 座標（v10）
# (y1, y2, x1, x2)
ROI_1P_NEXT_TOP: tuple[int, int, int, int] = (162, 237, 710, 785)
ROI_1P_NEXT_BOT: tuple[int, int, int, int] = (222, 297, 710, 785)
ROI_1P_DNEXT_TOP: tuple[int, int, int, int] = (293, 343, 765, 815)
ROI_1P_DNEXT_BOT: tuple[int, int, int, int] = (340, 390, 765, 815)

# 2P 側 ROI 座標（1P を画面中央 x=960 で水平ミラー）
# 1P NEXT center x=747 → 2P NEXT center x=1173 (= 2*960-747)
# 1P DNEXT center x=790 → 2P DNEXT center x=1130 (= 2*960-790)
ROI_2P_NEXT_TOP: tuple[int, int, int, int] = (162, 237, 1135, 1210)
ROI_2P_NEXT_BOT: tuple[int, int, int, int] = (222, 297, 1135, 1210)
ROI_2P_DNEXT_TOP: tuple[int, int, int, int] = (293, 343, 1105, 1155)
ROI_2P_DNEXT_BOT: tuple[int, int, int, int] = (340, 390, 1105, 1155)

# 分類時は ROI の中心 80% だけを使う（CNN 用）
INNER_CROP_RATIO: float = 0.80
# HSV 投票時はさらに絞り込み、puyo 中心ピクセルのみを使う
INNER_CROP_RATIO_HSV: float = 0.55

# HSV ベース色分類のルール: (hue 範囲, 最小彩度, 最小明度, 色コード)
HSV_COLOR_RULES: tuple[tuple[int, int, int, int, int], ...] = (
    (0,   8,   100, 80,  COLOR_RED),     # 赤 (hue 0-8)
    (170, 179, 100, 80,  COLOR_RED),     # 赤 (wrap、hue 170-179)
    (35,  75,  80,  60,  COLOR_GREEN),   # 緑
    (20,  34,  80,  100, COLOR_YELLOW),  # 黄
    (130, 165, 80,  60,  COLOR_PURPLE),  # 紫
    (110, 129, 80,  60,  COLOR_BLUE),    # 真の青ぷよ（背景より暗い）
)
HSV_VOTE_MIN_RATIO: float = 0.10  # この比率を超えれば「色あり」と判定

# サイド別背景: 1P は青、2P は赤系の dotted パターン
# このピクセルは puyo ではないので除外する
BG_HUE_RANGES_1P: tuple[tuple[int, int], ...] = ((92, 110),)        # 水色
BG_HUE_RANGES_2P: tuple[tuple[int, int], ...] = ((0, 12), (165, 179))  # ピンク/赤系

# 黄色は背景の赤と hue が近いので 2P では追加で saturation 下限を上げる
HSV_COLOR_RULES_2P: tuple[tuple[int, int, int, int, int], ...] = (
    (0,   8,   200, 60,  COLOR_RED),     # 赤 (高彩度のみ。背景ピンクは彩度低)
    (170, 179, 200, 60,  COLOR_RED),
    (35,  75,  80,  60,  COLOR_GREEN),
    (15,  34,  150, 120, COLOR_YELLOW),  # 黄 (彩度+明度両方高)
    (130, 165, 80,  60,  COLOR_PURPLE),
    (95,  129, 80,  60,  COLOR_BLUE),    # 青ぷよ
)


def hsv_dominant_color(
    bgr_patch: np.ndarray,
    side: str = "1P",
) -> int | None:
    """
    HSV ベースで puyo の主要色を返す。サイドに応じて背景を除外。

    Args:
        bgr_patch: BGR uint8 画像
        side: "1P" (青背景) or "2P" (赤系背景)

    戻り値:
        色コード、または None（puyo がほぼない＝空/不明）
    """
    if bgr_patch is None or bgr_patch.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    total = h_ch.size

    rules = HSV_COLOR_RULES_2P if side == "2P" else HSV_COLOR_RULES

    votes: dict[int, int] = {}
    for hmin, hmax, smin, vmin, code in rules:
        mask = (h_ch >= hmin) & (h_ch <= hmax) & (s_ch >= smin) & (v_ch >= vmin)
        cnt = int(mask.sum())
        if cnt > 0:
            votes[code] = votes.get(code, 0) + cnt

    if not votes:
        return None
    best_code, best_cnt = max(votes.items(), key=lambda kv: kv[1])
    if best_cnt / total < HSV_VOTE_MIN_RATIO:
        return None
    return best_code


@dataclass(frozen=True)
class NextDetectionResult:
    """ネクスト・ダブルネクストの検出結果（1 サイド分）。"""
    next_top: int          # NEXT 大ペア上のぷよ色
    next_bot: int          # NEXT 大ペア下のぷよ色
    dnext_top: int         # DNEXT 小ペア上のぷよ色
    dnext_bot: int         # DNEXT 小ペア下のぷよ色

    @property
    def next_pair(self) -> tuple[int, int]:
        return (self.next_top, self.next_bot)

    @property
    def dnext_pair(self) -> tuple[int, int]:
        return (self.dnext_top, self.dnext_bot)


@dataclass(frozen=True)
class NextDetectionBothResult:
    """1P / 2P 両側の検出結果。"""
    p1: NextDetectionResult
    p2: NextDetectionResult

    @property
    def colors_agree(self) -> bool:
        """両側のネクスト・ダブルネクストの色が一致するか（同じツモを見ているため通常一致）。"""
        return (
            self.p1.next_pair == self.p2.next_pair
            and self.p1.dnext_pair == self.p2.dnext_pair
        )


class NextDetector:
    """1P 側のネクスト・ダブルネクスト検出器。"""

    def __init__(
        self, classifier,
        centroid_classifier=None,
        centroid_path: Path | None = None,
    ) -> None:
        """
        Args:
            classifier: classify(bgr_patch)->int を実装するもの
                通常は GatedCnnClassifier(CnnPatchClassifier.load(...))
            centroid_classifier: NextPairCentroid (W9-G、平均色 1-NN)。
                None なら centroid 統合なし (従来動作)。
            centroid_path: centroid_classifier 未指定時、ここから自動 load。
        """
        if not hasattr(classifier, "classify"):
            raise TypeError("classifier は classify(patch)->int を実装する必要がある")
        self._classifier = classifier
        self._centroid = centroid_classifier
        if self._centroid is None and centroid_path is not None:
            try:
                from src.centroid_classifier import CentroidClassifier
                cen = CentroidClassifier()
                cen.load(centroid_path)
                if cen.centroids:
                    self._centroid = cen
            except Exception:
                self._centroid = None

    @classmethod
    def load_default(cls, cnn_path: Path = Path("models/cnn_global_best.pt")) -> "NextDetector":
        """既定の CNN + ゲート + centroid (あれば) でデフォルト構築。"""
        from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
        cnn = CnnPatchClassifier.load(cnn_path)
        gated = GatedCnnClassifier(color_classifier=cnn)
        return cls(
            classifier=gated,
            centroid_path=Path("models/next_pair_centroid_v1.npz"),
        )

    def _extract(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        y1, y2, x1, x2 = roi
        h, w = frame.shape[:2]
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        return frame[y1:y2, x1:x2]

    def _inner_crop(
        self,
        patch: np.ndarray,
        ratio: float = INNER_CROP_RATIO,
    ) -> np.ndarray:
        """中心 ratio 分だけ切り出し（背景の影響を低減）。"""
        if patch.size == 0:
            return patch
        h, w = patch.shape[:2]
        cy, cx = h // 2, w // 2
        rh = max(1, int(h * ratio / 2))
        rw = max(1, int(w * ratio / 2))
        y1 = max(0, cy - rh)
        y2 = min(h, cy + rh)
        x1 = max(0, cx - rw)
        x2 = min(w, cx + rw)
        return patch[y1:y2, x1:x2]

    def _classify_side(
        self,
        frame: np.ndarray,
        rois: tuple[tuple[str, tuple[int, int, int, int]], ...],
        side: str = "1P",
    ) -> NextDetectionResult:
        labels: dict[str, int] = {}
        for key, roi in rois:
            full = self._extract(frame, roi)
            cnn_patch = self._inner_crop(full, INNER_CROP_RATIO)
            hsv_patch = self._inner_crop(full, INNER_CROP_RATIO_HSV)
            cnn_code = self._classifier.classify(cnn_patch)
            hsv_code = hsv_dominant_color(hsv_patch, side=side)
            # W9-G: NextPairCentroid (平均色 1-NN) を 3 つ目の signal として使用
            cen_code = None
            if self._centroid is not None and full.size > 0:
                try:
                    cen_code = int(self._centroid.classify(full))
                except Exception:
                    cen_code = None

            # 多数決: 3 signals (CNN, HSV, centroid) のうち多数派を採用
            # 全部違う場合は HSV (背景補正済) > centroid > CNN の順
            votes: dict[int, int] = {}
            for c in (cnn_code, hsv_code, cen_code):
                if c is None:
                    continue
                votes[c] = votes.get(c, 0) + 1
            if votes:
                # 2 票以上は確定、1 票しかないときは優先順位
                max_votes = max(votes.values())
                top = [c for c, n in votes.items() if n == max_votes]
                if len(top) == 1:
                    labels[key] = top[0]
                elif hsv_code is not None and hsv_code in top:
                    labels[key] = hsv_code
                elif cen_code is not None and cen_code in top:
                    labels[key] = cen_code
                else:
                    labels[key] = cnn_code
            else:
                labels[key] = cnn_code
        return NextDetectionResult(
            next_top=labels["next_top"], next_bot=labels["next_bot"],
            dnext_top=labels["dnext_top"], dnext_bot=labels["dnext_bot"],
        )

    @staticmethod
    def _check_resolution(frame: np.ndarray) -> None:
        if frame is None or frame.ndim != 3:
            raise ValueError("frame is invalid")
        h, w = frame.shape[:2]
        if (h, w) != (1080, 1920):
            raise ValueError(f"解像度不一致: {(h, w)}, 期待 (1080, 1920)")

    def detect(self, frame: np.ndarray) -> NextDetectionResult:
        """1P 側のフレームから 4 セル抽出 → 色分類して結果を返す。"""
        self._check_resolution(frame)
        return self._classify_side(frame, (
            ("next_top", ROI_1P_NEXT_TOP),
            ("next_bot", ROI_1P_NEXT_BOT),
            ("dnext_top", ROI_1P_DNEXT_TOP),
            ("dnext_bot", ROI_1P_DNEXT_BOT),
        ), side="1P")

    def detect_2p(self, frame: np.ndarray) -> NextDetectionResult:
        """2P 側のフレームから 4 セル抽出。"""
        self._check_resolution(frame)
        return self._classify_side(frame, (
            ("next_top", ROI_2P_NEXT_TOP),
            ("next_bot", ROI_2P_NEXT_BOT),
            ("dnext_top", ROI_2P_DNEXT_TOP),
            ("dnext_bot", ROI_2P_DNEXT_BOT),
        ), side="2P")

    def detect_both(self, frame: np.ndarray) -> NextDetectionBothResult:
        """1P / 2P 両側を同時検出して返す。両者が一致するかも見られる。"""
        return NextDetectionBothResult(
            p1=self.detect(frame),
            p2=self.detect_2p(frame),
        )

    def extract_patches(self, frame: np.ndarray, side: str = "1P") -> dict[str, np.ndarray]:
        """ROI のパッチ画像を辞書で返す（レビュー用）。side='1P' or '2P'。"""
        if side == "2P":
            return {
                "next_top": self._extract(frame, ROI_2P_NEXT_TOP),
                "next_bot": self._extract(frame, ROI_2P_NEXT_BOT),
                "dnext_top": self._extract(frame, ROI_2P_DNEXT_TOP),
                "dnext_bot": self._extract(frame, ROI_2P_DNEXT_BOT),
            }
        return {
            "next_top": self._extract(frame, ROI_1P_NEXT_TOP),
            "next_bot": self._extract(frame, ROI_1P_NEXT_BOT),
            "dnext_top": self._extract(frame, ROI_1P_DNEXT_TOP),
            "dnext_bot": self._extract(frame, ROI_1P_DNEXT_BOT),
        }

    def detect_stable(
        self,
        frames: list[np.ndarray],
    ) -> NextDetectionResult | None:
        """
        複数フレームで判定、全て同一結果の場合のみ返す（静止状態のみ採用）。

        ぷよぷよ動画ではダブルネクスト→ネクスト→盤面 の遷移時にパッチが
        中間状態（ぷよ移動中）になり判定がぶれる。連続フレームで一致した
        判定だけを採用することで静止状態のみを抽出する。

        Args:
            frames: 連続フレームのリスト（同一秒数間隔推奨、最低 2 枚）

        Returns:
            全フレームで一致した判定 / None（遷移中で判定保留）
        """
        if not frames:
            return None
        results = [self.detect(f) for f in frames]
        first = results[0]
        for r in results[1:]:
            if (r.next_top, r.next_bot, r.dnext_top, r.dnext_bot) != \
               (first.next_top, first.next_bot, first.dnext_top, first.dnext_bot):
                return None
        return first
