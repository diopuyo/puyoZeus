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

    def _prepare_side(
        self,
        frame: np.ndarray,
        rois: tuple[tuple[str, tuple[int, int, int, int]], ...],
    ) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """ROI ごとに (key, full, cnn_patch, hsv_patch) を切り出す。

        高速化 (2026-07-31): CNN 分類をサイド横断で 1 バッチに束ねるため、
        切り出しと分類を分離した。切り出し内容は従来と同一。

        Args:
            frame: 1920x1080 の BGR フレーム。
            rois: (key, ROI) の並び。

        Returns:
            (key, full パッチ, CNN 用パッチ, HSV 用パッチ) のリスト。
        """
        return [
            (
                key,
                (full := self._extract(frame, roi)),
                self._inner_crop(full, INNER_CROP_RATIO),
                self._inner_crop(full, INNER_CROP_RATIO_HSV),
            )
            for key, roi in rois
        ]

    def _classify_cnn_batch(self, cnn_patches: list[np.ndarray]) -> list[int]:
        """CNN 分類をまとめて実行する (classify_batch があれば 1 回で)。

        高速化 (2026-07-31): 旧実装は 4 ROI x 2 サイド = **8 回の単発 CNN 呼び出し**
        で、実測 559us/回・合計 4.47ms/frame (認識全体の約12%) を占めていた。
        パッチ 1 枚の推論としては固定費が支配的なので、束ねると大きく縮む。
        盤面側は既に 156 セルを 2 回に束ねている (classify_batch)。

        classify_batch を持たない分類器 (テストのスタブ等) では従来通り
        1 枚ずつ classify する (backwards compat)。

        Args:
            cnn_patches: CNN 用パッチのリスト。

        Returns:
            各パッチの色コード (入力と同じ順序)。
        """
        if not cnn_patches:
            return []
        batch_fn = getattr(self._classifier, "classify_batch", None)
        if batch_fn is None:
            return [self._classifier.classify(p) for p in cnn_patches]
        return [int(c) for c in batch_fn(cnn_patches)]

    def _vote_side(
        self,
        prepared: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
        cnn_codes: list[int],
        side: str = "1P",
    ) -> NextDetectionResult:
        """CNN 結果を受け取って HSV / centroid と多数決を取る。

        多数決ロジックは旧 `_classify_side` からそのまま切り出したもの。

        Args:
            prepared: `_prepare_side` の出力。
            cnn_codes: 各 ROI の CNN 判定 (prepared と同じ順序)。
            side: "1P" / "2P" (HSV の背景補正に使う)。

        Returns:
            NextDetectionResult。
        """
        labels: dict[str, int] = {}
        for (key, full, _cnn_patch, hsv_patch), cnn_code in zip(prepared, cnn_codes):
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

    def _classify_side(
        self,
        frame: np.ndarray,
        rois: tuple[tuple[str, tuple[int, int, int, int]], ...],
        side: str = "1P",
    ) -> NextDetectionResult:
        """1 サイド分を切り出し → CNN 分類 → 多数決 (従来の入口を維持)。

        `detect_both` はサイド横断で CNN を束ねるためこれを経由しないが、
        `detect` / `detect_2p` 単独呼び出しの互換のために残す。
        """
        prepared = self._prepare_side(frame, rois)
        cnn_codes = self._classify_cnn_batch([p[2] for p in prepared])
        return self._vote_side(prepared, cnn_codes, side=side)

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
        return self._classify_side(frame, self.ROIS_1P, side="1P")

    def detect_2p(self, frame: np.ndarray) -> NextDetectionResult:
        """2P 側のフレームから 4 セル抽出。"""
        self._check_resolution(frame)
        return self._classify_side(frame, self.ROIS_2P, side="2P")

    # 1P / 2P の ROI 並び (detect / detect_2p / detect_both で共有)
    ROIS_1P: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
        ("next_top", ROI_1P_NEXT_TOP),
        ("next_bot", ROI_1P_NEXT_BOT),
        ("dnext_top", ROI_1P_DNEXT_TOP),
        ("dnext_bot", ROI_1P_DNEXT_BOT),
    )
    ROIS_2P: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
        ("next_top", ROI_2P_NEXT_TOP),
        ("next_bot", ROI_2P_NEXT_BOT),
        ("dnext_top", ROI_2P_DNEXT_TOP),
        ("dnext_bot", ROI_2P_DNEXT_BOT),
    )

    def detect_both(self, frame: np.ndarray) -> NextDetectionBothResult:
        """1P / 2P 両側を同時検出して返す。両者が一致するかも見られる。

        高速化 (2026-07-31): 旧実装は detect() と detect_2p() を別々に呼び、
        **1フレームで 8 回の単発 CNN 推論** (実測 559us/回、合計 4.47ms) をしていた。
        両サイドのパッチを集めて **CNN を 1 バッチに束ねる**。
        切り出し・HSV・多数決のロジックは一切変えていない。
        """
        self._check_resolution(frame)
        prep_1p = self._prepare_side(frame, self.ROIS_1P)
        prep_2p = self._prepare_side(frame, self.ROIS_2P)
        # 8 枚まとめて 1 回の CNN 推論
        codes = self._classify_cnn_batch(
            [p[2] for p in prep_1p] + [p[2] for p in prep_2p],
        )
        n1 = len(prep_1p)
        return NextDetectionBothResult(
            p1=self._vote_side(prep_1p, codes[:n1], side="1P"),
            p2=self._vote_side(prep_2p, codes[n1:], side="2P"),
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
