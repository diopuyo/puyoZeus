"""中央テロップ (試合中継「チャレンジャー リーグ」等) の検出と被覆判定。

ぷよぷよe スポーツ大会動画では試合終了告知や次試合誘導のため、画面中央に
固定テロップが数秒〜数十秒表示される。これが盤面の中央右側に被ると、
HSV/CNN がぷよを誤認識する (m27 で実証済)。

機能:
    - is_visible(): テロップ表示中フラグ
    - detect(): bbox 込みで詳細結果を返す (V3.1 追加)
    - cells_covered(region): 指定盤面 region のうち被覆セル {(row, col)}
      を返す (V3.1 追加、ImageReader 統合用)

利用例:
    detector = TelopDetector.load_default()
    result = detector.detect(frame_bgr)
    if result.is_visible:
        covered = detector.cells_covered(p1_region, frame_shape=frame_bgr.shape)
        # covered 内のセルは COLOR_UNKNOWN として扱う
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS

# 既定テンプレートディレクトリ
DEFAULT_TEMPLATE_DIR: Path = Path("models/ui_templates")
# テロップテンプレ名 prefix (telop_*.png をすべて読み込む)
TELOP_PREFIX: str = "telop_"

# NCC マッチ閾値 (0-1)
DEFAULT_NCC_THRESHOLD: float = 0.55

# 検索する画面領域 (1920x1080 基準)
# テロップは画面中央〜やや上に表示される。盤面領域 (左右) は除外。
SEARCH_X: int = 600
SEARCH_Y: int = 300
SEARCH_W: int = 720
SEARCH_H: int = 400


@dataclass(frozen=True)
class TelopResult:
    """検出結果。

    bbox: (x, y, w, h) 画面座標系でのテロップ矩形。is_visible=False のとき None。
    """
    is_visible: bool
    template_name: str | None
    score: float
    bbox: tuple[int, int, int, int] | None = None


class TelopDetector:
    """中央テロップを NCC マッチで検出する。"""

    def __init__(
        self,
        templates: dict[str, np.ndarray],
        threshold: float = DEFAULT_NCC_THRESHOLD,
    ) -> None:
        self._templates = templates
        self._threshold = threshold

    @classmethod
    def load_default(
        cls,
        template_dir: Path = DEFAULT_TEMPLATE_DIR,
        threshold: float = DEFAULT_NCC_THRESHOLD,
    ) -> "TelopDetector":
        """既定ディレクトリから telop_*.png を読み込む。"""
        templates: dict[str, np.ndarray] = {}
        if template_dir.exists():
            for p in sorted(template_dir.glob(f"{TELOP_PREFIX}*.png")):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                templates[p.stem] = gray
        return cls(templates=templates, threshold=threshold)

    def detect(self, frame_bgr: np.ndarray) -> TelopResult:
        """フレーム中央領域に対してテンプレートマッチ。最大スコアと bbox を返す。"""
        if not self._templates or frame_bgr is None or frame_bgr.size == 0:
            return TelopResult(
                is_visible=False, template_name=None, score=0.0, bbox=None,
            )
        h, w = frame_bgr.shape[:2]
        # 検索範囲を画像サイズで clamp
        x1 = max(0, min(SEARCH_X, w - 1))
        y1 = max(0, min(SEARCH_Y, h - 1))
        x2 = max(x1 + 1, min(SEARCH_X + SEARCH_W, w))
        y2 = max(y1 + 1, min(SEARCH_Y + SEARCH_H, h))
        roi = frame_bgr[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        best_name: str | None = None
        best_score: float = -1.0
        best_bbox: tuple[int, int, int, int] | None = None
        for name, tmpl in self._templates.items():
            tH, tW = tmpl.shape[:2]
            if roi_gray.shape[0] < tH or roi_gray.shape[1] < tW:
                continue
            result = cv2.matchTemplate(roi_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = float(max_val)
                best_name = name
                # max_loc は ROI 座標系。フレーム座標系へ変換
                bx = x1 + max_loc[0]
                by = y1 + max_loc[1]
                best_bbox = (bx, by, tW, tH)

        is_vis = best_score >= self._threshold
        return TelopResult(
            is_visible=is_vis,
            template_name=best_name,
            score=best_score,
            bbox=best_bbox if is_vis else None,
        )

    def is_visible(self, frame_bgr: np.ndarray) -> bool:
        """簡易メソッド。"""
        return self.detect(frame_bgr).is_visible

    @staticmethod
    def cells_covered_by_bbox(
        bbox: tuple[int, int, int, int],
        region: "BoardRegion",
    ) -> set[tuple[int, int]]:
        """テロップ bbox に被覆される盤面 region のセル {(row, col)} を返す。

        判定: セルの sample_rect が bbox と矩形重複するか。少しでも被れば被覆扱い。
        隠し段 (row < HIDDEN_ROWS) は対象外。

        Args:
            bbox: (x, y, w, h) フレーム座標系。
            region: 1P または 2P の BoardRegion。

        Returns:
            被覆されたセルの (row, col) 集合。
        """
        bx, by, bw, bh = bbox
        bx2 = bx + bw
        by2 = by + bh

        covered: set[tuple[int, int]] = set()
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                cx1, cy1, cx2, cy2 = region.cell_sample_rect(row, col)
                # 矩形重複チェック
                if cx2 <= bx or cx1 >= bx2:
                    continue
                if cy2 <= by or cy1 >= by2:
                    continue
                covered.add((row, col))
        return covered


__all__ = [
    "DEFAULT_NCC_THRESHOLD",
    "DEFAULT_TEMPLATE_DIR",
    "SEARCH_H",
    "SEARCH_W",
    "SEARCH_X",
    "SEARCH_Y",
    "TELOP_PREFIX",
    "TelopDetector",
    "TelopResult",
]
