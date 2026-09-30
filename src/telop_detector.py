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
import os
from pathlib import Path

import cv2
import numpy as np

from src.prepared_template import PreparedImage, PreparedTemplate

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

# 高速経路 (既定 OFF、環境変数 PUYO_FAST_TELOP=1): 1/4 縮小の粗スコアが床未満のテンプレートは、
# 閾値 (0.55) 未満と扱って全解像度照合を省く。粗スコア >= 床の時は従来の全解像度経路へ進むので、
# 誤って「不可視」にできるのは粗スコアが床未満なのに全解像度が閾値以上のときだけ (実測は docs/PHASE_J_PERF_2026-09-30)。
# 省いたテンプレートは template_name/score に寄与しない (これらの値は is_visible=False では未使用)。
FAST_ENV: str = "PUYO_FAST_TELOP"
COARSE_SCALE: float = 0.25
PREFILTER_FLOOR: float = 0.30


@dataclass(frozen=True)
class TelopResult:
    """検出結果。

    bbox: (x, y, w, h) 画面座標系でのテロップ矩形。is_visible=False のとき None。
    """
    is_visible: bool
    template_name: str | None
    score: float
    bbox: tuple[int, int, int, int] | None = None


def _shrink(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, None, fx=COARSE_SCALE, fy=COARSE_SCALE, interpolation=cv2.INTER_AREA)


class TelopDetector:
    """中央テロップを NCC マッチで検出する。"""

    def __init__(
        self,
        templates: dict[str, np.ndarray],
        threshold: float = DEFAULT_NCC_THRESHOLD,
        fast: bool | None = None,
    ) -> None:
        """fast=None は環境変数 PUYO_FAST_TELOP=1 の時だけ高速経路 (既定OFF = 従来と bit-identical)。"""
        self._templates = templates
        self._prepared = {name: PreparedTemplate(value) for name, value in templates.items()}
        self._threshold = threshold
        self.fast = os.environ.get(FAST_ENV) == "1" if fast is None else fast
        self._coarse = {name: _shrink(value) for name, value in templates.items()} if self.fast else {}
        self.coarse_rejects = 0  # 予備判定で全解像度照合を省いた回数 (テンプレート単位。実測用)

    @property
    def template_count(self) -> int:
        """読み込んだテンプレート数。0 なら detect は常に「テロップなし」を返す (配布版でテンプレート不在の状態)。"""
        return len(self._templates)

    @classmethod
    def load_default(
        cls,
        template_dir: Path = DEFAULT_TEMPLATE_DIR,
        threshold: float = DEFAULT_NCC_THRESHOLD,
        fast: bool | None = None,
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
        return cls(templates=templates, threshold=threshold, fast=fast)

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
        prepared = PreparedImage(roi_gray)
        coarse_roi = _shrink(roi_gray) if self.fast else None

        best_name: str | None = None
        best_score: float = -1.0
        best_bbox: tuple[int, int, int, int] | None = None
        for name, tmpl in self._templates.items():
            tH, tW = tmpl.shape[:2]
            if roi_gray.shape[0] < tH or roi_gray.shape[1] < tW:
                continue
            if coarse_roi is not None and self._coarse_below_floor(name, coarse_roi):
                continue
            max_val, max_loc = self._prepared[name].peak(roi_gray, self._threshold, prepared)
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

    def _coarse_below_floor(self, name: str, coarse_roi: np.ndarray) -> bool:
        """縮小 ROI での最大 NCC が床未満なら True (= このテンプレートは閾値未満と扱える)。"""
        coarse = self._coarse[name]
        if coarse_roi.shape[0] < coarse.shape[0] or coarse_roi.shape[1] < coarse.shape[1]:
            return False  # 縮小後にテンプレートが入らない場合は判断せず従来経路へ
        peak = float(cv2.minMaxLoc(cv2.matchTemplate(coarse_roi, coarse, cv2.TM_CCOEFF_NORMED))[1])
        below = peak < PREFILTER_FLOOR
        self.coarse_rejects += below
        return below

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
