"""
キャリブレーションモジュール

実対戦フレーム画像から ImageReader 用の盤面座標 (BoardRegion) と
色分類閾値 (HsvRange) を抽出するユーティリティ。

使用フロー:
    1. 対戦動画から代表フレームを 1 枚用意する
    2. そのフレーム上の 1P/2P 盤面の左上・右下座標を annotation.json に記述
    3. 各色が写っているセルの (row, col) 位置も annotation.json に記述
    4. CalibrationHelper.calibrate_from_reference() で CalibratedConfig を生成
    5. save() で config JSON に永続化 → ImageReader 初期化時に load() で復元

annotation.json 例:
    {
      "p1_corners": {"top_left": [195, 57], "bottom_right": [405, 551]},
      "p2_corners": {"top_left": [1515, 57], "bottom_right": [1725, 551]},
      "color_samples": {
        "1": [[12, 0], [11, 0]],   # 赤の位置 (row, col)
        "2": [[12, 1]],
        ...
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.board import VALID_COLORS
from src.image_reader import (
    CELL_SAMPLE_RATIO,
    BoardRegion,
    ColorClassifier,
    HsvRange,
    ImageReader,
)

# ============================
# 定数定義
# ============================

# HsvRange の padding (サンプル中央値 ± この幅を閾値とする)
DEFAULT_H_PADDING: int = 8
DEFAULT_S_PADDING: int = 40
DEFAULT_V_PADDING: int = 40

# H の循環 (0-180) を考慮した赤の折り返し閾値
H_MAX_VALUE: int = 180
H_WRAP_THRESHOLD: int = 15

# annotation.json のキー名
KEY_P1_CORNERS: str = "p1_corners"
KEY_P2_CORNERS: str = "p2_corners"
KEY_COLOR_SAMPLES: str = "color_samples"
KEY_TOP_LEFT: str = "top_left"
KEY_BOTTOM_RIGHT: str = "bottom_right"

# 自動検出: 盤面背景の HSV 閾値 (空セルの暗い領域)
AUTO_DETECT_BLOCK: int = 10            # 解析ブロックサイズ (px)
AUTO_DETECT_V_STD_MAX: int = 25        # 暗くて均一なブロック判定用
AUTO_DETECT_V_MEAN_MAX: int = 90
AUTO_DETECT_S_MEAN_MAX: int = 180
AUTO_DETECT_MIN_BOARD_W: int = 200     # 検出する矩形の最小幅
AUTO_DETECT_MIN_BOARD_H: int = 400
AUTO_DETECT_ASPECT_MIN: float = 1.5    # 縦横比下限
AUTO_DETECT_ASPECT_MAX: float = 2.3    # 縦横比上限
AUTO_DETECT_CLOSE_KERNEL: int = 30
AUTO_DETECT_OPEN_KERNEL: int = 10
AUTO_DETECT_CENTER_GAP: int = 100      # 中央 ±gap は除外


# ============================
# データクラス
# ============================


@dataclass
class CalibrationAnnotation:
    """
    人手で記述するキャリブレーション入力。

    Attributes:
        p1_top_left: 1P 盤面左上座標 (x, y)。
        p1_bottom_right: 1P 盤面右下座標。
        p2_top_left: 2P 盤面左上座標。
        p2_bottom_right: 2P 盤面右下座標。
        color_samples: 色コード→サンプル位置リスト [(row, col), ...]。
    """
    p1_top_left: tuple[int, int]
    p1_bottom_right: tuple[int, int]
    p2_top_left: tuple[int, int]
    p2_bottom_right: tuple[int, int]
    color_samples: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "CalibrationAnnotation":
        """JSON ファイルから annotation を読み込む。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        p1 = data[KEY_P1_CORNERS]
        p2 = data[KEY_P2_CORNERS]
        # JSON の key は文字列なので int 化
        samples_raw = data.get(KEY_COLOR_SAMPLES, {})
        samples: dict[int, list[tuple[int, int]]] = {}
        for k, v in samples_raw.items():
            color_code = int(k)
            samples[color_code] = [(int(r), int(c)) for r, c in v]
        return cls(
            p1_top_left=tuple(p1[KEY_TOP_LEFT]),
            p1_bottom_right=tuple(p1[KEY_BOTTOM_RIGHT]),
            p2_top_left=tuple(p2[KEY_TOP_LEFT]),
            p2_bottom_right=tuple(p2[KEY_BOTTOM_RIGHT]),
            color_samples=samples,
        )


@dataclass
class CalibratedConfig:
    """
    キャリブレーション結果。ImageReader にそのまま渡せる形式。

    Attributes:
        p1_region: 1P 盤面の BoardRegion。
        p2_region: 2P 盤面の BoardRegion。
        color_ranges: 色コード→HsvRange リストの辞書。
    """
    p1_region: BoardRegion
    p2_region: BoardRegion
    color_ranges: dict[int, list[HsvRange]] = field(default_factory=dict)

    # ============================
    # シリアライズ
    # ============================

    def to_dict(self) -> dict[str, Any]:
        """JSON 保存可能な辞書に変換する。"""
        return {
            "p1_region": _region_to_dict(self.p1_region),
            "p2_region": _region_to_dict(self.p2_region),
            "color_ranges": {
                str(code): [_hsv_to_dict(r) for r in ranges]
                for code, ranges in self.color_ranges.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibratedConfig":
        """辞書から CalibratedConfig を復元する。"""
        color_ranges: dict[int, list[HsvRange]] = {}
        for k, v in data.get("color_ranges", {}).items():
            color_ranges[int(k)] = [_hsv_from_dict(d) for d in v]
        return cls(
            p1_region=_region_from_dict(data["p1_region"]),
            p2_region=_region_from_dict(data["p2_region"]),
            color_ranges=color_ranges,
        )

    def save(self, path: Path) -> None:
        """JSON ファイルに永続化する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CalibratedConfig":
        """JSON ファイルから読み込む。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ============================
    # ImageReader 生成
    # ============================

    def build_reader(self) -> ImageReader:
        """この config で ImageReader を生成する。"""
        classifier = (
            ColorClassifier(color_ranges=self.color_ranges)
            if self.color_ranges else ColorClassifier()
        )
        return ImageReader(
            classifier=classifier,
            p1_region=self.p1_region,
            p2_region=self.p2_region,
        )


# ============================
# 内部ヘルパー (dict 変換)
# ============================


def _region_to_dict(r: BoardRegion) -> dict[str, int]:
    return {"x": r.x, "y": r.y, "width": r.width, "height": r.height}


def _region_from_dict(d: dict[str, int]) -> BoardRegion:
    return BoardRegion(x=d["x"], y=d["y"], width=d["width"], height=d["height"])


def _hsv_to_dict(r: HsvRange) -> dict[str, int]:
    return {
        "h_min": r.h_min, "h_max": r.h_max,
        "s_min": r.s_min, "s_max": r.s_max,
        "v_min": r.v_min, "v_max": r.v_max,
    }


def _hsv_from_dict(d: dict[str, int]) -> HsvRange:
    return HsvRange(
        h_min=d["h_min"], h_max=d["h_max"],
        s_min=d.get("s_min", 80), s_max=d.get("s_max", 255),
        v_min=d.get("v_min", 80), v_max=d.get("v_max", 255),
    )


# ============================
# CalibrationHelper
# ============================


class CalibrationHelper:
    """
    フレーム画像と annotation から CalibratedConfig を生成する。

    Usage:
        frame = cv2.imread("reference.png")
        ann = CalibrationAnnotation.from_json(Path("annotation.json"))
        helper = CalibrationHelper()
        config = helper.calibrate_from_reference(frame, ann)
        config.save(Path("models/calibration.json"))
    """

    def __init__(
        self,
        h_padding: int = DEFAULT_H_PADDING,
        s_padding: int = DEFAULT_S_PADDING,
        v_padding: int = DEFAULT_V_PADDING,
    ) -> None:
        """
        Args:
            h_padding: HsvRange の Hue 許容幅。
            s_padding: HsvRange の Saturation 許容幅。
            v_padding: HsvRange の Value 許容幅。
        """
        self._h_pad = h_padding
        self._s_pad = s_padding
        self._v_pad = v_padding

    # ============================
    # 公開メソッド
    # ============================

    def region_from_corners(
        self,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
    ) -> BoardRegion:
        """
        左上・右下の 2 点から BoardRegion を生成する。

        Args:
            top_left: 盤面左上座標 (x, y)。
            bottom_right: 盤面右下座標 (x, y)。

        Returns:
            BoardRegion: 計算された盤面領域。

        Raises:
            ValueError: 座標が逆転している場合。
        """
        x1, y1 = top_left
        x2, y2 = bottom_right
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"座標が逆転しています: tl={top_left} br={bottom_right}"
            )
        return BoardRegion(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def sample_cell_hsv(
        self,
        frame: np.ndarray,
        region: BoardRegion,
        row: int,
        col: int,
    ) -> tuple[int, int, int]:
        """
        フレーム上の指定セル中央領域の HSV 中央値を返す。

        Args:
            frame: BGR フレーム画像。
            region: 盤面領域。
            row: セル行。
            col: セル列。

        Returns:
            tuple[int, int, int]: (H, S, V) の中央値。
        """
        x1, y1, x2, y2 = region.cell_sample_rect(row, col)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        patch = frame[y1:y2, x1:x2]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return (
            int(np.median(hsv_patch[:, :, 0])),
            int(np.median(hsv_patch[:, :, 1])),
            int(np.median(hsv_patch[:, :, 2])),
        )

    def hsv_range_from_samples(
        self,
        frame: np.ndarray,
        region: BoardRegion,
        positions: list[tuple[int, int]],
    ) -> list[HsvRange]:
        """
        指定セル群の HSV サンプルから HsvRange (複数可) を生成する。

        Hue が 0/180 境界を跨ぐ場合 (赤) は 2 本の Range に分割する。

        Args:
            frame: BGR フレーム。
            region: 盤面領域。
            positions: [(row, col), ...] のサンプル位置。

        Returns:
            list[HsvRange]: 生成された HSV 範囲リスト。

        Raises:
            ValueError: positions が空の場合。
        """
        if not positions:
            raise ValueError("サンプル位置が空です")

        samples = [
            self.sample_cell_hsv(frame, region, r, c) for r, c in positions
        ]
        hues = [s[0] for s in samples]
        sats = [s[1] for s in samples]
        vals = [s[2] for s in samples]

        s_min = max(0, min(sats) - self._s_pad)
        s_max = min(255, max(sats) + self._s_pad)
        v_min = max(0, min(vals) - self._v_pad)
        v_max = min(255, max(vals) + self._v_pad)

        return self._build_hue_ranges(hues, s_min, s_max, v_min, v_max)

    def detect_board_regions(
        self,
        frame: np.ndarray,
    ) -> tuple[BoardRegion, BoardRegion] | None:
        """
        フレーム画像から 1P/2P 盤面矩形を自動検出する。

        盤面の空セル背景 (暗くて低彩度かつ均一) を全画面で検索し、
        左右半分でそれぞれ最大の縦長矩形を盤面とみなす。

        Args:
            frame: BGR フレーム画像。

        Returns:
            tuple[BoardRegion, BoardRegion] | None:
                (1P盤面, 2P盤面)。両方検出できなければ None。
        """
        h, w = frame.shape[:2]
        mask = self._board_background_mask(frame)
        mid = w // 2
        left = self._largest_in_range(
            mask, 0, mid - AUTO_DETECT_CENTER_GAP,
        )
        right = self._largest_in_range(
            mask, mid + AUTO_DETECT_CENTER_GAP, w,
        )
        if left is None or right is None:
            return None
        return left, right

    def calibrate_from_auto_detection(
        self,
        frame: np.ndarray,
    ) -> CalibratedConfig | None:
        """
        フレームから自動検出した座標で CalibratedConfig を作る (色閾値はデフォルト)。

        Returns:
            CalibratedConfig | None: 盤面が検出できなければ None。
        """
        regions = self.detect_board_regions(frame)
        if regions is None:
            return None
        return CalibratedConfig(
            p1_region=regions[0],
            p2_region=regions[1],
            color_ranges={},
        )

    def calibrate_from_reference(
        self,
        frame: np.ndarray,
        annotation: CalibrationAnnotation,
    ) -> CalibratedConfig:
        """
        参照フレームと annotation から完全な CalibratedConfig を生成する。

        Args:
            frame: 参照 BGR フレーム。
            annotation: 人手で記述したアノテーション。

        Returns:
            CalibratedConfig: キャリブレーション結果。
        """
        p1_region = self.region_from_corners(
            annotation.p1_top_left, annotation.p1_bottom_right,
        )
        p2_region = self.region_from_corners(
            annotation.p2_top_left, annotation.p2_bottom_right,
        )

        color_ranges: dict[int, list[HsvRange]] = {}
        for color_code, positions in annotation.color_samples.items():
            if color_code not in VALID_COLORS:
                raise ValueError(f"不正な色コード: {color_code}")
            if not positions:
                continue
            color_ranges[color_code] = self.hsv_range_from_samples(
                frame, p1_region, positions,
            )

        return CalibratedConfig(
            p1_region=p1_region,
            p2_region=p2_region,
            color_ranges=color_ranges,
        )

    # ============================
    # 内部メソッド
    # ============================

    @staticmethod
    def _board_background_mask(frame: np.ndarray) -> np.ndarray:
        """盤面背景 (暗めで均一なブロック) のマスクを返す。"""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rows = h // AUTO_DETECT_BLOCK
        cols = w // AUTO_DETECT_BLOCK
        block_mask = np.zeros((rows, cols), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                y1 = r * AUTO_DETECT_BLOCK
                x1 = c * AUTO_DETECT_BLOCK
                patch = hsv[y1:y1+AUTO_DETECT_BLOCK, x1:x1+AUTO_DETECT_BLOCK]
                if (
                    np.std(patch[:, :, 2]) < AUTO_DETECT_V_STD_MAX
                    and np.mean(patch[:, :, 2]) < AUTO_DETECT_V_MEAN_MAX
                    and np.mean(patch[:, :, 1]) < AUTO_DETECT_S_MEAN_MAX
                ):
                    block_mask[r, c] = 255
        full = cv2.resize(block_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        close_k = np.ones(
            (AUTO_DETECT_CLOSE_KERNEL, AUTO_DETECT_CLOSE_KERNEL), np.uint8,
        )
        open_k = np.ones(
            (AUTO_DETECT_OPEN_KERNEL, AUTO_DETECT_OPEN_KERNEL), np.uint8,
        )
        full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, close_k)
        full = cv2.morphologyEx(full, cv2.MORPH_OPEN, open_k)
        return full

    @staticmethod
    def _largest_in_range(
        mask: np.ndarray, x_start: int, x_end: int,
    ) -> BoardRegion | None:
        """指定 X 範囲の最大縦長矩形を BoardRegion として返す。"""
        sub = mask[:, x_start:x_end]
        contours, _ = cv2.findContours(
            sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        best: BoardRegion | None = None
        best_area = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < AUTO_DETECT_MIN_BOARD_W or h < AUTO_DETECT_MIN_BOARD_H:
                continue
            aspect = h / w
            if not (
                AUTO_DETECT_ASPECT_MIN <= aspect <= AUTO_DETECT_ASPECT_MAX
            ):
                continue
            area = w * h
            if area > best_area:
                best_area = area
                best = BoardRegion(
                    x=x + x_start, y=y, width=w, height=h,
                )
        return best

    def _build_hue_ranges(
        self,
        hues: list[int],
        s_min: int, s_max: int, v_min: int, v_max: int,
    ) -> list[HsvRange]:
        """
        Hue リストから HsvRange を生成する。赤の折り返しに対応する。
        """
        h_min_raw = min(hues) - self._h_pad
        h_max_raw = max(hues) + self._h_pad

        # 0 付近と 180 付近にサンプルが跨る場合 (赤) は 2 分割
        has_low = any(h <= H_WRAP_THRESHOLD for h in hues)
        has_high = any(h >= H_MAX_VALUE - H_WRAP_THRESHOLD for h in hues)
        if has_low and has_high:
            low_hues = [h for h in hues if h <= H_MAX_VALUE // 2]
            high_hues = [h for h in hues if h > H_MAX_VALUE // 2]
            ranges: list[HsvRange] = []
            if low_hues:
                ranges.append(HsvRange(
                    h_min=max(0, min(low_hues) - self._h_pad),
                    h_max=min(H_MAX_VALUE, max(low_hues) + self._h_pad),
                    s_min=s_min, s_max=s_max, v_min=v_min, v_max=v_max,
                ))
            if high_hues:
                ranges.append(HsvRange(
                    h_min=max(0, min(high_hues) - self._h_pad),
                    h_max=min(H_MAX_VALUE, max(high_hues) + self._h_pad),
                    s_min=s_min, s_max=s_max, v_min=v_min, v_max=v_max,
                ))
            return ranges

        return [HsvRange(
            h_min=max(0, h_min_raw),
            h_max=min(H_MAX_VALUE, h_max_raw),
            s_min=s_min, s_max=s_max, v_min=v_min, v_max=v_max,
        )]
