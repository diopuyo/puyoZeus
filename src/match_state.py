"""
フレームが「対戦中（in-match）」か「非対戦（解説画面・VS画面・メニュー等）」かを判定する。

原理:
    対戦中は盤面領域の背景が暗い（puyo フィールドの暗色 BG）。
    非対戦時はブラウザ画面 / VS バナー / キャラクタ絵などで明るい。
    盤面領域内 puyo が到達しにくい上部セルをサンプリングし、HSV.V が低ければ in-match。

使い方:
    detector = MatchStateDetector.load_default()
    state = detector.detect(bgr_frame)       # MatchState.IN_MATCH or NOT_IN_MATCH
    metric = detector.bg_value(bgr_frame)    # 生の明度値（デバッグ用）

閾値:
    bg_value (HSV V 平均) < IN_MATCH_V_MAX → in-match
    experimentally: 試合 69-131, 非試合 178-231 (十分な分離)
    ただし試合中でも盤面が混雑して上部にぷよが積み上がると V=130-160 に
    達することがあり、150 では誤判定が発生 (m27 で実証)。170 まで上げて
    not-in-match (>=178) との安全マージンを確保する。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from src.calibration import CalibratedConfig
from src.image_reader import BoardRegion

# 上部サンプル領域（puyo がほとんど到達しない行）
SAMPLE_ROW_MIN: int = 2
SAMPLE_ROW_MAX: int = 5
# 列端は盤面フレームが入るので中央寄りだけ使う
SAMPLE_COL_MIN: int = 1
SAMPLE_COL_MAX: int = 4
# セル中央の N×N ピクセルをサンプル
SAMPLE_HALF: int = 8

# 試合/非試合の境界（HSV V 平均）
# m27 の実測: 試合中混雑時に V=151-156 まで上昇するケースあり、
# 非試合は V=178-231。中央 (170) を採用してマージン確保。
IN_MATCH_V_MAX: float = 170.0


class MatchState(str, Enum):
    IN_MATCH = "in_match"
    NOT_IN_MATCH = "not_in_match"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MatchDetectResult:
    state: MatchState
    bg_value: float      # 背景明度（閾値判定に使用）
    bg_saturation: float # 参考
    samples: int         # 採用したサンプル数


class MatchStateDetector:
    """盤面領域の背景明度で in-match を判定する。"""

    def __init__(
        self,
        p1_region: BoardRegion,
        p2_region: BoardRegion,
        v_threshold: float = IN_MATCH_V_MAX,
    ) -> None:
        self._p1 = p1_region
        self._p2 = p2_region
        self._v_threshold = v_threshold

    @classmethod
    def load_default(
        cls,
        calib_path: Path = Path("models/calibration_video01.json"),
        v_threshold: float = IN_MATCH_V_MAX,
    ) -> "MatchStateDetector":
        config = CalibratedConfig.load(calib_path)
        return cls(config.p1_region, config.p2_region, v_threshold=v_threshold)

    def _sample_bg(self, frame: np.ndarray) -> tuple[float, float, int]:
        if frame is None or frame.ndim != 3:
            return 0.0, 0.0, 0
        h, w = frame.shape[:2]
        sats: list[float] = []
        vals: list[float] = []
        for region in (self._p1, self._p2):
            for row in range(SAMPLE_ROW_MIN, SAMPLE_ROW_MAX + 1):
                for col in range(SAMPLE_COL_MIN, SAMPLE_COL_MAX + 1):
                    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    ys = max(0, cy - SAMPLE_HALF)
                    ye = min(h, cy + SAMPLE_HALF)
                    xs = max(0, cx - SAMPLE_HALF)
                    xe = min(w, cx + SAMPLE_HALF)
                    patch = frame[ys:ye, xs:xe]
                    if patch.size == 0:
                        continue
                    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    sats.append(float(hsv[:, :, 1].mean()))
                    vals.append(float(hsv[:, :, 2].mean()))
        if not vals:
            return 0.0, 0.0, 0
        return float(np.mean(sats)), float(np.mean(vals)), len(vals)

    def detect(self, frame: np.ndarray) -> MatchDetectResult:
        sat, val, n = self._sample_bg(frame)
        if n == 0:
            return MatchDetectResult(
                state=MatchState.UNKNOWN, bg_value=val, bg_saturation=sat, samples=0,
            )
        state = MatchState.IN_MATCH if val < self._v_threshold else MatchState.NOT_IN_MATCH
        return MatchDetectResult(
            state=state, bg_value=val, bg_saturation=sat, samples=n,
        )

    def is_in_match(self, frame: np.ndarray) -> bool:
        return self.detect(frame).state == MatchState.IN_MATCH

    def bg_value(self, frame: np.ndarray) -> float:
        return self._sample_bg(frame)[1]
