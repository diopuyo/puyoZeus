"""
プレイヤースコアが「00000000」にリセットされている状態を検出する。

用途:
    試合切り替わり判定。試合終了 → 次試合開始の境界で両サイドのスコアが
    00000000 に戻る。「どちらかのスコアがゼロ→非ゼロ」or「両方ゼロ」を
    信頼性の高い試合境界シグナルとして使う。

テンプレート:
    models/ui_templates/score_zero/zero_1P.png  (1P 側スコア領域の 00000000)
    models/ui_templates/score_zero/zero_2P.png  (2P 側)

判定:
    テンプレートに対する NCC が閾値以上ならゼロ表示とみなす。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # 型注釈のみ (実行時 import しない = 既存の import 閉包を変えない)
    from src.score_ocr import ScoreOcr

# スコア領域（1920×1080 前提）
SCORE_1P_REGION: tuple[int, int, int, int] = (890, 955, 200, 680)  # y1, y2, x1, x2
SCORE_2P_REGION: tuple[int, int, int, int] = (890, 955, 1260, 1740)

# テンプレート
DEFAULT_TEMPLATE_DIR: Path = Path("models/ui_templates/score_zero")
TEMPLATE_1P: Path = DEFAULT_TEMPLATE_DIR / "zero_1P.png"
TEMPLATE_2P: Path = DEFAULT_TEMPLATE_DIR / "zero_2P.png"

# ゼロ判定閾値（NCC）
ZERO_NCC_THRESHOLD: float = 0.85


@dataclass(frozen=True)
class ScoreZeroResult:
    """両サイドのゼロ判定結果。"""
    is_1p_zero: bool
    is_2p_zero: bool
    score_1p: float
    score_2p: float

    @property
    def both_zero(self) -> bool:
        return self.is_1p_zero and self.is_2p_zero


class ScoreZeroDetector:
    """スコアが 00000000 であるかを両サイド同時判定する。"""

    def __init__(
        self,
        template_1p: np.ndarray,
        template_2p: np.ndarray,
        threshold: float = ZERO_NCC_THRESHOLD,
    ) -> None:
        # グレースケール化
        self._tpl_1p = cv2.cvtColor(template_1p, cv2.COLOR_BGR2GRAY) if template_1p.ndim == 3 else template_1p
        self._tpl_2p = cv2.cvtColor(template_2p, cv2.COLOR_BGR2GRAY) if template_2p.ndim == 3 else template_2p
        self._threshold = threshold

    @classmethod
    def load_default(
        cls,
        template_1p: Path = TEMPLATE_1P,
        template_2p: Path = TEMPLATE_2P,
        threshold: float = ZERO_NCC_THRESHOLD,
    ) -> "ScoreZeroDetector":
        tpl1 = cv2.imread(str(template_1p))
        tpl2 = cv2.imread(str(template_2p))
        if tpl1 is None or tpl2 is None:
            raise FileNotFoundError(
                f"score_zero テンプレートが見つからない: {template_1p}, {template_2p}"
            )
        return cls(tpl1, tpl2, threshold=threshold)

    def _match(self, frame: np.ndarray, region: tuple[int, int, int, int], tpl: np.ndarray) -> float:
        y1, y2, x1, x2 = region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # テンプレと ROI の shape が同じなら matchTemplate の出力は 1 要素
        th, tw = tpl.shape[:2]
        rh, rw = gray.shape[:2]
        if rh < th or rw < tw:
            return 0.0
        result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    def detect(self, frame: np.ndarray) -> ScoreZeroResult:
        if frame is None or frame.ndim != 3:
            return ScoreZeroResult(False, False, 0.0, 0.0)
        h, w = frame.shape[:2]
        if (h, w) != (1080, 1920):
            return ScoreZeroResult(False, False, 0.0, 0.0)
        s1 = self._match(frame, SCORE_1P_REGION, self._tpl_1p)
        s2 = self._match(frame, SCORE_2P_REGION, self._tpl_2p)
        return ScoreZeroResult(
            is_1p_zero=s1 >= self._threshold,
            is_2p_zero=s2 >= self._threshold,
            score_1p=s1,
            score_2p=s2,
        )


# ============================
# ScoreOcr 由来のゼロ判定 (配布版でスコアゼロ画像を同梱しないための代替、既定OFF)
# ============================

# 環境変数 PUYO_SCORE_ZERO_FROM_OCR=1 で有効 (spawn 子 process へ配布ランチャが伝える。他の PUYO_* フラグと同方式)
FROM_OCR_ENV: str = "PUYO_SCORE_ZERO_FROM_OCR"
ENV_ON_VALUE: str = "1"
# ScoreOcr.read_side が返す score の「全桁 0」値 (8 桁全て読めて平均信頼度規則を満たした時のみ int で返る)
SCORE_ZERO_VALUE: int = 0


def score_zero_from_ocr_enabled(explicit: bool | None = None) -> bool:
    """explicit=None は環境変数、明示指定はそれを優先 (既定 OFF = 従来のテンプレート方式と bit-identical)。"""
    import os
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(FROM_OCR_ENV) == ENV_ON_VALUE


class OcrScoreZeroDetector:
    """ScoreOcr で読んだ両サイドのスコアが 8 桁全て 0 かどうかで判定する (画像テンプレート不要)。

    ScoreZeroDetector と同じ detect(frame) -> ScoreZeroResult の口を持つ。ScoreOcr の既存の信頼度規則
    (各桁 NCC 下限・1位2位の差・平均信頼度下限) を通った読取りだけがゼロと認められる。
    読めない (None) フレームは「ゼロではない」側に倒す (従来のテンプレート方式が不一致時に False なのと同じ向き)。
    ScoreZeroResult.score_1p/2p には ScoreOcr の信頼度 (0..1) を入れる。
    """

    def __init__(self, ocr: "ScoreOcr") -> None:
        self._ocr = ocr

    def detect(self, frame: np.ndarray) -> ScoreZeroResult:
        if frame is None or frame.ndim != 3:
            return ScoreZeroResult(False, False, 0.0, 0.0)
        value_1p, conf_1p = self._ocr.read_side(frame, "1P")
        value_2p, conf_2p = self._ocr.read_side(frame, "2P")
        return ScoreZeroResult(
            is_1p_zero=value_1p == SCORE_ZERO_VALUE,
            is_2p_zero=value_2p == SCORE_ZERO_VALUE,
            score_1p=float(conf_1p),
            score_2p=float(conf_2p),
        )
