"""左右の終了告知の因果的な組照合。解除は呼び出し側の得点観測で行う。"""
from __future__ import annotations

from dataclasses import dataclass, field
import os

import cv2
import numpy as np

from src.match_end_detector import MatchEndDetectionResult, MatchEndDetector

# 2026-10-24: 録画1本の真陽性最小0.555・対照最大0.317の間。別標本で未検証。
PAIR_NCC_THRESHOLD = 0.50
PAIR_WINDOW_SEC = 0.5
YATTA_TEXT_ROWS = slice(210, 365)
YATTA_TEXT_COLS = slice(70, 500)
PAIR_ROIS = {
    "yatta_1p": (100, 100, 800, 600),
    "yatta_2p": (1020, 100, 800, 600),
    "batan_1p": (70, 200, 700, 500),
    "batan_2p": (1150, 200, 700, 500),
}


@dataclass(frozen=True)
class PairDetectionResult(MatchEndDetectionResult):
    """領域ごとの最新NCC・観測時刻と、過去窓を含む組判定。"""
    ncc: dict[str, float]
    winner: str | None
    ncc_observed_t_sec: dict[str, float | None] = field(default_factory=dict)
    scanned_roi: str | None = None


class MatchEndPairDetector(MatchEndDetector):
    """組検出後は明示的な解除まで保持する。"""

    def __init__(self, templates: dict[str, np.ndarray]) -> None:
        from src.production_config import MATCH_END_YATTA_TEXT_TEMPLATE_DEFAULT
        if os.environ.get('PUYO_MATCH_END_YATTA_TEXT_TEMPLATE',
                          str(int(MATCH_END_YATTA_TEXT_TEMPLATE_DEFAULT))) == '1':
            templates = {name: value[YATTA_TEXT_ROWS, YATTA_TEXT_COLS].copy()
                         if 'yatta' in name else value for name, value in templates.items()}
        super().__init__(templates, threshold=PAIR_NCC_THRESHOLD)
        from src.image_free_ui import enabled
        self._image_free = enabled()   # 画像なし方式: 切り抜き画像の代わりに学んだ森で領域ごとの採否を出す (11/14)
        self._observations: dict[str, tuple[float, float]] = {}
        self._next_roi = 0
        self.lock_started_t: float | None = None

    def _scores(self, frame: np.ndarray) -> dict[str, float]:
        """同じテンプレートを左右それぞれで照合する（文字画像は反転しない）。"""
        return {key: self._score_roi(frame, key) for key in PAIR_ROIS}

    def _score_roi(self, frame: np.ndarray, key: str) -> float:
        """指定した1領域だけを、同じ解像度・NCC式で照合する。"""
        score = -1.0
        if frame is None or frame.size == 0:
            return score
        x, y, width, height = PAIR_ROIS[key]
        crop = frame[y:y + height, x:x + width]
        return self._score_crop(crop, key)

    def _score_crop(self, crop: np.ndarray, key: str) -> float:
        """複製済みの領域も同じNCC式で評価する。"""
        score = -1.0
        if crop.size == 0:
            return score
        if getattr(self, '_image_free', False):
            from src.image_free_ui import learned_score
            return learned_score('pair_' + key.split('_')[0], crop)   # 左右で同じ森 (pair_yatta / pair_batan)
        roi = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for name, template in self._templates.items():
            if key.split("_")[0] not in name:
                continue
            if any(a < b for a, b in zip(roi.shape, template.shape)):
                continue
            ncc = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            score = max(score, float(cv2.minMaxLoc(ncc)[1]))
        return score

    def _result(self, scores: dict[str, float], hits: dict[str, float],
                observed: dict[str, float | None] | None = None,
                scanned_roi: str | None = None) -> PairDetectionResult:
        """両方向が成立した曖昧なコマは勝者を決めない。"""
        pairs = {"1P": min(hits.get("yatta_1p", -1), hits.get("batan_2p", -1)),
                 "2P": min(hits.get("yatta_2p", -1), hits.get("batan_1p", -1))}
        winners = [side for side, score in pairs.items() if score >= PAIR_NCC_THRESHOLD]
        winner = winners[0] if len(winners) == 1 else None
        return PairDetectionResult(winner is not None, None, max(pairs.values()),
                                   scores, winner, observed or {}, scanned_roi)

    def detect(self, frame_bgr: np.ndarray) -> PairDetectionResult:
        """状態を変更せず、同じコマの組のみを返す。"""
        scores = self._scores(frame_bgr)
        return self._result(scores, scores)

    def update(self, frame_bgr: np.ndarray, t_sec: float) -> bool:
        """毎コマ1領域を巡回し、最新観測のうち過去0.5秒以内で組を作る。"""
        key = tuple(PAIR_ROIS)[self._next_roi]
        self._next_roi = (self._next_roi + 1) % len(PAIR_ROIS)
        self._observations = {k: hit for k, hit in self._observations.items()
                              if hit[0] <= t_sec}
        self._observations[key] = (t_sec, self._score_roi(frame_bgr, key))
        scores = {k: self._observations.get(k, (None, -1.0))[1] for k in PAIR_ROIS}
        observed = {k: self._observations.get(k, (None, -1.0))[0] for k in PAIR_ROIS}
        hits = {k: hit[1] for k, hit in self._observations.items()
                if 0 <= t_sec - hit[0] <= PAIR_WINDOW_SEC}
        # 初回一巡前には、未観測の反対方向を陰性とみなさない。
        if len(self._observations) < len(PAIR_ROIS):
            hits = {}
        result = self._result(scores, hits, observed, key)
        self.last_detection, self.last_detection_t = result, t_sec
        if result.detected:
            if self.lock_started_t is None:
                self.lock_started_t = t_sec
            self._last_detected_t = t_sec
        return self.is_locked(t_sec)

    def is_locked(self, t_sec: float) -> bool:
        return self.lock_started_t is not None

    def announcement_recent(self, t_sec: float) -> bool:
        """得点移動の補助解除には、従来どおり告知後5秒の抑止を残す。"""
        return super().is_locked(t_sec)

    def release(self) -> None:
        """診断の直近観測を残し、保持と組の履歴を解除する。"""
        self._observations.clear()
        self._next_roi = 0
        self.lock_started_t = None
        self._last_detected_t = None

    def reset(self) -> None:
        super().reset()
        self.release()
