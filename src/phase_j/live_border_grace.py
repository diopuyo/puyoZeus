"""外枠線分だけの失敗を、検証済み入力の短時間に限って猶予する。"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.production_config import LIVE_BORDER_GRACE_DEFAULT
from .live_verify_reasons import observed

GRACE_SEC = 3.0
ANCHOR_HALF_WIDTH = 4
ANCHOR_STRIDE = 4
ANCHOR_MAX_MAE = 15.0


def wrap_verifier(verifier: Any) -> Any:
    """機器ごとの入力所有者が一度だけ生成する。既定OFFは同一オブジェクト。"""
    enabled = os.environ.get('PUYO_LIVE_BORDER_GRACE', str(int(LIVE_BORDER_GRACE_DEFAULT))) == '1'
    return BorderGraceVerifier(verifier) if enabled else verifier


@observed
def verify_screen(verifier: Any, image: Any, captured: float,
                  size: Any, generation: int) -> bool:
    """猶予器だけに未正規化サイズと入力世代を渡す。"""
    if isinstance(verifier, BorderGraceVerifier):
        return verifier.verify(image, captured, size, generation)
    return bool(verifier(image))


def anchors(image: np.ndarray, regions: tuple) -> tuple[np.ndarray, ...]:
    """両盤面の左右・底辺を別々に比較し、盤面内部の変化を除く。"""
    result = []
    half, stride = ANCHOR_HALF_WIDTH, ANCHOR_STRIDE
    for region in regions:
        x, y, w, h = region.x, region.y, region.width, region.height
        for edge in (image[y:y+h:stride, x-half:x+half],
                     image[y:y+h:stride, x+w-half:x+w+half],
                     image[y+h-half:y+h+half, x:x+w:stride]):
            result.append(edge.astype(np.float32))
    return tuple(result)


class BorderGraceVerifier:
    """通常合格だけが期限を更新する。猶予合格で自己延長しない。"""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.context: Any = None
        self.last_pass = float('-inf')
        self.saved: tuple = ()
        self.scores: tuple = ()
        self.feedback = False
        self.diagnostics: dict = {}

    def verify(self, image: np.ndarray, captured: float, size: Any, generation: int) -> bool:
        context = (size, generation)
        if context != self.context:
            self.context, self.last_pass = context, float('-inf')
            self.saved, self.scores = (), ()
        passed = bool(self.base(image))
        self.feedback = bool(getattr(self.base, 'feedback', False))
        self.failure = getattr(self.base, 'failure', None)   # 猶予の外でも保留の種別判定へ届ける
        self.diagnostics = dict(strict=passed, grace=False, reason='strict' if passed else 'ineligible')
        if passed:
            self.last_pass = captured
            self.saved = anchors(image, self.base.regions)
            self.scores = tuple(self.base.score.read_side(image, side)[0] for side in ('1P','2P'))
            return True
        eligible = (not self.feedback and self.saved and self.scores
                    and 0 <= captured-self.last_pass <= GRACE_SEC)
        if not eligible or not self._continuing(image):
            # 得点・位置・自己画面の失敗を後の猶予で覆さない。
            self.last_pass = float('-inf')
            return False
        current = anchors(image, self.base.regions)
        errors = [float(np.mean(np.abs(a-b))) for a,b in zip(current, self.saved)]
        granted = all(error <= ANCHOR_MAX_MAE for error in errors)
        self.diagnostics.update(grace=granted, anchor_mae=errors,
                                reason='border_grace' if granted else 'anchor_changed')
        if granted:
            self.scores = self.pending_scores
        if not granted:
            self.last_pass = float('-inf')
        return granted

    def _continuing(self, image: np.ndarray) -> bool:
        from .live_device import GRID_MARGIN, GRID_POSITION_TOLERANCE
        scores = []
        for side, previous, region in zip(('1P','2P'), self.scores, self.base.regions):
            score = self.base.score.read_side(image, side)[0]
            if score is None or previous is None or score < previous:
                self.diagnostics['reason'] = 'score_discontinuous'
                return False
            scores.append(score)
            crop = image[region.y-GRID_MARGIN:region.y+region.height+GRID_MARGIN,
                         region.x-GRID_MARGIN:region.x+region.width+GRID_MARGIN]
            grid = self.base.grid.detect(crop)
            if grid is not None:
                expected = np.array([GRID_MARGIN, GRID_MARGIN,
                                     region.width+GRID_MARGIN, region.height+GRID_MARGIN])
                if np.max(np.abs(expected-np.array([*grid.top_left,*grid.bottom_right]))) > GRID_POSITION_TOLERANCE:
                    self.diagnostics['reason'] = 'grid_position'
                    return False
        self.pending_scores = tuple(scores)
        return True
