"""保留中の終了照合だけを供給し、盤面認識の入力ゲートは維持する。"""
from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import numpy as np

from src.production_config import LIVE_MATCH_END_FEED_ON_HOLD_DEFAULT


def enabled() -> bool:
    return os.environ.get('PUYO_LIVE_MATCH_END_FEED_ON_HOLD',
                          str(int(LIVE_MATCH_END_FEED_ON_HOLD_DEFAULT))) == '1'


def eligible(image: np.ndarray, verifier: Any, feedback_present: bool | None = None) -> bool:
    """同じ画像の両側を確認。得点か枠は各側に必要で、位置異常は拒否する。"""
    from .distribution_input import has_feedback
    from .live_device import (PuyoScreenVerifier, GRID_MARGIN, GRID_POSITION_TOLERANCE,
                              frame_border_matches)
    base = getattr(verifier, 'base', verifier)
    if not isinstance(base, PuyoScreenVerifier):
        return False
    if feedback_present if feedback_present is not None else has_feedback(image):
        return False
    for side, region in zip(('1P', '2P'), base.regions):
        crop = image[region.y-GRID_MARGIN:region.y+region.height+GRID_MARGIN,
                     region.x-GRID_MARGIN:region.x+region.width+GRID_MARGIN]
        grid = base.grid.detect(crop)
        if grid is not None:
            expected = np.array([GRID_MARGIN, GRID_MARGIN,
                                 region.width+GRID_MARGIN, region.height+GRID_MARGIN])
            if np.max(np.abs(expected-np.array([*grid.top_left, *grid.bottom_right]))) > GRID_POSITION_TOLERANCE:
                return False
        border = grid is not None or frame_border_matches(
            crop, region.width, region.height, exempt_top=base.relaxed)
        if not border and not base.score_ok(image, side):
            return False
    # 両側の固定位置で枠または8桁得点（緩和時は式）を要求する。
    return True


def feed(session: Any, image: np.ndarray | None, stamp: float,
         captured_at: float | None = None) -> None:
    """入力所有者からだけ呼ぶ。既存の組保持と0/0解除をそのまま使う。"""
    session._match_end_hold_eligible = image is not None
    if image is None:
        return
    pipe = session.pipe
    detector = pipe._match_end_detector
    locked = detector.update(image, stamp)
    if locked:
        pipe._update_post_match_lockdown_latch(image, stamp, True, False, None, None, False)
        pipe._last_match_end_locked = True
    # 認識通知のない保留中も、採用した終了証拠を既存の原票へ保存する。
    sink = getattr(pipe, '_live_terminal_observations', None)
    if sink is not None:
        from .live_observation_diagnostics import recognition_diagnostics
        from .live_source import RECOGNITION_HZ
        sink.append(dict(frame=round(stamp*RECOGNITION_HZ), t_sec=stamp,
                         captured_at=captured_at, recognized_at=perf_counter(),
                         ready=False, held=True, **recognition_diagnostics(pipe)))
