"""配信 C 群の独立した試作。既定 OFF、状態はパイプライン単位。"""
from __future__ import annotations

import os
from typing import Any, Callable
from src.board import Board
from src.production_config import LIVE_C_ITEMS_DEFAULTS

BOUNDARY_PERIOD_SEC = 0.5


def enabled(name: str) -> bool:
    """配布の配送済み設定を読み、未指定時は単一情報源へ戻す。"""
    return os.environ.get('PUYO_' + name.upper(), str(int(LIVE_C_ITEMS_DEFAULTS[name]))) == '1'


def boundary_detect(pipe: Any, detector: Any, frame: Any, stamp: float, key: str) -> Any:
    """同期検出だけを間引く。初回・巻戻し・検出器変更は即読み直す。"""
    if not enabled('live_boundary_half_second'):
        return detector.detect(frame)
    cache = getattr(pipe, '_live_boundary_cache', None)
    if cache is None:
        cache = pipe._live_boundary_cache = {}
    previous = cache.get(key)
    if previous is not None:
        when, owner, value, error = previous
        if owner is detector and when <= stamp < when + BOUNDARY_PERIOD_SEC:
            if error is not None:
                raise error
            return value
    try:
        value = detector.detect(frame)
    except Exception as error:
        cache[key] = (stamp, detector, None, error)
        raise
    cache[key] = (stamp, detector, value, None)
    return value


class DeferredHsv:
    """現在のコマ内で最初に必要になった時だけ HSV 盤を読む。"""

    def __init__(self, read: Callable[[], Board]) -> None:
        self.read = read
        self.done = False
        self.value: Board | None = None

    def resolve(self) -> Board | None:
        """失敗時 None は従来の _step_side の例外処理と同じ。"""
        if not self.done:
            self.done = True
            try:
                self.value = self.read()
            except Exception:
                self.value = None
        return self.value


def resolve_hsv(board: Board | DeferredHsv | None) -> Board | None:
    """OFF の通常盤はそのまま通す。"""
    return board.resolve() if isinstance(board, DeferredHsv) else board
