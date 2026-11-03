"""予告アイコンの単体分類を、順序と判定を保ったまま並列化する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from weakref import finalize

import numpy as np

from src.ojama_warning import (CELL_COUNT, COUNT_TABLE, FRAME_HEIGHT, FRAME_WIDTH,
    ICON_EMPTY, P1_BOARD_X, P2_BOARD_X, OjamaIcon, OjamaWarningDetector, OjamaWarningResult)
from src.production_config import B3_WARNING_MAX_WORKERS, B3_WARNING_WORKERS


class ParallelOjamaWarningDetector(OjamaWarningDetector):
    """モデルとテンプレートは読取専用。CNNキャッシュは呼出元で排他化する。"""

    def __init__(self, workers: int = B3_WARNING_WORKERS) -> None:
        if type(workers) is not int or not 1 <= workers <= B3_WARNING_MAX_WORKERS:
            raise ValueError(f'予告CNN worker数は1〜{B3_WARNING_MAX_WORKERS}の整数です')
        super().__init__()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='b3-warning')
        # 機器切替による破棄・例外終了でも、参照循環を作らずworkerを回収する。
        self._finalizer = finalize(self, self._pool.shutdown, wait=True)

    def close(self) -> None:
        """入力世代の終了時にworkerを回収する。繰り返し呼んでも安全。"""
        self._finalizer()

    def detect(self, frame: np.ndarray) -> tuple[OjamaWarningResult, OjamaWarningResult]:
        if frame is None or frame.ndim != 3 or frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
            return self._empty_pair()
        cells = [self._extract_cell(frame, x, i)
                 for x in (P1_BOARD_X, P2_BOARD_X) for i in range(CELL_COUNT)]
        kinds = list(self._pool.map(self._classify_cell, cells))
        return (self._result('1P', kinds[:CELL_COUNT]),
                self._result('2P', kinds[CELL_COUNT:]))

    @staticmethod
    def _result(side: str, kinds: list[str]) -> OjamaWarningResult:
        """mapの入力順で再構成し、旧detectと同じ空セル除外・合計にする。"""
        icons = tuple(OjamaIcon(kind, COUNT_TABLE[kind]) for kind in kinds if kind != ICON_EMPTY)
        return OjamaWarningResult(side, icons, sum(icon.count_value for icon in icons))
