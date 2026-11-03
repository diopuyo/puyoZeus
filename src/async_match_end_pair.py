"""終了の組を別スレッドで照合し、認識側には完了結果だけを渡す。"""
from __future__ import annotations

from dataclasses import replace
from threading import Condition, Thread
from time import monotonic
from typing import Any

import numpy as np

from src.match_end_pair import MatchEndPairDetector, PairDetectionResult, PAIR_ROIS
from src.recog_speed_py import ITEM_PAIR_LAZY, STATS, enabled

# 2026-10-24 user指示: 終了告知は別枠で0.5秒ごとでよい。開始間隔を壁時計で制限。
PAIR_SCAN_INTERVAL_SEC = 0.5


def close_pair_worker(pipe: Any) -> None:
    """認識入力の終了時だけworkerを回収する。OFFでは何もしない。"""
    owner = getattr(pipe, 'pipe', pipe)
    if getattr(owner, '_enable_match_end_pair', False):
        close = getattr(getattr(owner, '_match_end_detector', None), 'close', None)
        if close is not None:
            close()


class LazyCrops(dict):
    """未複製の 4 領域 (画像への view)。worker が取り出すときに複製して照合へ渡す。"""


class AsyncMatchEndPairDetector(MatchEndPairDetector):
    """実行中1件＋待機最新1件。保持と解除は認識スレッドだけが更新する。"""

    def __init__(self, templates: dict[str, np.ndarray]) -> None:
        super().__init__(templates)
        self._condition = Condition()
        self._pending: tuple[int, float, dict[str, np.ndarray]] | None = None
        self._completed: tuple[float, PairDetectionResult] | None = None
        self._generation = 0
        self._closed = False
        self._error: Exception | None = None
        self._thread: Thread | None = None

    def update(self, frame_bgr: np.ndarray, t_sec: float) -> bool:
        """照合を待たず、以前のコマの完了結果を採用して最新画像を渡す。"""
        with self._condition:
            if self._closed:
                raise RuntimeError('終了済みの組照合worker')
            if self._error is not None:
                raise RuntimeError('非同期組照合の失敗') from self._error
            completed, self._completed = self._completed, None
        if completed is not None:
            source_t, result = completed
            if source_t <= t_sec:
                self.last_detection, self.last_detection_t = result, source_t
                if result.detected:
                    if self.lock_started_t is None:
                        self.lock_started_t = source_t
                    self._last_detected_t = source_t
        views = {key: frame_bgr[y:y+h, x:x+w] for key, (x, y, w, h) in PAIR_ROIS.items()}
        if enabled(ITEM_PAIR_LAZY):
            # live_recog_speed_py (既定 OFF): worker が実際に取り出す 0.5 秒に 1 回だけ複製する
            # (毎コマの複製は約 5MB。後続のコマで置き換わり捨てられる分が大半)。
            # 画像は強参照で保持する。呼出元が画像を in-place で書き換えない前提 (src に書き込む箇所は無い)。
            crops = LazyCrops(views)
            STATS['pair_lazy_crops'] += 1
        else:
            crops = {key: view.copy() for key, view in views.items()}
        with self._condition:
            self._pending = (self._generation, t_sec, crops)
            if self._thread is None:
                self._thread = Thread(target=self._run, name='match-end-pair', daemon=True)
                self._thread.start()
            self._condition.notify_all()
        return self.is_locked(t_sec)

    def _evaluate(self, crops: dict[str, np.ndarray], t_sec: float) -> PairDetectionResult:
        """同一コマの4領域で組を作り、照合元の時刻を保存する。"""
        scores = {key: self._score_crop(crop, key) for key, crop in crops.items()}
        return replace(self._result(scores, scores),
                       ncc_observed_t_sec=dict.fromkeys(PAIR_ROIS, t_sec), scanned_roi='all')

    def _run(self) -> None:
        """待機中は最新画像に置換可能。NCCの実行中はロックを保持しない。"""
        next_start = float('-inf')
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                remaining = next_start - monotonic()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                generation, t_sec, crops = self._pending
                self._pending = None
                next_start = monotonic() + PAIR_SCAN_INTERVAL_SEC
            if type(crops) is LazyCrops:
                crops = {key: view.copy() for key, view in crops.items()}
            try:
                result = self._evaluate(crops, t_sec)
            except Exception as error:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
                return
            with self._condition:
                if generation == self._generation and not self._closed:
                    self._completed = (t_sec, result)
                self._condition.notify_all()

    def release(self) -> None:
        """解除前の待機・完了・実行中結果を無効化して再ロックを防ぐ。"""
        with self._condition:
            self._generation += 1
            self._pending = self._completed = None
        super().release()

    def close(self) -> None:
        """未着手を破棄し、実行中の1件だけを回収する。停止時のみ待つ。"""
        with self._condition:
            self._closed = True
            self._pending = self._completed = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()
