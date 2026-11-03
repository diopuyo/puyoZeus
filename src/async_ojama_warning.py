"""予告欄だけを最新一件で非同期処理する。認識側では推論を待たない。"""
from __future__ import annotations

from threading import Condition, Thread
from typing import Any

from src.ojama_warning import OjamaWarningResult


class AsyncWarning:
    """処理中一件＋待機中最新一件。完了結果は次の update から使用する。"""

    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self._condition = Condition()
        self._pending: Any = None
        self._closed = False
        self._error: Exception | None = None
        self._latest = tuple(OjamaWarningResult(side, (), 0) for side in ('1P', '2P'))
        self._source: tuple[int, float] | None = None
        self._thread = Thread(target=self._run, name='b3-warning-async', daemon=True)
        self._thread.start()

    def update(self, image: Any, fi: int, t: float) -> tuple[Any, dict]:
        """完了済み結果を採用してから複製セルを渡す。初回だけゼロ初期値。"""
        cells = self.detector.extract_cells(image, copy=True)
        with self._condition:
            if self._closed:
                raise RuntimeError('終了済みの予告 worker')
            if self._error is not None:
                raise RuntimeError('非同期予告の推論失敗') from self._error
            result, source = self._latest, self._source
            self._pending = (fi, t, cells)
            self._condition.notify()
        metadata = dict(source_frame=None if source is None else source[0],
                        source_t_sec=None if source is None else source[1],
                        lag_frames=None if source is None else fi-source[0],
                        lag_sec=None if source is None else t-source[1],
                        initialized=source is not None)
        return result, metadata

    def _run(self) -> None:
        """ロックは受け渡しだけに使い、CNN 実行中は保持しない。"""
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                fi, t, cells = self._pending
                self._pending = None
            try:
                result = (self.detector._empty_pair() if cells is None else
                          self.detector.detect_cells(cells))
            except Exception as error:
                with self._condition:
                    self._error = error
                return
            with self._condition:
                self._latest, self._source = result, (fi, t)
                self._condition.notify_all()

    def close(self) -> None:
        """未着手分を破棄し、実行中の一回が終わるまで待って回収する。"""
        with self._condition:
            self._closed, self._pending = True, None
            self._condition.notify_all()
        self._thread.join()
