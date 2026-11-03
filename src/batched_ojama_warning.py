"""予告アイコンの CNN 推論を 1 コマ 1 回にまとめる (既定 OFF の試作、2026-10-23)。

判定の手順は OjamaWarningDetector._classify_cell と同じ:
  1. セルが空 → empty
  2. セル全体の V 標準偏差 < PRESENCE_V_STD_MIN → empty (CNN を呼ばない)
  3. CNN の confidence ≥ 閾値 → その種類
  4. テンプレ NCC → 種類
  5. HSV 特徴の投票
変えるのは 3 の CNN を「存在判定を通ったセルだけ、まとめて 1 回で推論する」所だけ。
バッチの浮動小数の丸めは単体推論とわずかに違いうるので、閾値の近く
(|conf-閾値| ≤ BORDERLINE_MARGIN) のセルは単体推論をやり直して従来と同じ値で判定する。
同じ画素のパッチは従来の LRU と同じく厳密一致のときだけ再利用する。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.ojama_warning import (CELL_COUNT, FRAME_HEIGHT, FRAME_WIDTH, ICON_EMPTY,
    P1_BOARD_X, P2_BOARD_X, PRESENCE_V_STD_MIN, OjamaWarningDetector, OjamaWarningResult)
from src.parallel_ojama_warning import ParallelOjamaWarningDetector

PIXEL_MAX = 255.0
# 単体とバッチの丸め差は 1e-6 程度。閾値からこの幅以内なら単体推論で判定し直す。
BORDERLINE_MARGIN = 1e-4
# 従来の予告 CNN の LRU (scripts/b2_recollect_cache.ICON_CACHE_SIZE) と同じ上限。
ICON_CACHE_SIZE = 512


class BatchedOjamaWarningDetector(OjamaWarningDetector):
    """12 セルの CNN を 1 回の forward にまとめ、判定は旧 detect と同じ順で行う。"""

    def __init__(self, cache_size: int = ICON_CACHE_SIZE, onnx_path: str | None = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._icon_cache: OrderedDict = OrderedDict()
        self._icon_cache_size = cache_size
        # 試作の比較用: ONNX Runtime (1 スレッド) で同じ forward を行う。既定は torch。
        self._session = None
        if onnx_path is not None and self._cnn is not None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(onnx_path, options, providers=['CPUExecutionProvider'])
        self.stats = dict(frames=0, present_cells=0, cache_hits=0, batched_cells=0,
                          forwards=0, borderline=0)

    def close(self) -> None:
        """並列版と同じ口。所有する worker は無い。"""

    def detect(self, frame: np.ndarray) -> tuple[OjamaWarningResult, OjamaWarningResult]:
        cells = self.extract_cells(frame)
        return self._empty_pair() if cells is None else self.detect_cells(cells)

    def extract_cells(self, frame: np.ndarray, copy: bool = False) -> list[np.ndarray] | None:
        """旧 detect と同じ切り出し (1P 6 セル → 2P 6 セル)。copy=True は先読み用の複製。"""
        if frame is None or frame.ndim != 3 or frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
            return None
        cells = [self._extract_cell(frame, x, i)
                 for x in (P1_BOARD_X, P2_BOARD_X) for i in range(CELL_COUNT)]
        return [cell.copy() for cell in cells] if copy else cells

    def detect_cells(self, cells: list[np.ndarray]) -> tuple[OjamaWarningResult, OjamaWarningResult]:
        self.stats['frames'] += 1
        feats = [None if cell.size == 0 else self._patch_features(cell) for cell in cells]
        predictions = self._predict_present(cells, feats)
        kinds = [self._decide(cell, feat, predictions.get(i))
                 for i, (cell, feat) in enumerate(zip(cells, feats))]
        self.last_kinds = kinds  # 測定台用のセル別の種類 (判定には使わない)
        return (ParallelOjamaWarningDetector._result('1P', kinds[:CELL_COUNT]),
                ParallelOjamaWarningDetector._result('2P', kinds[CELL_COUNT:]))

    def _decide(self, cell: np.ndarray, feats: dict | None,
                prediction: tuple[str, float] | None) -> str:
        """OjamaWarningDetector._classify_cell と同じ分岐。CNN の値だけ外から受け取る。"""
        if cell.size == 0 or feats is None:
            return ICON_EMPTY
        if feats["v_std"] < PRESENCE_V_STD_MIN:
            return ICON_EMPTY
        if prediction is not None:
            kind, conf = prediction
            if conf >= self._cnn_confidence_min:
                return kind
        templ_kind = self._match_templates(cell)
        if templ_kind is not None:
            return templ_kind
        return self._classify_by_features(feats)

    def _predict_present(self, cells: list[np.ndarray],
                         feats: list[dict | None]) -> dict[int, tuple[str, float]]:
        """存在判定を通ったセルだけを集め、キャッシュに無いものを 1 回で推論する。"""
        if self._cnn is None:
            return {}
        found: dict[int, tuple[str, float]] = {}
        todo: list[tuple[int, bytes, np.ndarray]] = []
        for i, (cell, feat) in enumerate(zip(cells, feats)):
            if feat is None or feat["v_std"] < PRESENCE_V_STD_MIN:
                continue
            patch = self._extract_center_patch(cell)
            if patch is None:
                continue
            self.stats['present_cells'] += 1
            key = patch.tobytes()
            if key in self._icon_cache:
                self._icon_cache.move_to_end(key)
                found[i] = self._icon_cache[key]
                self.stats['cache_hits'] += 1
            else:
                todo.append((i, key, patch))
        if todo:
            for (i, key, patch), value in zip(todo, self._forward([p for _, _, p in todo])):
                self._remember(key, value)
                found[i] = value
        return found

    def _forward(self, patches: list[np.ndarray]) -> list[tuple[str, float]]:
        """単体推論と同じ前処理で 1 回の forward。閾値付近だけ単体推論に戻す。"""
        from src.ojama_cnn import OJAMA_CLASSES
        self.stats['forwards'] += 1
        self.stats['batched_cells'] += len(patches)
        x = torch.from_numpy(np.stack(patches)).float() / PIXEL_MAX
        x = x.permute(0, 3, 1, 2)
        with torch.no_grad():
            if self._session is not None:
                logits = torch.from_numpy(self._session.run(None, {'x': x.contiguous().numpy()})[0])
            else:
                logits = self._cnn.forward(x)
            probs = F.softmax(logits, dim=1)
            best, indices = probs.max(dim=1)
        values = []
        for patch, conf, idx in zip(patches, best.tolist(), indices.tolist()):
            if abs(conf - self._cnn_confidence_min) <= BORDERLINE_MARGIN:
                self.stats['borderline'] += 1
                values.append(self._cnn.predict_class_readonly(patch))
            else:
                values.append((OJAMA_CLASSES[idx], float(conf)))
        return values

    def _remember(self, key: bytes, value: tuple[str, float]) -> None:
        self._icon_cache[key] = value
        self._icon_cache.move_to_end(key)
        if len(self._icon_cache) > self._icon_cache_size:
            self._icon_cache.popitem(last=False)


class PrefetchingWarning:
    """予告アイコン読取を盤面認識の前に 1 本の worker で始めておく (b3_warning_prefetch、既定 OFF)。

    先読みは同じコマの 12 セルを複製して BatchedOjamaWarningDetector.detect_cells に渡すだけで、
    判定の関数・入力の画素は同期読取と同じ。detect(frame) は同じ frame の先読みがあればその結果を、
    無ければ (入力世代の切替直後など) その場で同期読取した結果を返す。detector の状態 (LRU) に
    触れるのは常に 1 本だけ: 先読みが残っていれば、次の先読み・同期読取の前に必ず待つ。
    """

    def __init__(self, detector: BatchedOjamaWarningDetector) -> None:
        from concurrent.futures import ThreadPoolExecutor
        from weakref import finalize
        self.detector = detector
        # ObservationCapture の資産確認と b2_recollect_cache.enable が参照する属性を見せる
        self._cnn, self._templates = detector._cnn, detector._templates
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='b3-warning-prefetch')
        self._finalizer = finalize(self, self._pool.shutdown, wait=True)
        self._pending: tuple[Any, Any] | None = None
        self.stats = dict(submitted=0, used=0, missed=0, discarded=0)

    def prefetch(self, frame: np.ndarray) -> bool:
        """同じコマの読取を worker で始める。形が違うコマは何もしない (同期側が空を返す)。"""
        self._settle()
        cells = self.detector.extract_cells(frame, copy=True)
        if cells is None:
            return False
        self._pending = (frame, self._pool.submit(self.detector.detect_cells, cells))
        self.stats['submitted'] += 1
        return True

    def detect(self, frame: np.ndarray) -> tuple[OjamaWarningResult, OjamaWarningResult]:
        pending, self._pending = self._pending, None
        if pending is not None:
            source, future = pending
            result = future.result()
            if source is frame:
                self.stats['used'] += 1
                return result
            self.stats['discarded'] += 1
        self.stats['missed'] += 1
        return self.detector.detect(frame)

    def _settle(self) -> None:
        """使われなかった先読みは終わるのを待って捨てる (worker と同期読取を重ねない)。"""
        pending, self._pending = self._pending, None
        if pending is not None:
            self.stats['discarded'] += 1
            if pending[1].exception() is not None:  # 使わない結果の例外は数えるだけ (出力に関係しない)
                self.stats['discarded_errors'] = self.stats.get('discarded_errors', 0) + 1

    def close(self) -> None:
        """入力世代の終了時に worker を回収する。繰り返し呼んでも安全。"""
        try:
            self._settle()
        finally:
            self._finalizer()
