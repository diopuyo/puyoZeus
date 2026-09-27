"""spawnした認識プロセスと評価worker間の、欠落しない差分通知FIFO。"""
from __future__ import annotations

from dataclasses import fields, replace
import multiprocessing as mp
import json
from pathlib import Path
import pickle
from queue import Empty
import random
import time
import traceback
from types import SimpleNamespace
from typing import Any, Iterator

import cv2
import numpy as np

from .live_bridge import RecognitionBridge, RecognitionNotice, QUEUE_CAPACITY, POLL_SEC, JOIN_SEC, recognize
from .live_source import VideoFileSource

PROTOCOL = pickle.HIGHEST_PROTOCOL
NOTICE_FIELDS = frozenset({'side', 'state', 'confirmed_board', 'score', 'score_delta',
    'chain_event', 'next_pair', 'dnext_pair', 'next_slide_motion', 'landing_chain_started'})
EMPTY_COUNTER_RESULT = (0.0, float('nan'), float('nan'))


class NoticeDeltaCodec:
    """画像を送らず、確定盤面・状態・得点・連鎖通知の変更フィールドだけ送る。"""

    def __init__(self) -> None:
        self.previous: dict[str, bytes] = {}

    def encode(self, notice: RecognitionNotice) -> tuple[RecognitionNotice, dict[str, bytes]]:
        result = notice.result()
        values = {'root': {f.name: getattr(result, f.name) for f in fields(result)
                           if f.name not in ('p1', 'p2')}, 'pipeline': notice.pipeline}
        for side in ('p1', 'p2'):
            for field in fields(getattr(result, side)):
                if field.name in NOTICE_FIELDS:
                    values[side + '.' + field.name] = getattr(getattr(result, side), field.name)
        serialized = {key: pickle.dumps(value, protocol=PROTOCOL) for key, value in values.items()}
        delta = {key: value for key, value in serialized.items() if self.previous.get(key) != value}
        self.previous = serialized
        return replace(notice, result_bytes=b'', pipeline=None), delta

    def decode(self, packet: tuple[RecognitionNotice, dict[str, bytes]]) -> RecognitionNotice:
        from src.recognition_pipeline import PipelineResult, SideResult
        notice, delta = packet
        self.previous.update(delta)
        values = {key: pickle.loads(value) for key, value in self.previous.items()}
        sides = {side: SideResult(cnn_board=None, inferred_board=None, drift=None,
                                 **{key.split('.', 1)[1]: value for key, value in values.items()
                                    if key.startswith(side + '.')}) for side in ('p1', 'p2')}
        result = PipelineResult(**values['root'], **sides)
        return replace(notice, result_bytes=pickle.dumps(result, protocol=PROTOCOL),
                       pipeline=values['pipeline'])


def recognition_worker(queue: Any, cancel: Any, config: dict[str, Any],
                       video: str, bounds: tuple, realtime: bool, device: str | None,
                       latest_frame: Any = None) -> None:
    """GPU認識器はspawn先で生成し、親のCUDA状態を継承しない。"""
    from src.recognition_pipeline import RecognitionPipeline
    from scripts.run_e3_exchange_eval_20260926 import SEED
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    capture = None
    try:
        pipe = RecognitionPipeline.load_default(**config)
        fps, start, end, stride = bounds
        if device:
            from .live_device_session import DeviceSession
            pipe = DeviceSession(pipe, Path(device), (end-start)/fps, queue)
            source = pipe.source
        else:
            capture = cv2.VideoCapture(video)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            source = VideoFileSource(capture, fps, start, end, stride, realtime)
        send_notices(queue, cancel, pipe, source, latest_frame)
    except BaseException:
        queue.put(('error', traceback.format_exc()))
    finally:
        if capture is not None:
            capture.release()
        queue.put(('done', None))


def send_notices(queue: Any, cancel: Any, pipe: Any, source: Any, latest_frame: Any = None) -> None:
    codec = NoticeDeltaCodec()
    count, wire_bytes = 0, 0
    for frame in source:
        if cancel.is_set():
            break
        notice = recognize(pipe, frame)
        if not getattr(pipe, 'publishing_ready', True):
            continue
        if latest_frame is not None:
            latest_frame.value = notice.frame
        packet = codec.encode(notice)
        wire_bytes += len(pickle.dumps(packet, protocol=PROTOCOL))
        queue.put(('notice', packet))
        count += 1
    queue.put(('summary', dict(dropped=source.dropped, sent=count, wire_bytes=wire_bytes)))


class ProcessRecognitionBridge(RecognitionBridge):
    """評価はこのprocessの単一所有者。認識との間では差分だけをIPCする。"""

    def __init__(self, realtime: bool, callback: Any, video: Path,
                 coalesce: bool = False, device: Path | None = None) -> None:
        super().__init__(realtime, callback)
        context = mp.get_context('spawn')
        self.queue = context.Queue(maxsize=QUEUE_CAPACITY)
        self.latest_frame = context.Value('q', -1)
        self.cancel = context.Event()
        self.context, self.video, self.device = context, str(video), str(device) if device else None
        self.coalesce, self.skip_features = coalesce, False
        self.codec = NoticeDeltaCodec()
        self.received, self.sent, self.wire_bytes = 0, 0, 0
        self.finished = False
        self.feature_skips: dict[str, int] = {}
        self.source = SimpleNamespace(dropped=0)
        self.on_hold: Any = None
        self.duration = 0.0
        self.config_path: Path | None = None

    def pipeline_config(self, **kwargs: Any) -> dict[str, Any]:
        if self.config_path:
            self.config_path.write_text(json.dumps(kwargs, indent=2), encoding='utf-8')
        return kwargs

    def open_capture(self, video: str) -> Any:
        if self.device:
            from .live_device import DeviceMetadataCapture
            return DeviceMetadataCapture(self.duration)
        return cv2.VideoCapture(video)

    def observations(self, pipe: Any, capture: Any, fps: float,
                     start: int, end: int, stride: int) -> Iterator[RecognitionNotice]:
        self.frame_bounds = (fps, start, end, stride)
        self.worker = self.context.Process(target=recognition_worker, name='recognition-process',
            args=(self.queue, self.cancel, pipe, self.video, self.frame_bounds, self.realtime,
                  self.device, self.latest_frame))
        self.worker.start()
        while not self.finished:
            batch = self._receive_batch()
            self.batch_ready_at = time.perf_counter()
            if batch:
                self.batch_starts.append(self.batch_ready_at)
                self.batch_sizes.append(len(batch))
            for index, notice in enumerate(batch):
                self.batch_remaining = len(batch)-index-1
                self.skip_features = self.feature_is_stale(notice.frame)
                yield from self.timed_notice(notice)
        if self.sent != self.received:
            raise RuntimeError(f'通知欠落: {self.received}/{self.sent}')

    def feature_is_stale(self, frame: int) -> bool:
        """EOF/summary等の制御通知を新しい盤面と誤認しない。"""
        return self.coalesce and self.latest_frame.value > frame

    def _receive_batch(self) -> list[RecognitionNotice]:
        try:
            messages = [self.queue.get(timeout=POLL_SEC)]
        except Empty:
            if not self.worker.is_alive():
                raise RuntimeError(f'認識process異常終了: {self.worker.exitcode}')
            return []
        depth = self.queue.qsize()
        self.queue_max = max(self.queue_max, min(QUEUE_CAPACITY, depth+1))
        self.pending_max = max(self.pending_max, depth+1)
        # デバイス制御通知と盤面通知の順序を維持し、古い盤面でHOLDを上書きしない。
        for _ in range(0 if self.device else depth):
            try:
                messages.append(self.queue.get_nowait())
            except Empty:
                break
        return [notice for message in messages if (notice := self._message(message)) is not None]

    def _message(self, message: tuple[str, Any]) -> RecognitionNotice | None:
        kind, value = message
        if kind == 'error':
            raise RuntimeError(value)
        if kind == 'summary':
            self.source.dropped, self.sent, self.wire_bytes = value['dropped'], value['sent'], value['wire_bytes']
        if kind == 'done':
            self.finished = True
        if kind == 'hold' and self.on_hold:
            if isinstance(value, dict) and value.get('epoch', -1) != getattr(self, 'input_epoch', -1):
                self.input_changed = value.get('epoch', -1) > 0
                self.input_epoch = value.get('epoch', -1)
            self.on_hold(value)
        if kind != 'notice':
            return None
        notice = self.codec.decode(value)
        self.received += 1
        self.recognition_rows.append(dict(frame=notice.frame, t_sec=notice.t_sec,
            captured_at=notice.captured_at, recognized_at=notice.recognized_at,
            dropped_before=notice.dropped_before))
        return notice

    def close(self) -> None:
        self.cancel.set()
        if self.worker is not None:
            self.worker.join(JOIN_SEC)
            if self.worker.is_alive():
                self.worker.terminate()
                self.worker.join(JOIN_SEC)
                raise RuntimeError('認識processを強制停止しました')
        self.queue.close()


def coalescing_overlay(base: type, bridge: ProcessRecognitionBridge) -> type:
    """通知・発火・終了・会計・着弾は常に処理し、通常の特徴更新だけ間引く。"""
    class CoalescingOverlay(base):
        def _refresh_features(self, snapshot: Any, t_sec: float) -> None:
            if bridge.skip_features:
                bridge.feature_skips['refresh'] = bridge.feature_skips.get('refresh', 0)+1
                return
            super()._refresh_features(snapshot, t_sec)

        def _static(self, snapshot: Any, t_sec: float) -> None:
            # 区間を閉じるstaticは状態通知の一部なので、進行中の撃ち合いでは省かない。
            if bridge.skip_features and self.tracker.current is None:
                bridge.feature_skips['static'] = bridge.feature_skips.get('static', 0)+1
                return
            super()._static(snapshot, t_sec)
    return CoalescingOverlay


def coalescing_cache(base: type, bridge: ProcessRecognitionBridge) -> type:
    """表示補助用の盤面探索も途中盤面では前回値を使う。通知会計には触れない。"""
    class CoalescingCache(base):
        def update(self, *args: Any, **kwargs: Any) -> tuple:
            if bridge.skip_features:
                bridge.feature_skips['legacy'] = bridge.feature_skips.get('legacy', 0)+1
                return (self._adv, self._threat, self._drivers,
                        self._ukey1, self._ukey2, self._sat1, self._sat2)
            return super().update(*args, **kwargs)
    return CoalescingCache


def coalescing_counter(base: type, bridge: ProcessRecognitionBridge) -> type:
    """旧表示補助のMC探索も途中盤面では再探索しない。交換状態機械とは独立。"""
    class CoalescingCounter(base):
        def update(self, *args: Any, **kwargs: Any) -> tuple:
            budget = args[2] if len(args) > 2 else kwargs.get('budget_sec', 0.0)
            if bridge.skip_features and budget > 0:
                bridge.feature_skips['counter'] = bridge.feature_skips.get('counter', 0)+1
                return self._last_result or EMPTY_COUNTER_RESULT
            return super().update(*args, **kwargs)
    return CoalescingCounter
