"""spawnした認識プロセスと評価worker間の、欠落しない差分通知FIFO。"""
from __future__ import annotations

from dataclasses import fields, replace
from contextlib import ExitStack
import multiprocessing as mp
import json
import logging
import os
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

from .live_bridge import (RecognitionBridge, RecognitionNotice, QUEUE_CAPACITY, POLL_SEC,
                          JOIN_SEC, recognize, recognition_guard_audits)
from .live_source import VideoFileSource

PROTOCOL = pickle.HIGHEST_PROTOCOL
NOTICE_FIELDS = frozenset({'side', 'state', 'confirmed_board', 'score', 'score_delta',
    'chain_event', 'next_pair', 'dnext_pair', 'next_slide_motion', 'landing_chain_started',
    'midchain_board', 'prefire_snapshot', 'prefire_origin_hold', 'evaluation_confirmed_board'})
EMPTY_COUNTER_RESULT = (0.0, float('nan'), float('nan'))
PROGRESS_PERIOD_SEC = 1.0


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
                       latest_frame: Any = None, lifecycle: bool = False,
                       live_config: str | None = None, audit_path: str | None = None,
                       fault_plan: dict | None = None, parent_pid: int | None = None,
                       retention_path: str | None = None,
                       placement_log_path: str | None = None) -> None:
    """GPU認識器はspawn先で生成し、親のCUDA状態を継承しない。"""
    from .live_lifetime import protect_parent
    protect_parent(parent_pid)
    from src.recognition_pipeline import RecognitionPipeline
    from scripts.run_e3_exchange_eval_20260926 import SEED
    import torch
    from .live_cpu import apply_runtime
    apply_runtime('recognition')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    from .live_notice_sender import wrap_queue, close_queue
    queue = wrap_queue(queue, cancel)  # 既定OFFでは同一objectを返す
    try:
        with ExitStack() as storage:
            configure_recognition_retention(storage)
            pipe = RecognitionPipeline.load_default(**config)
            from .live_native_hsv import record_runtime
            record_runtime(pipe)
            from .live_board_runtime import record_runtime as record_board_runtime
            record_board_runtime(pipe)
            from .live_native_read_board_runtime import record_runtime as record_read_board_runtime
            record_read_board_runtime(pipe)
            if os.environ.get('PUYO_OBS_PREPARE_EXACT') == '1':
                from .live_obs_prepare import prepare_recognition
                prepare_recognition(pipe)
            if os.environ.get('PUYO_OBS_DEEP_TIMING') == '1':
                from .live_obs_recognition_timing import install
                install(pipe)
            configure_placement_log(storage, pipe, placement_log_path)
            run_recognition_input(queue, cancel, pipe, video, bounds, realtime, device,
                                  latest_frame, lifecycle, live_config, audit_path, fault_plan, retention_path)
    except BaseException:
        queue.put(('error', traceback.format_exc()))
    finally:
        queue.put(('done', None))
        close_queue(queue)


def configure_recognition_retention(storage: ExitStack) -> None:
    from .live_spool import bounded_enabled
    if bounded_enabled():
        from .live_cache import bounded_chain_caches
        storage.enter_context(bounded_chain_caches())


def configure_placement_log(storage: ExitStack, pipe: Any, path: str | None) -> None:
    """照合の採否はそのまま、試験時の原票保存だけを認識所有者から分離する。"""
    if not path:
        return
    recorder = pipe._placement_reconcile
    if recorder is None:
        logging.getLogger(__name__).warning('placementログ指定がありますが設置照合器が無効です: %s', path)
    elif os.environ.get('PUYO_OBS_FAST_AUDIT') != '1':
        recorder.log_to(Path(path))
    else:
        from .live_obs_audit import open_audit
        from . import live_stage_timing as timing
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stream = open_audit(target)
        storage.callback(stream.close)
        def write(row: dict) -> None:
            payload = json.dumps(row, ensure_ascii=False)+'\n'
            timing.measure('placement_audit_write', stream.write, payload)
        recorder.sink = write


def run_recognition_input(queue: Any, cancel: Any, pipe: Any, video: str, bounds: tuple,
                          realtime: bool, device: str | None, latest_frame: Any,
                          lifecycle: bool, live_config: str | None,
                          audit_path: str | None, fault_plan: dict | None,
                          retention_path: str | None = None) -> None:
    from .live_audit import RecognitionAudit
    capture, fault = None, None
    try:
        fps, start, end, stride = bounds
        # コマ落ちに強い数え方 (既定OFF): 動画は stride、OBS 実機は 1 (index = round(秒×30))。
        configure_media_tick(pipe, 1 if device else stride)
        if device:
            from .live_device_session import DeviceSession
            pipe = DeviceSession(pipe, Path(device), (end-start)/fps, queue)
            source = pipe.source
        else:
            capture = cv2.VideoCapture(video)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            if fault_plan:
                from .live_faults import FaultPlan, FaultInjector
                fault = FaultInjector(FaultPlan(**fault_plan), lambda event: queue.put(('fault', event)))
            source = VideoFileSource(capture, fps, start, end, stride, realtime, fault=fault)
            if lifecycle:
                from .live_video_session import video_session
                from .live_config import input_identity
                pipe, source = video_session(pipe, source, queue, input_identity(live_config, video), bool(fault))
        with ExitStack() as storage:
            from . import live_stage_timing as timing
            timing.install(storage, Path(retention_path) if retention_path else None, 'recognition')
            diagnostics = recognition_logs(storage, source, retention_path, pipe)
            audit = RecognitionAudit(Path(audit_path) if audit_path else None)
            if hasattr(audit.rows, 'close'):
                storage.callback(audit.rows.close)
            send_notices(queue, cancel, pipe, source, latest_frame, audit, diagnostics)
    finally:
        if getattr(getattr(pipe, 'pipe', pipe), '_enable_match_end_pair', False):
            from src.async_match_end_pair import close_pair_worker
            close_pair_worker(pipe)
        from .live_stage_timing import finish
        finish(aborted=True)
        from .live_b3 import close_capture
        close_capture(pipe)
        if capture is not None:
            capture.release()
        if fault is not None:
            fault.close()


FRAME_DROP_ROBUST_MARK = 'FRAME_DROP_ROBUST '   # 実行時の刻みの幅を run.log へ残す印 (検証の梯子 原則7)


def configure_media_tick(pipe: Any, step: int) -> None:
    """コマ落ちに強い数え方 (2026-10-23、既定OFF): 最初の update の前に刻みの幅を認識器へ渡す。

    OFF (PUYO_LIVE_FRAME_DROP_ROBUST != 1) では何もしない (呼び出しも印も無し = 従来と同一)。
    ON では実行時の値 (有効か・刻みの幅) を標準出力へ 1 行残し、測定器が実 run の値を読めるようにする。
    """
    from .live_speed import frame_drop_robust_enabled, rules_off_from_env
    if not frame_drop_robust_enabled():
        return
    target = getattr(pipe, 'pipe', pipe)
    target.set_media_tick_step(step)
    from src.production_config import LIVE_FRAME_DROP_ROBUST_SITES
    state = dict(requested_step=step, media_tick_step=getattr(target, 'media_tick_step', None),
                 enabled=bool(getattr(target, '_enable_media_tick_counters', False)),
                 sites=sorted(LIVE_FRAME_DROP_ROBUST_SITES))   # 認識 process が実際に読んだ採否 (R06・R10・R15)
    rules_off = rules_off_from_env()
    if rules_off:   # 箇所別 OFF (既定は空 = 印も従来と同じ)。差し替えた箇所を実行時の値として残す
        from .live_fdr_rules import apply_rules_off
        state['rules_off'] = list(apply_rules_off(rules_off))
    print(FRAME_DROP_ROBUST_MARK+json.dumps(state), flush=True)


def send_notices(queue: Any, cancel: Any, pipe: Any, source: Any, latest_frame: Any = None,
                 audit: Any = None, diagnostics: Any = None) -> None:
    from .live_memory import MemoryProbe
    from .live_b3 import prepare_capture
    prepare_capture(pipe)
    probe = MemoryProbe('recognition')
    codec = NoticeDeltaCodec()
    count, wire_bytes, gated = 0, 0, 0
    last_progress = float('-inf')
    for frame in source:
        if cancel.is_set():
            break
        started, cpu = time.perf_counter(), time.process_time()
        from .live_row1_audit import action_scope
        with action_scope(audit, frame):
            notice = recognize(pipe, frame)
        from .live_degrade import feedback_event
        feedback_event(source, notice)
        ready = getattr(pipe, 'publishing_ready', True)
        if frame.media_sec-last_progress >= PROGRESS_PERIOD_SEC:
            queue.put(('progress', dict(t_sec=frame.media_sec, dropped=source.dropped)))
            last_progress = frame.media_sec
        if audit is not None:
            audit.append(notice, frame, time.perf_counter()-started, time.process_time()-cpu, ready)
        if diagnostics is not None:
            from .live_retention import archive_recognition_alerts
            archive_recognition_alerts(pipe, diagnostics)
            from .live_observation_diagnostics import archive_terminal_observation
            archive_terminal_observation(pipe, notice, ready)
        if probe.path:
            probe.sample(frame.media_sec, dict(pipe=getattr(pipe, 'pipe', pipe), audit=audit,
                                              source=source, codec=codec))
        if not ready:
            from .live_stage_timing import finish
            finish(held=True)
            gated += 1
            continue
        if latest_frame is not None:
            latest_frame.value = notice.frame
        packet = codec.encode(notice)
        wire_bytes += len(pickle.dumps(packet, protocol=PROTOCOL))
        sent_at = time.perf_counter()
        from .live_stage_timing import measure, finish
        measure('notice_queue', queue.put, ('notice', packet))
        finish(held=False)
        if audit is not None:
            audit.queue_wait(time.perf_counter()-sent_at)
        count += 1
    finish_recognition(queue, source, audit, count, wire_bytes, gated, pipe)


def report_media_tick(pipe: Any) -> None:
    """コマ落ちに強い数え方 (既定OFF) の診断の数えを run.log へ 1 行残す。OFF では何もしない。"""
    from .live_speed import frame_drop_robust_enabled
    if not frame_drop_robust_enabled():
        return
    from src.media_tick import audit_snapshot
    target = getattr(pipe, 'pipe', pipe)
    print(FRAME_DROP_ROBUST_MARK+json.dumps(dict(
        audit=audit_snapshot(), media_tick_step=getattr(target, 'media_tick_step', None),
        unset_calls=getattr(target, '_media_tick_unset_calls', None))), flush=True)


def report_recog_speed() -> None:
    """live_recog_speed_py が ON のとき、どの項目が働き、実際に何回使い回したかを run.log へ 1 行残す。

    要求値 (設定) だけでは子 process で本当に効いたかが分からない (fail-silent の防止)。OFF では何も出さない。
    """
    from src.recog_speed_py import runtime_report
    report = runtime_report()
    if report is not None:
        from src.live_board_vectorized import STATS as board_stats
        report['board_stats'] = dict(board_stats)
        print('RECOG_SPEED_PY '+json.dumps(report), flush=True)


def finish_recognition(queue: Any, source: Any, audit: Any, count: int,
                       wire_bytes: int, gated: int, pipe: Any = None) -> None:
    report_media_tick(pipe)
    report_recog_speed()
    from .live_cpu import runtime_snapshot
    runtime = runtime_snapshot('recognition')
    if audit is not None:
        audit.save(runtime, source)
    from .live_notice_sender import queue_stats
    summary = dict(dropped=source.dropped, sent=count, wire_bytes=wire_bytes, runtime=runtime,
                   dropped_times=list(getattr(source, 'dropped_times', [])),
                   recognition_audits=recognition_guard_audits(pipe, count+gated),
                   gated=gated+getattr(source, 'gated_frames', 0))
    if (sender := queue_stats(queue)) is not None:
        summary['notice_sender'] = sender  # 既定OFFでは従来と同じキー集合
    queue.put(('summary', summary))


def recognition_logs(storage: ExitStack, source: Any, path: str | None, pipe: Any = None) -> Any:
    from .live_spool import bounded_enabled, DiskFIFO
    if path is None or not bounded_enabled():
        return None
    from .live_retention import spool
    raw = source
    while hasattr(raw, 'source'):
        raw = raw.source
    if hasattr(raw, 'dropped_times'):
        raw.dropped_times = spool(storage, Path(path), 'dropped_times')
    target = getattr(pipe, 'pipe', pipe)
    if target is not None:
        target._live_terminal_observations = spool(storage, Path(path), 'terminal_observations')
    for side in ('1p', '2p'):
        name = '_pending_tsumo_'+side
        if hasattr(target, name):
            fifo = DiskFIFO(Path(path)/(name+'.pickle'))
            for value in getattr(target, name):
                fifo.append(value)
            setattr(target, name, fifo)
            storage.callback(fifo.close)
    return spool(storage, Path(path), 'recognition_alerts')


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
        self.placement_log_path: str | None = None

    def pipeline_config(self, **kwargs: Any) -> dict[str, Any]:
        from .live_speed import recognition_config
        kwargs = recognition_config(kwargs, live_input=bool(self.device))
        if self.config_path:
            self.config_path.write_text(json.dumps(kwargs, indent=2), encoding='utf-8')
        return kwargs

    def placement_log(self, path: Path) -> None:
        """実認識器が存在する子プロセスへ、設置監査の出力先を渡す。"""
        self.placement_log_path = str(path)

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
                  self.device, self.latest_frame, getattr(self, 'lifecycle', False),
                  getattr(self, 'live_config', None), getattr(self, 'audit_path', None),
                  getattr(self, 'fault_plan', None), os.getpid(),
                  str(self.config_path.parent/'spool') if self.config_path else None,
                  self.placement_log_path))
        self.worker.start()
        from .live_cpu import apply_runtime
        self.cpu_runtime = apply_runtime('evaluation')
        while not self.finished:
            from .live_completion import receive_completion
            if not receive_completion(self, POLL_SEC):
                continue
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
        for _ in range(0 if self.device or getattr(self, 'lifecycle', False) else depth):
            try:
                messages.append(self.queue.get_nowait())
            except Empty:
                break
        return [notice for message in messages if (notice := self._message(message)) is not None]

    def _message(self, message: tuple[str, Any]) -> RecognitionNotice | None:
        kind, value = message
        if kind == 'progress':
            self.progress = value
        if kind == 'fault' and getattr(self, 'runtime_path', None):
            with (self.runtime_path/'faults.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value)+'\n')
        if kind == 'error':
            raise RuntimeError(value)
        if kind == 'summary':
            self.recognition_summary_received = True
            self.recognition_runtime = value.get('runtime')
            self.recognition_audits = value.get('recognition_audits', {})
            self.source.dropped, self.sent, self.wire_bytes = value['dropped'], value['sent'], value['wire_bytes']
            self.gated_frames = value.get('gated', 0)
            self.notice_sender = value.get('notice_sender')
            self.source.dropped_times = value.get('dropped_times', [])
        if kind == 'done':
            self.finished = True
        if kind == 'hold' and self.on_hold:
            if isinstance(value, dict) and value.get('phase') == 'no_puyo_screen':
                diagnostics = value.get('input_diagnostics') or {}
                if (diagnostics.get('transport_ready') is True
                        and diagnostics.get('verifier_fail_streak', 0)
                        >= diagnostics.get('verifier_fail_limit', float('inf'))):
                    self.input_failure_epoch = value.get('epoch')
            if isinstance(value, dict) and value.get('epoch', -1) != getattr(self, 'input_epoch', -1):
                self.input_boundary_recovery = (value.get('device_id') is not None
                    and value.get('device_id') == getattr(self, 'input_device_id', None))
                self.input_device_id = value.get('device_id')
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
