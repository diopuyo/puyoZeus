"""B2専用の認識通知FIFOと、既存評価ループへの接続。"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import pickle
from queue import Empty, Full, Queue
from threading import Event, Thread
import time
from typing import Any, Callable, Iterator

from .live_source import CapturedFrame, FrameSource, VideoFileSource

BATCH_PERIOD_SEC = 0.5
QUEUE_CAPACITY = 30
POLL_SEC = 0.05
JOIN_SEC = 10.0
MILLISECONDS = 1000.0


@dataclass(frozen=True)
class PipelineView:
    """可変pipelineを共有せず、当該観測時点の値だけを評価へ渡す。"""

    counts: tuple[int, int]
    formula_totals: tuple[float | None, float | None]
    displayed_scores: tuple[float | None, float | None]
    formula_visible: tuple[bool, bool]

    def tsumo_count(self, side: str) -> int:
        return self.counts[('1P', '2P').index(side)]


@dataclass(frozen=True)
class RecognitionNotice:
    """bytes化した盤面・状態・得点は、認識の次フレームから独立する。"""

    frame: int
    t_sec: float
    captured_at: float
    recognized_at: float
    result_bytes: bytes
    pipeline: PipelineView
    dropped_before: int

    def result(self) -> Any:
        # 自プロセスが生成したbytesだけを読む。外部pickle入力は受け付けない。
        return pickle.loads(self.result_bytes)


def recognize(pipe: Any, frame: CapturedFrame) -> RecognitionNotice:
    """OCRの実読と段累積も認識所有スレッド内で同時点に確定する。"""
    from src.exchange_event_m0 import formula_totals_from_pipeline
    from src.exchange_event_overlay import displayed_scores_from_pipeline, formula_visible_from_pipeline
    result = pipe.update(frame.index, frame.media_sec, frame.image)
    view = PipelineView(tuple(pipe.tsumo_count(side) for side in ('1P', '2P')),
                        formula_totals_from_pipeline(pipe),
                        displayed_scores_from_pipeline(pipe, frame.image),
                        formula_visible_from_pipeline(pipe))
    frozen = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    return RecognitionNotice(frame.index, frame.media_sec, frame.captured_at,
                             time.perf_counter(), frozen, view, frame.dropped_before)


class RecognitionBridge:
    """通知FIFOは単一writer/single reader。通知自体は欠落させない。"""

    def __init__(self, realtime: bool, callback: Callable[..., None],
                 source_factory: Callable[..., FrameSource] = VideoFileSource) -> None:
        self.realtime, self.callback = realtime, callback
        self.source_factory = source_factory
        self.queue: Queue[RecognitionNotice] = Queue(maxsize=QUEUE_CAPACITY)
        self.done, self.cancel = Event(), Event()
        self.error: BaseException | None = None
        self.worker: Thread | None = None
        self.source: FrameSource | None = None
        self.queue_max = 0
        self.batch_remaining = 0
        self.pending_max = 0
        self.batch_starts: list[float] = []
        self.batch_sizes: list[int] = []
        self.recognition_rows: list[dict[str, Any]] = []
        self.frame_bounds = (0.0, 0, 0, 1)
        self.profile: list[dict[str, float]] = []
        self.batch_ready_at = 0.0
        self.meter: Any = None
        self.counters: list[Any] = []
        self.input_changed = False
        self.split_evaluation = False
        self.notification_count = 0
        self.next_calculation = float('-inf')
        self.state_updates: list[dict[str, Any]] = []
        self.calculation_rows: list[dict[str, Any]] = []

    def calculation_due(self, notice: RecognitionNotice) -> bool:
        """状態更新は全件計装し、FIFOの最新状態かつ公開周期に達した時だけ計算する。"""
        now = time.perf_counter()
        self.notification_count += 1
        depth = self.queue.qsize()+self.batch_remaining
        self.state_updates.append(dict(frame=notice.frame, t_sec=notice.t_sec,
            queue_depth=depth, milliseconds=(now-self.state_started)*MILLISECONDS))
        clock = now if self.realtime else notice.t_sec
        latest = getattr(self, 'latest_frame', None)
        behind = self.realtime and (self.batch_remaining > 0 or
                                   (latest is not None and latest.value > notice.frame))
        if behind or clock < self.next_calculation:
            return False
        self.calculation_started = now
        self.skip_features = False
        return True

    def calculation_finished(self, notice: RecognitionNotice) -> None:
        now = time.perf_counter()
        self.calculation_rows.append(dict(frame=notice.frame, t_sec=notice.t_sec,
            notifications=self.notification_count, started_at=self.calculation_started,
            milliseconds=(now-self.calculation_started)*MILLISECONDS))
        clock = now if self.realtime else notice.t_sec
        self.next_calculation = clock+BATCH_PERIOD_SEC

    def consume_input_boundary(self) -> bool:
        """機器切替後の最初の公開盤面だけ、既存の試合リセット経路を通す。"""
        changed, self.input_changed = self.input_changed, False
        return changed

    def counter_context(self, result: Any, game: int) -> None:
        """非STABLE通知を含めて確定盤面世代と試合境界を追跡する。"""
        context = (game, getattr(result, 'is_match_active', True), *(
            None if side.confirmed_board is None else side.confirmed_board.grid_bytes()
            for side in (result.p1, result.p2)))
        for counter in self.counters:
            counter.invalidate(context)

    def _put(self, notice: RecognitionNotice) -> None:
        while not self.cancel.is_set():
            try:
                self.queue.put(notice, timeout=POLL_SEC)
                self.queue_max = max(self.queue_max, self.queue.qsize())
                self.pending_max = max(self.pending_max, self.queue.qsize()+self.batch_remaining)
                return
            except Full:
                continue

    def _recognize(self, pipe: Any) -> None:
        try:
            for frame in self.source:
                if self.cancel.is_set():
                    break
                notice = recognize(pipe, frame)
                self.recognition_rows.append(dict(frame=notice.frame, t_sec=notice.t_sec,
                    captured_at=notice.captured_at, recognized_at=notice.recognized_at,
                    dropped_before=notice.dropped_before))
                self._put(notice)
        except BaseException as error:
            self.error = error
        finally:
            self.done.set()

    def observations(self, pipe: Any, capture: Any, fps: float,
                     start: int, end: int, stride: int) -> Iterator[RecognitionNotice]:
        """最大2Hzでバッチを開始し、バッチ内はframe順に全通知を適用する。"""
        self.source = self.source_factory(capture, fps, start, end, stride, self.realtime)
        self.frame_bounds = (fps, start, end, stride)
        self.worker = Thread(target=self._recognize, args=(pipe,), name='recognition', daemon=True)
        self.worker.start()
        next_batch = time.perf_counter() + BATCH_PERIOD_SEC
        while not self.done.is_set() or not self.queue.empty():
            if self.cancel.wait(max(0.0, next_batch - time.perf_counter())):
                break
            started = time.perf_counter()
            batch = self._drain()
            if batch:
                self.batch_starts.append(started)
                self.batch_sizes.append(len(batch))
                for index, notice in enumerate(batch):
                    self.batch_remaining = len(batch)-index-1
                    self.batch_ready_at = started
                    yield from self.timed_notice(notice)
            next_batch = started + BATCH_PERIOD_SEC
        if self.error is not None:
            raise RuntimeError('認識workerが失敗しました') from self.error

    def timed_notice(self, notice: RecognitionNotice) -> Iterator[RecognitionNotice]:
        """CPU時間と壁時計の差はGIL/OS待ちの上限であり、GILだけとは断定しない。"""
        started, cpu = time.perf_counter(), time.thread_time()
        self.state_started = started
        if self.meter is None:
            yield notice
        else:
            with self.meter.frame(notice.frame, notice.t_sec):
                self.meter.current.update(legacy_features=0.0, review=0.0, counter=0.0)
                yield notice
        finished = time.perf_counter()
        self.profile.append(dict(frame=notice.frame, recognized_at=notice.recognized_at,
            started_at=started, finished_at=finished,
            batch_wait_sec=max(0.0, self.batch_ready_at-notice.recognized_at),
            within_batch_wait_sec=started-max(self.batch_ready_at, notice.recognized_at),
            evaluation_wall_sec=finished-started, evaluation_cpu_sec=time.thread_time()-cpu))

    def _drain(self) -> list[RecognitionNotice]:
        batch = []
        for _ in range(self.queue.qsize()):
            try:
                batch.append(self.queue.get_nowait())
            except Empty:
                break
        return batch

    def observe(self, notice: RecognitionNotice, probability: float, advantage: float,
                overlay: Any, result: Any, game: int, write_frame: int,
                counter_trackers: tuple[Any, ...] = ()) -> None:
        self.active_counters = [counter for counter in counter_trackers if hasattr(counter, 'status')]
        if self.split_evaluation:
            self.calculation_finished(notice)
        if notice.frame >= write_frame:
            self.callback(notice, probability, advantage, overlay, result, game,
                          self.queue.qsize()+self.batch_remaining, time.perf_counter())

    def mc_in_display(self, overlay: Any, counter: Any, resolved: Any, active: bool) -> bool:
        """学習済み勝率で上書きされたMCを、表示への寄与と誤記しない。"""
        import math
        if overlay is not None and overlay.tracker.probability is not None:
            self.mc_included = False
        elif active:
            self.mc_included = math.isfinite(getattr(resolved, 'hold_defender_prob', float('nan')))
        else:
            result = getattr(counter, '_last_result', None)
            self.mc_included = bool(result and counter.last_budget_sec > 0 and
                getattr(counter, 'contributes', all(math.isfinite(p) for p in result[1:])))
        return self.mc_included

    def close(self) -> None:
        self.cancel.set()
        if self.worker is not None:
            self.worker.join(JOIN_SEC)
            if self.worker.is_alive():
                raise RuntimeError('認識workerが停止しません')


class _NotificationCalls(ast.NodeTransformer):
    """評価側から生pipelineや画像を参照する呼出を通知値へ置き換える。"""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        aliases = {'formula_totals_from_pipeline': 'formula_totals',
                   'displayed_scores_from_pipeline': 'displayed_scores',
                   'formula_visible_from_pipeline': 'formula_visible'}
        if isinstance(node.func, ast.Name) and node.func.id in aliases:
            return ast.copy_location(ast.parse('pipe.' + aliases[node.func.id]).body[0].value, node)
        return self.generic_visit(node)


def adapt_loop(loop: ast.For) -> None:
    """認識境界より後ろの評価コードは元の順序を一切変えない。"""
    codes = [ast.unparse(node) for node in loop.body]
    recognition = next(i for i, code in enumerate(codes) if code.startswith('r = pipe.update('))
    drawing = next(i for i, code in enumerate(codes) if code.startswith('waiting ='))
    body = loop.body[recognition + 1:drawing]
    reset = next(i for i, node in enumerate(body) if ast.unparse(node).startswith('if _formal_boundary:'))
    body.insert(reset, ast.parse('_formal_boundary = _live_bridge.consume_input_boundary() '
                                'or _formal_boundary').body[0])
    boundary = next(i for i, node in enumerate(body)
                    if ast.unparse(node).startswith('snap = _drive_ojama'))
    body.insert(boundary, ast.parse('_live_bridge.counter_context(r, game_idx)').body[0])
    callback = ast.parse('_live_bridge.observe(packet, disp_p1, disp_adv, event_overlay, '
                         'r, game_idx, write_frame, '
                         '(counter_tracker, resolved_tracker._counter_tracker))').body[0]
    index = next(i for i, node in enumerate(body) if ast.unparse(node).startswith('if fi < write_frame:'))
    body.insert(index, callback)
    body.insert(index, ast.parse('_live_bridge.mc_in_display(event_overlay, counter_tracker, '
                                 'resolved_tracker, resolved_active)').body[0])
    prefix = ast.parse('fi = packet.frame\nt = packet.t_sec\nr = packet.result()\npipe = packet.pipeline').body
    loop.target = ast.Name(id='packet', ctx=ast.Store())
    loop.iter = ast.parse('_live_bridge.observations(pipe, cap, fps, start_frame, n, stride)').body[0].value
    loop.body = prefix + body
    _NotificationCalls().visit(loop)


def build_live_generate(module: Any, bridge: RecognitionBridge) -> Callable[..., int]:
    """新経路に限り既存generateを接続。旧CLI・既存ファイルには変更なし。"""
    tree = ast.parse(inspect.getsource(module.generate))
    function = tree.body[0]
    loops = [node for node in function.body if isinstance(node, ast.For)
             and ast.unparse(node.target) == 'fi']
    if len(loops) != 1:
        raise ValueError('旧評価ループの構造が変更されています')
    adapt_loop(loops[0])
    if bridge.split_evaluation:
        split_loop(loops[0])
    if hasattr(bridge, 'pipeline_config'):
        for node in ast.walk(function):
            if isinstance(node, ast.If) and ast.unparse(node.test) == 'review_enabled':
                node.test = ast.parse('review_enabled and not _live_bridge.skip_features').body[0].value
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func) == 'RecognitionPipeline.load_default'):
                node.func = ast.parse('_live_bridge.pipeline_config').body[0].value
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func) == 'cv2.VideoCapture'):
                node.func = ast.parse('_live_bridge.open_capture').body[0].value
    namespace = dict(vars(module), _live_bridge=bridge)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(module), 'exec'), namespace)
    return namespace['generate']


def split_loop(loop: ast.For) -> None:
    """会計・状態機械の後で公開周期を判定し、旧数値合成部は公開時だけ通す。"""
    body = loop.body
    index = next(i for i, node in enumerate(body)
                 if ast.unparse(node).startswith('settled_ran_this_frame ='))
    # 得点/手数の配送予告は差分を持つので、旧settled条件のまま全通知へ移す。
    fc = next(node for node in ast.walk(loop) if isinstance(node, ast.Assign)
              and ast.unparse(node).startswith('fc = fctracker.update('))
    update = ast.parse('if b1 is not None and b2 is not None and settled:\n    pass').body[0]
    update.body = [ast.parse(ast.unparse(fc)).body[0]]
    fc.value = ast.Name(id='fc', ctx=ast.Load())
    checkpoint = ast.parse(
        'if event_overlay is None:\n    raise ValueError("B7にはイベント評価が必要です")\n'
        'if not _live_bridge.calculation_due(packet):\n    continue\n'
        'event_overlay.calculate()').body
    body[index:index] = [update, *checkpoint]


def notice_digest(notice: RecognitionNotice) -> str:
    """認識通知全体の同一性を公開DTOへ記録する。"""
    return 'sha256:' + hashlib.sha256(notice.result_bytes + repr(notice.pipeline).encode()).hexdigest()
