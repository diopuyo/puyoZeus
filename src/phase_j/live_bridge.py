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

from src.production_config import FORCED_BOUNDARY_DEBOUNCE_ISOLATION_DEFAULT
from .live_source import CapturedFrame, FrameSource, VideoFileSource
from .live_boundary import RecoveryBoundary
from . import live_stage_timing as timing

BATCH_PERIOD_SEC = 0.5
QUEUE_CAPACITY = 30
POLL_SEC = 0.05
JOIN_SEC = 10.0
MILLISECONDS = 1000.0
RECOGNITION_AUDIT_DEFAULTS = dict(next_recolor_guard_counts={}, next_recolor_guard_log=[],
                                verification_pending_expired_count=0)
# 試合境界は得点リセット (R1) と入力世代切替の強制境界 (R2) の2経路。評価ループは境界のたびに
# _last_formal_boundary_t (R1用の5秒デバウンス枠) を更新するため、R2が枠を消費すると直後の
# 本物のR1が拒否される。R2単独の境界では枠を境界前の値へ戻す (同一フレームのR1は枠を更新する)。
LEGACY_FORCED_BOUNDARY_BEFORE = ('_formal_boundary = _live_bridge.consume_input_boundary() '
                                 'or _formal_boundary')
ISOLATED_FORCED_BOUNDARY_BEFORE = (
    '_scored_boundary = _formal_boundary\n'
    '_forced_boundary = _live_bridge.consume_input_boundary()\n'
    '_debounce_before_forced = _last_formal_boundary_t\n'
    '_formal_boundary = _forced_boundary or _formal_boundary')
ISOLATED_FORCED_BOUNDARY_AFTER = (
    'if _forced_boundary and not _scored_boundary:\n'
    '    _last_formal_boundary_t = _debounce_before_forced')


def recognition_guard_audits(pipe: Any, recognition_frames: int | None = None) -> dict[str, Any]:
    """認識が終了した時点の監査を、設定辞書や最終通知と分離して受け渡す。"""
    target = getattr(pipe, 'pipe', pipe)
    return dict({name: getattr(target, name, default) for name, default in RECOGNITION_AUDIT_DEFAULTS.items()},
                summary_received=True, recognition_frames=recognition_frames)


def finish_recording(local: dict[str, Any]) -> None:
    """旧評価を計算しなかった通知は、fallback欠測の更新行として確定する。"""
    recorder = local.get('event_recorder')
    if recorder is not None and recorder.pending_update is not None:
        recorder.write(recorder.pending_update)
        recorder.pending_update = None


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
    b3: tuple | None = None

    def result(self) -> Any:
        # 自プロセスが生成したbytesだけを読む。外部pickle入力は受け付けない。
        return pickle.loads(self.result_bytes)

    def evaluation_result(self) -> Any:
        """評価専用。公開ガードの補正を履歴・死亡判定へ混入させない。"""
        from src.row1_publication import evaluation_result
        return evaluation_result(self.result())


def recognize(pipe: Any, frame: CapturedFrame) -> RecognitionNotice:
    """OCRの実読と段累積も認識所有スレッド内で同時点に確定する。"""
    from src.exchange_event_m0 import formula_totals_from_pipeline
    from src.exchange_event_overlay import formula_visible_from_pipeline
    from .live_score_share import displayed_scores
    from .live_snapshot import enrich_recognition, prepare_recognition
    prepare_recognition(pipe)
    timing.identify(frame, getattr(pipe, 'epoch', None))
    from .live_score_share import prepare
    prepare(getattr(pipe, 'pipe', pipe))
    import os
    if os.environ.get('PUYO_B3_WARNING_PREFETCH') == '1':
        # 試作 (既定OFF): 予告アイコン読取を盤面認識と重ねる。判定の関数・画素は同期読取と同じ。
        from .live_b3 import prefetch_warning
        timing.measure('b3_warning_prefetch', prefetch_warning, pipe, frame)
    result = timing.measure('board_recognition', pipe.update, frame.index, frame.media_sec, frame.image)
    from src.row1_publication import evaluation_result, publication_result
    result = publication_result(timing.measure('recognition_enrich', enrich_recognition,
        pipe, frame.image, evaluation_result(result), frame.media_sec), result)
    view = PipelineView(tuple(pipe.tsumo_count(side) for side in ('1P', '2P')),
                        formula_totals_from_pipeline(pipe),
                        displayed_scores(pipe, frame),
                        formula_visible_from_pipeline(pipe))
    frozen = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    import os
    b3 = None
    if os.environ.get('PUYO_B3_JUDGMENT') == '1':
        from .live_b3 import capture
        b3 = timing.measure('b3_capture_total', capture, pipe, result, frame, view.counts)
    return RecognitionNotice(frame.index, frame.media_sec, frame.captured_at,
                             time.perf_counter(), frozen, view, frame.dropped_before, b3)


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
        from .live_memory import MemoryProbe
        self.memory_probe = MemoryProbe('evaluation')
        from .live_degrade import EvaluationPacer
        self.pacer = EvaluationPacer(realtime)
        self.calculation_period = BATCH_PERIOD_SEC
        self.recognition_audits: dict[str, Any] = {}
        self.recognition_summary_received = False
        self.recovery_boundary = RecoveryBoundary()
        import os
        from .live_match_number import PublicMatchNumber
        self.public_matches = (PublicMatchNumber()
            if os.environ.get('PUYO_LIVE_PUBLIC_MATCH_NUMBER') == '1' else None)
        if os.environ.get('PUYO_OBS_PREPARE_EXACT') == '1':
            from .live_obs_prepare import prepare_imports
            prepare_imports('evaluation')

    def boundary_increment(self, forced: bool, scored: bool, now: float) -> int:
        """境界の初期化処理は毎回通し、入力復帰後の番号重複だけを除く。"""
        return self.recovery_boundary.increment(forced, scored, now,
            getattr(self, 'input_boundary_recovery', True))

    def record_terminal_boundary(self, game: int, forced: bool, scored: bool) -> None:
        """表示専用の境界理由を保存し、判定用の番号・状態は変更しない。"""
        recovery = (forced and getattr(self, 'input_boundary_recovery', False)
                    and getattr(self, 'input_failure_epoch', None) == getattr(self, 'input_epoch', -1))
        self.terminal_boundary = (game, 'scored' if scored else
                                  'input_recovery' if recovery else 'unknown')

    def record_public_boundary(self, game: int, forced: bool, scored: bool) -> None:
        """内部境界の処理後に公開番号だけ記録し、評価行の既存メタデータも維持する。"""
        self.public_matches.boundary(game, forced, scored,
            getattr(self, 'input_boundary_recovery', False))

    @timing.timed('memory_sample')
    def memory_sample(self, local: dict[str, Any], t_sec: float) -> None:
        if getattr(self, 'b3', None) is not None:
            timing.measure('b3_dispatch', self.b3.update, self.current_notice, local['event_overlay'])
            self.b3_context = (local['event_overlay'], local['r'], local['game_idx'])
            if getattr(self.b3, 'completion_driven', False):
                timing.measure('b3_completion_publish', self.on_b3_completed)
        from .live_memory import evaluation_roots
        if hasattr(self, 'spool_directory') and local.get('game_idx') != getattr(self, '_retention_game', None):
            self._retention_game = local.get('game_idx')
            recorder = local.get('event_recorder')
            if recorder is not None:
                recorder.static_keys.clear()
        if self.memory_probe.path:
            self.memory_probe.sample(t_sec, evaluation_roots(local, self))

    def calculation_due(self, notice: RecognitionNotice) -> bool:
        """状態更新は全件計装し、FIFOの最新状態かつ公開周期に達した時だけ計算する。"""
        now = time.perf_counter()
        self.notification_count += 1
        depth = self.queue.qsize()+self.batch_remaining
        self.state_updates.append(dict(frame=notice.frame, t_sec=notice.t_sec,
            queue_depth=depth, milliseconds=(now-self.state_started)*MILLISECONDS))
        clock = now if self.realtime else notice.t_sec
        if self.pacer.enabled:
            self.calculation_period = self.pacer.period(now, now-notice.captured_at)
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
            period_sec=self.calculation_period,
            milliseconds=(now-self.calculation_started)*MILLISECONDS))
        clock = now if self.realtime else notice.t_sec
        self.next_calculation = clock+self.calculation_period

    def consume_input_boundary(self) -> bool:
        """機器切替後の最初の公開盤面だけ、既存の試合リセット経路を通す。"""
        changed, self.input_changed = self.input_changed, False
        return changed

    def counter_context(self, result: Any, game: int, t_sec: float | None = None,
                        captured_at: float | None = None,
                        counter_trackers: tuple | None = None) -> None:
        """非STABLE通知を含めて確定盤面世代と試合境界を追跡する。"""
        context = (game, getattr(result, 'is_match_active', True), *(
            None if side.confirmed_board is None else side.confirmed_board.grid_bytes()
            for side in (result.p1, result.p2)))
        if t_sec is not None:
            self.game_starts = getattr(self, 'game_starts', [])
            if context[1] and getattr(self, 'previous_game_context', None) != context[:2]:
                self.game_starts.append(dict(game=game, t_sec=t_sec, captured_at=captured_at))
            self.previous_game_context = context[:2]
        from .live_spool import bounded_enabled
        counters = counter_trackers if counter_trackers is not None and bounded_enabled() else self.counters
        for counter in counters:
            if not hasattr(counter, 'invalidate'):
                continue
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
        count = 0
        try:
            for frame in self.source:
                if self.cancel.is_set():
                    break
                notice = recognize(pipe, frame)
                count += 1
                self.recognition_rows.append(dict(frame=notice.frame, t_sec=notice.t_sec,
                    captured_at=notice.captured_at, recognized_at=notice.recognized_at,
                    dropped_before=notice.dropped_before))
                self._put(notice)
        except BaseException as error:
            self.error = error
        finally:
            if getattr(getattr(pipe, 'pipe', pipe), '_enable_match_end_pair', False):
                from src.async_match_end_pair import close_pair_worker
                close_pair_worker(pipe)
            self.recognition_audits = recognition_guard_audits(pipe, count)
            self.recognition_summary_received = True
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
        self.current_notice = notice
        timing.begin('evaluation', notice.captured_at, notice.frame, notice.t_sec)
        started, cpu = time.perf_counter(), time.thread_time()
        self.state_started = started
        b3 = getattr(self, 'b3', None)
        if b3 is not None:
            b3.submit_ex_early(notice)
        from .live_death_retr import capture
        capture(b3, notice)
        if self.meter is None:
            yield notice
        else:
            with self.meter.frame(notice.frame, notice.t_sec):
                self.meter.current.update(legacy_features=0.0, review=0.0, counter=0.0)
                yield notice
        finished = time.perf_counter()
        timing.detail(recognized_at=notice.recognized_at)
        timing.finish(evaluation_wall_ms=(finished-started)*MILLISECONDS,
                      evaluation_cpu_ms=(time.thread_time()-cpu)*MILLISECONDS)
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

    @timing.timed('display_assemble')
    def observe(self, notice: RecognitionNotice, probability: float, advantage: float,
                overlay: Any, result: Any, game: int, write_frame: int,
                counter_trackers: tuple[Any, ...] = ()) -> None:
        self.active_counters = [counter for counter in counter_trackers if hasattr(counter, 'status')]
        if self.split_evaluation:
            self.calculation_finished(notice)
        if notice.frame >= write_frame:
            # 評価が終わった後、描画へは補正済みの公開セルを渡す。
            from src.row1_publication import publication_result
            result = publication_result(result, notice.result())
            self.callback(notice, probability, advantage, overlay, result, game,
                          self.queue.qsize()+self.batch_remaining, time.perf_counter())
        if hasattr(self, 'event_evaluator'):
            self.event_evaluator.succeeded()

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

    def visit_If(self, node: ast.If) -> ast.AST | None:
        # 評価側に画像はない。通知にある画像由来情報を保持し、未取得は欠測のまま扱う。
        if ast.unparse(node.test) in ('terminal_detector is not None',
                                     'midchain_reader is not None', 'snapshot_reader is not None'):
            return None
        return self.generic_visit(node)


def forced_boundary_statements(isolate: bool, terminal_recovery: bool = False) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """強制境界の注入文を (境界処理の前, 後) で返す。OFFは従来の1文だけで、後は空。"""
    before = ISOLATED_FORCED_BOUNDARY_BEFORE if isolate else LEGACY_FORCED_BOUNDARY_BEFORE
    after = ast.parse(ISOLATED_FORCED_BOUNDARY_AFTER).body if isolate else []
    if terminal_recovery:
        if not isolate:
            # 表示のための計装だけ追加し、旧構成のデバウンス挙動も変えない。
            before = ('_scored_boundary = _formal_boundary\n'
                      '_forced_boundary = _live_bridge.consume_input_boundary()\n'
                      '_formal_boundary = _forced_boundary or _scored_boundary')
        after += ast.parse('if _formal_boundary:\n'
            '    _live_bridge.record_terminal_boundary(game_idx, _forced_boundary, _scored_boundary)').body
    return ast.parse(before).body, after


def adapt_loop(loop: ast.For,
               isolate_forced_boundary: bool = FORCED_BOUNDARY_DEBOUNCE_ISOLATION_DEFAULT,
               coalesce_recovery: bool = False, public_match_number: bool = False) -> None:
    """認識境界より後ろの評価コードは元の順序を一切変えない。"""
    codes = [ast.unparse(node) for node in loop.body]
    recognition = next(i for i, code in enumerate(codes) if code.startswith('r = pipe.update('))
    drawing = next(i for i, code in enumerate(codes) if code.startswith('waiting ='))
    body = loop.body[recognition + 1:drawing]
    reset = next(i for i, node in enumerate(body) if ast.unparse(node).startswith('if _formal_boundary:'))
    import os
    terminal_recovery = os.environ.get('PUYO_TERMINAL_HOLD_INPUT_RECOVERY') == '1'
    before, after = forced_boundary_statements(isolate_forced_boundary or coalesce_recovery, terminal_recovery)
    if public_match_number:
        if not (isolate_forced_boundary or coalesce_recovery or terminal_recovery):
            before = ast.parse('_scored_boundary = _formal_boundary\n'
                '_forced_boundary = _live_bridge.consume_input_boundary()\n'
                '_formal_boundary = _forced_boundary or _scored_boundary').body
        after += ast.parse('if _formal_boundary:\n'
            '    _live_bridge.record_public_boundary(game_idx, _forced_boundary, _scored_boundary)').body
    if coalesce_recovery:
        coalesce_boundary_block(body[reset])
    body[reset + 1:reset + 1] = after  # 既存の境界処理 (if _formal_boundary:) の直後
    body[reset:reset] = before
    boundary = next(i for i, node in enumerate(body)
                    if ast.unparse(node).startswith('snap = _drive_ojama'))
    body.insert(boundary, ast.parse('_live_bridge.counter_context(r, game_idx, t, packet.captured_at, '
                                    '(counter_tracker, resolved_tracker._counter_tracker))').body[0])
    callback = ast.parse('_live_bridge.observe(packet, disp_p1, disp_adv, event_overlay, '
                         'r, game_idx, write_frame, '
                         '(counter_tracker, resolved_tracker._counter_tracker))').body[0]
    index = next(i for i, node in enumerate(body) if ast.unparse(node).startswith('if fi < write_frame:'))
    body.insert(index, callback)
    body.insert(index, ast.parse('_live_bridge.mc_in_display(event_overlay, counter_tracker, '
                                 'resolved_tracker, resolved_active)').body[0])
    prefix = ast.parse('fi = packet.frame\nt = packet.t_sec\nr = packet.evaluation_result()\npipe = packet.pipeline').body
    loop.target = ast.Name(id='packet', ctx=ast.Store())
    loop.iter = ast.parse('_live_bridge.observations(pipe, cap, fps, start_frame, n, stride)').body[0].value
    loop.body = prefix + body
    _NotificationCalls().visit(loop)


def coalesce_boundary_block(block: ast.If) -> None:
    """死亡reset・会計resetは元のまま、番号加算の右辺だけを差し替える。"""
    increments = [node for node in block.body if isinstance(node, ast.AugAssign)
                  and ast.unparse(node) == 'game_idx += 1']
    if len(increments) != 1:
        raise ValueError('試合番号加算の構造が変更されています')
    increments[0].value = ast.parse(
        '_live_bridge.boundary_increment(_forced_boundary, _scored_boundary, t)').body[0].value


def build_live_generate(module: Any, bridge: RecognitionBridge) -> Callable[..., int]:
    """新経路に限り既存generateを接続。旧CLI・既存ファイルには変更なし。"""
    tree = ast.parse(inspect.getsource(module.generate))
    function = tree.body[0]
    # 画像由来の観測は認識通知で取得済み。評価processには読取器を生成しない。
    readers = {'terminal_detector', 'midchain_reader', 'snapshot_reader'}
    for node in function.body:
        if isinstance(node, ast.If) and any(isinstance(child, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in readers for target in child.targets)
                for child in node.body):
            node.test = ast.Constant(False)
    loops = [node for node in function.body if isinstance(node, ast.For)
             and ast.unparse(node.target) == 'fi']
    if len(loops) != 1:
        raise ValueError('旧評価ループの構造が変更されています')
    boundary_options = {'coalesce_recovery': True} if getattr(
        bridge, 'recovery_boundary_coalescing', False) else {}
    if getattr(bridge, 'public_matches', None) is not None:
        if boundary_options:
            raise ValueError('公開番号の分離と旧境界統合は併用できません')
        boundary_options['public_match_number'] = True
    adapt_loop(loops[0], getattr(bridge, 'forced_boundary_debounce_isolation',
                                 FORCED_BOUNDARY_DEBOUNCE_ISOLATION_DEFAULT), **boundary_options)
    if bridge.split_evaluation:
        split_loop(loops[0])
    from .live_eval_supervisor import guard_loop, evaluation_failure, NOTICE_PREFIX_SIZE
    if getattr(bridge, 'isolate_evaluation', False):
        guard_loop(loops[0])
    finalize = ast.parse('_finish_recording(locals())').body
    loops[0].body[NOTICE_PREFIX_SIZE:] = [ast.Try(body=loops[0].body[NOTICE_PREFIX_SIZE:],
        handlers=[], orelse=[], finalbody=finalize)]
    _RecognitionAudits().visit(function)
    from .live_retention import TIMELINE_ROWS, timeline_spool
    for node in function.body:
        if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) in TIMELINE_ROWS:
            node.value = ast.parse(f'_timeline_spool(_live_bridge, {ast.unparse(node.target)!r})').body[0].value
    if hasattr(bridge, 'pipeline_config'):
        adapt_process_calls(function)
    namespace = dict(vars(module), _live_bridge=bridge, _timeline_spool=timeline_spool,
                     _evaluation_failure=evaluation_failure, _finish_recording=finish_recording)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(module), 'exec'), namespace)
    return namespace['generate']


class _RecognitionAudits(ast.NodeTransformer):
    """終了時の監査出力は認識所有者が返した値を参照する。"""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # 二つの既存監査JSONに受信状態と認識母数を併記する。
        names = {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
                 and isinstance(child.value, ast.Name) and child.value.id == 'pipe'}
        if isinstance(node.func, ast.Name) and node.func.id == 'dict' and names & RECOGNITION_AUDIT_DEFAULTS.keys():
            for key, code in {
                'summary_received': '_live_bridge.recognition_summary_received',
                'recognition_frames': "_live_bridge.recognition_audits.get('recognition_frames')",
            }.items():
                node.keywords.append(ast.keyword(arg=key, value=ast.parse(code).body[0].value))
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == 'pipe' and node.attr in RECOGNITION_AUDIT_DEFAULTS:
            default = RECOGNITION_AUDIT_DEFAULTS[node.attr]
            code = f'_live_bridge.recognition_audits.get({node.attr!r}, {default!r})'
            return ast.copy_location(ast.parse(code).body[0].value, node)
        return self.generic_visit(node)


def adapt_process_calls(function: ast.FunctionDef) -> None:
    """認識器の構築と監査出力先を、子プロセスへの設定配送に置換する。"""
    calls = {'RecognitionPipeline.load_default': '_live_bridge.pipeline_config',
             'cv2.VideoCapture': '_live_bridge.open_capture',
             'pipe._placement_reconcile.log_to': '_live_bridge.placement_log'}
    for node in ast.walk(function):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'review_enabled':
            node.test = ast.parse('review_enabled and not _live_bridge.skip_features').body[0].value
        if isinstance(node, ast.Call) and (replacement := calls.get(ast.unparse(node.func))):
            node.func = ast.parse(replacement).body[0].value


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
        '_live_bridge.memory_sample(locals(), t)\n'
        'if event_overlay is None:\n    raise ValueError("B7にはイベント評価が必要です")\n'
        'if not _live_bridge.calculation_due(packet):\n    continue\n'
        'event_overlay.calculate()').body
    body[index:index] = [update, *checkpoint]


def notice_digest(notice: RecognitionNotice) -> str:
    """認識通知全体の同一性を公開DTOへ記録する。"""
    return 'sha256:' + hashlib.sha256(notice.result_bytes + repr(notice.pipeline).encode()).hexdigest()
