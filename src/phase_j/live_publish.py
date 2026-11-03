"""B2専用のPhase J DTO生成と既存SSEサーバの接続。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Any
from uuid import uuid4
import os

from src.stream_overlay import StreamOverlayServer, StreamState
from .contracts import OverlaySnapshot
from .display_projector import PublicationContext, project_snapshot
from .reducer import Bootstrap, initial_state
from .snapshot_hub import SnapshotHub
from .overlay_config import validate_overlay, with_debug_defaults
from .overlay_http import OverlayPagesMixin
from .live_freshness import apply_freshness, evaluation_stale
from .live_terminal_hold import TerminalDisplayHold
from .live_match_number import PublicMatchNumber
from .live_instance_guard import exclusive_bind_default
from .live_debug_panel import with_debug_panel
from src.production_config import EVALUATION_FRESHNESS_GATE_DEFAULT, TERMINAL_DISPLAY_HOLD_DEFAULT

PUBLISH_PERIOD_SEC = 0.5
MILLISECONDS = 1000.0
EVEN_THRESHOLD = 3.0
HTTP_OK = 200
PUBLISHER_JOIN_PERIODS = 4
PROJECTED_SOURCES = frozenset({'S3_landing', 'unavoidable_death'})
HTML = Path(__file__).with_name('overlay.html').read_text(encoding='utf-8')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initial_snapshot(assets: dict[str, str]) -> OverlaySnapshot:
    state = initial_state(Bootstrap('live-' + uuid4().hex, assets=assets))
    return project_snapshot(state, PublicationContext(0, utc_now(), int(time.perf_counter() * MILLISECONDS)))


def result_snapshot(initial: OverlaySnapshot, row: dict[str, Any], now: float,
                    revision: int, hold_started: float | None,
                    freshness_gate: bool = EVALUATION_FRESHNESS_GATE_DEFAULT) -> OverlaySnapshot:
    """既存DTOにsourceと単調時刻の観測列だけを追加する。"""
    payload = initial.to_mapping()
    if row.get('v49f') is not None:
        payload['evaluations']['v49f'] = row['v49f']
    for key in ('ama_plain', 'b3_model_label', 'x_provisional', 'live24h'):
        if row.get(key) is not None:
            payload['evaluations'][key] = row[key]
    payload['identity'].update(reducer_revision=revision, match_id=str(row.get('public_game', row['game'])),
        match_state_seq=row['frame'], input_generation=row['frame'])
    payload['timing'].update(source_available_frame=row['frame'],
        source_available_ms=int(row['t_sec'] * MILLISECONDS), published_at_utc=utc_now(),
        capture_to_publish_latency_ms=(now-row['captured_at'])*MILLISECONDS,
        calculation_age_ms=(now-row['evaluated_at'])*MILLISECONDS,
        last_confirmed_age_ms=None, capture_monotonic_sec=row['captured_at'],
        recognized_monotonic_sec=row['recognized_at'], evaluated_monotonic_sec=row['evaluated_at'])
    payload['runtime'].update(status='healthy', practical_worker_health='healthy',
        practical_queue_depth=int(row['queue_depth'] > 0),
        recognition_notification_queue_depth=row['queue_depth'], best_action_worker_health='disabled')
    payload['input'].update(event_seq=row['frame'])
    calibration = row.get('input_calibration')
    if calibration and 'score_roi_alignment' in calibration:
        payload['display']['score_roi_alignment'] = calibration['score_roi_alignment']
    if row.get('_terminal_display_held'):
        fill_terminal(payload, row, now)
        return OverlaySnapshot.from_mapping(payload)
    if row.get('evaluation_error'):
        evaluation_hold(payload, row, now, hold_started)
        return OverlaySnapshot.from_mapping(payload)
    if (calibration and calibration['phase'] != 'ready') or row.get('input_verifying'):
        payload['timing']['calculation_age_ms'] = None
        input_hold(payload, row, now, hold_started)
        return OverlaySnapshot.from_mapping(payload)
    if calibration:
        payload['display'].update(input_status='ready', calibration_progress=100)
    if not row.get('match_active', True):
        payload['timing']['calculation_age_ms'] = None
        payload['display'].update(message='次試合待ち', match_phase='between_matches')
        return OverlaySnapshot.from_mapping(payload)
    payload['display']['match_phase'] = 'playing'
    if 'display_layers' in row:
        payload['evaluations']['display_layers'] = row['display_layers']
    if 'counter_search' in row:
        payload['evaluations']['counter_search'] = row['counter_search']
    payload['evaluations']['prediction_sources'] = row.get('prediction_sources', [])
    if row['raw_probability'] is None:
        payload['timing']['calculation_age_ms'] = None
        return OverlaySnapshot.from_mapping(payload)
    if row.get('b3_state_frame') is not None and row['b3_state_frame'] < row['frame']:
        hold_started = now if hold_started is None else hold_started
    fill_evaluation(payload, row, now, hold_started)
    fill_b3_timing(payload, row)
    if freshness_gate:
        apply_freshness(payload, row, now)
    return OverlaySnapshot.from_mapping(payload)


def fill_terminal(payload: dict[str, Any], row: dict[str, Any], now: float) -> None:
    """既存の片側死亡sourceを表示する。両側ロゴ確認の別契約へ偽装しない。"""
    fill_evaluation(payload, row, now, None)
    winner = '1P' if row['probability'] == 1.0 else '2P'
    payload['display'].update(match_phase='playing', message=f'決着・{winner}勝利')


def fill_b3_timing(payload: dict[str, Any], row: dict[str, Any]) -> None:
    """B3の基準映像時刻と、公開時の単調時計による鮮度は別々に保持する。"""
    if row.get('b3_state_t_sec') is None:
        return
    stamp = row['b3_state_t_sec']
    payload['display']['message'] = f'B3 基準 {stamp:.3f}秒'
    payload['timing']['calculation_age_ms'] = (row['t_sec']-stamp)*MILLISECONDS
    payload['evaluations']['practical'].update(request_id=f"b3-frame-{row['b3_state_frame']}",
        input_generation=row['b3_state_frame'], calculation_age_ms=(row['t_sec']-stamp)*MILLISECONDS)


def fill_evaluation(payload: dict[str, Any], row: dict[str, Any], now: float,
                    hold_started: float | None) -> None:
    """入力が利用可能な場合だけ評価表示を埋める。"""
    projected = row['source'] in PROJECTED_SOURCES
    hold = hold_started is not None
    payload['display'].update(visibility='visible', status='hold' if hold else
        ('physical_prediction' if projected else 'live'),
        update_reason='hold_started' if hold else 'prediction_committed',
        primary_hold_reason='calculation_pending' if hold else None,
        all_hold_reasons=['calculation_pending'] if hold else [],
        hold_started_ms=int((row['t_sec'] if row.get('hold_media_sec') is None
                             else row['hold_media_sec'])*MILLISECONDS) if hold else None,
        hold_elapsed_ms=int((now-hold_started)*MILLISECONDS) if hold else None)
    payload['input'].update(p1_board_provenance='physics_projected' if projected else 'confirmed',
        p2_board_provenance='physics_projected' if projected else 'confirmed',
        physical_prediction_used=projected,
        recognition_quality={'status': 'partial' if row['capture_gap'] else 'trusted',
                             'reason_codes': ['capture_gap'] if row['capture_gap'] else []})
    payload['evaluations']['practical'].update(availability='available',
        request_id=f"frame-{row['frame']}", input_generation=row['frame'], input_digest=row['digest'],
        calculation_latency_ms=(row['evaluated_at']-row['recognized_at'])*MILLISECONDS,
        calculation_age_ms=(now-row['evaluated_at'])*MILLISECONDS,
        p1_win_probability=row['probability'], p2_win_probability=1-row['probability'],
        advantage_score=row['advantage'], is_even=abs(row['advantage']) <= EVEN_THRESHOLD,
        origin='physical_prediction' if projected else 'model',
        calibration_id=row.get('calibration_id', 'legacy-display' if row['source'] == 'legacy_mc'
                               else 'exchange-v2-existing-display'),
        evaluated_positions=1, source=row['source'])
    if row.get('b3_contents') is not None:
        payload['evaluations']['b3_contents'] = row['b3_contents']


def input_hold(payload: dict[str, Any], row: dict[str, Any], now: float,
               hold_started: float | None) -> None:
    """未確認入力では架空の勝率を作らず、HOLDの文字だけを表示する。"""
    state = row.get('input_calibration', dict(phase='verifying', progress=0))
    phase, progress = state['phase'], state['progress']
    message = {'overlay_feedback': '表示の映り込み：仮想カメラの出力をソースにしてください', 'verifying': '入力確認中', 'no_puyo_screen': '試合待ち',
               'calibrating': f'色を較正中 {progress}%'}[phase]
    payload['display'].update(visibility='visible', status='hold', input_status=phase,
        calibration_progress=progress, message=message, update_reason='hold_started',
        primary_hold_reason='recognition_unreliable', all_hold_reasons=['recognition_unreliable'],
        hold_started_ms=int(row['t_sec']*MILLISECONDS),
        hold_elapsed_ms=int((now-(hold_started or now))*MILLISECONDS))


def evaluation_hold(payload: dict, row: dict, now: float, started: float | None) -> None:
    """失敗した計算は値として公開せず、次の成功まで理由付きHOLDにする。"""
    payload['timing']['calculation_age_ms'] = None
    payload['runtime'].update(status='degraded', practical_worker_health='degraded')
    payload['display'].update(visibility='visible', status='hold', message='評価エラー',
        update_reason='hold_started', primary_hold_reason='prediction_worker_fault',
        all_hold_reasons=['prediction_worker_fault'],
        hold_started_ms=int(row['t_sec']*MILLISECONDS),
        hold_elapsed_ms=int((now-(started or now))*MILLISECONDS))


class LiveStreamState(StreamState):
    """既存SSEの購読契約を保ち、遅い購読者にも最新一件だけ残す。"""

    def publish(self, snapshot: OverlaySnapshot) -> None:
        payload = snapshot.to_mapping()
        with self._lock:
            self._latest_payload = payload
            for queue in self._subscribers:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass
                queue.put_nowait(payload)

    def subscribe(self) -> Queue:
        queue: Queue = Queue(maxsize=1)
        with self._lock:
            self._subscribers.append(queue)
            if self._latest_payload is not None:
                queue.put_nowait(self._latest_payload)
        return queue


class LiveOverlayServer(StreamOverlayServer):
    """HTTP/SSE処理は既存を再用し、新経路のHTMLと送出計装だけ追加する。"""

    def __init__(self, state: LiveStreamState, host: str, port: int,
                 overlay: dict[str, Any] | None = None, completion_driven: bool = False,
                 exclusive_bind: bool | None = None) -> None:
        # None = 二重起動の許可設定 (環境変数) に従う。既定は排他 bind (同じ port の 2 つ目は失敗する)。
        super().__init__(state, host, port,
                         exclusive_bind=exclusive_bind_default() if exclusive_bind is None else exclusive_bind)
        self.overlay = with_debug_defaults(validate_overlay({} if overlay is None else overlay))
        self.sent: dict[int, dict[str, Any]] = {}
        self.sent_lock = Lock()
        self.invalidated_at = float('-inf')
        self.completion_driven = completion_driven
        self.terminal_hold_enabled = False
        self.terminal_delivery: tuple[tuple | None, float] = (None, float('-inf'))

    def sendable(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """2Hz待機中に失効した評価は送らず、公開済みの最新HOLDを優先する。"""
        latest = self.state.latest()
        if latest and latest['identity']['stream_seq'] > data['identity']['stream_seq']:
            data = latest
        terminal_key, terminal_deadline = self.terminal_delivery
        if (self.terminal_hold_enabled
                and data['evaluations']['practical'].get('source') == 'confirmed_death'
                and (time.perf_counter() >= terminal_deadline or terminal_key != (
                    data['identity']['match_id'], data['timing']['capture_monotonic_sec']))):
            return None
        if (data['evaluations']['practical']['availability'] == 'available' and
                data['timing']['capture_monotonic_sec'] <= self.invalidated_at
                and not (data['evaluations']['practical'].get('source') == 'confirmed_death'
                         and time.perf_counter() < terminal_deadline)):
            return None
        return data

    def _make_handler_class(self) -> type:
        parent = super()._make_handler_class()
        owner = self

        class Handler(OverlayPagesMixin, parent):
            last_analysis = 0.0
            last_sequence: int | None = None
            overlay_options = owner.overlay
            overlay_html = HTML.replace('const COMPLETION_DRIVEN = false;',
                f'const COMPLETION_DRIVEN = {str(owner.completion_driven).lower()};')

            def _write_sse(self, event: str, data: dict[str, Any]) -> None:
                if event == 'analysis':
                    if not owner.completion_driven:
                        time.sleep(max(0.0, self.last_analysis + PUBLISH_PERIOD_SEC-time.perf_counter()))
                    data = owner.sendable(data)
                    if data is None:
                        return
                    sequence = data['identity']['stream_seq']
                    if owner.completion_driven and sequence == self.last_sequence:
                        return
                super()._write_sse(event, data)
                if event == 'analysis':
                    self.last_analysis = time.perf_counter()
                    self.last_sequence = sequence
                if event == 'analysis' and 'capture_monotonic_sec' in data['timing']:
                    with owner.sent_lock:
                        owner.sent.setdefault(data['identity']['stream_seq'],
                            dict(frame=data['timing']['source_available_frame'],
                                 available=data['evaluations']['practical']['availability'] == 'available',
                                 sent_at=time.perf_counter(), **data['timing']))

        return Handler


class LivePublisher:
    """評価からはlatest一件だけ受け取り、SnapshotHubへ単一writerで公開する。"""

    def __init__(self, assets: dict[str, str], host: str, port: int,
                 evaluation_driven: bool = False, overlay: dict[str, Any] | None = None,
                 completion_driven: bool = False,
                 freshness_gate: bool = EVALUATION_FRESHNESS_GATE_DEFAULT,
                 terminal_display_hold: bool = TERMINAL_DISPLAY_HOLD_DEFAULT,
                 exclusive_bind: bool | None = None) -> None:
        self.initial = initial_snapshot(assets)
        self.hub = SnapshotHub(self.initial)
        self.state = LiveStreamState()
        self.server = LiveOverlayServer(self.state, host, port, overlay=overlay,
                                        completion_driven=completion_driven, exclusive_bind=exclusive_bind)
        self.lock, self.stop = Lock(), Event()
        self.ready = Event()
        self.evaluation_driven = evaluation_driven
        self.completion_driven = completion_driven
        self.freshness_gate = freshness_gate
        self.terminal = TerminalDisplayHold() if terminal_display_hold else None
        if self.terminal is not None and os.environ.get('PUYO_TERMINAL_HOLD_INPUT_RECOVERY') == '1':
            self.terminal.input_recovery = True
        self.terminal_visible = False
        # 判定・保持のキーはgameのまま。公開番号だけ別途付与する。
        self.public_matches: PublicMatchNumber | None = None
        self.server.terminal_hold_enabled = terminal_display_hold
        self.offered_id: tuple | None = None
        self.latest: dict[str, Any] | None = None
        self.input_calibration: dict[str, Any] | None = None
        self.input_events: list[dict[str, Any]] = []
        self.publications: list[dict[str, Any]] = []
        self.ama_publication = None
        if os.environ.get('PUYO_LIVE_AMA_PLAIN') == '1':
            from .live_ama_publication import AmaPublication
            self.ama_publication = AmaPublication()
        self.error: BaseException | None = None
        self.thread = Thread(target=self._run, name='sse-publisher', daemon=True)

    def start(self) -> None:
        self.server.start()
        self.thread.start()

    def offer(self, row: dict[str, Any]) -> None:
        from .live_stage_timing import locked
        with locked('offer_lock_wait', self.lock):
            if self.public_matches is not None:
                row = self.public_matches.decorate(row)
            if self.terminal is not None:
                if not self.terminal.accepts(row):
                    return
                self.terminal.observe(row, time.perf_counter())
                self._sync_terminal_delivery()
            if getattr(self,'ama_publication',None) is not None:
                self.ama_publication.observe(row,time.perf_counter())
            if self.completion_driven:
                key = tuple(row.get('judgment_id', (row['game'], row['frame'])))
                if self.terminal is not None:
                    key = (row['game'], *key)
                if key == self.offered_id:
                    return
                self.offered_id = key
            self.latest = dict(row)
            if self.input_calibration:
                self.latest['input_calibration'] = dict(self.input_calibration)
            self.ready.set()

    def evaluation_hold(self, notice: Any, game: int) -> None:
        """評価例外を配信し、待機中の旧評価もこのHOLDへ置き換える。"""
        with self.lock:
            self.latest = dict(frame=notice.frame, t_sec=notice.t_sec, game=game,
                captured_at=notice.captured_at, recognized_at=notice.recognized_at,
                evaluated_at=time.perf_counter(), queue_depth=0, raw_probability=None,
                hold=True, evaluation_error=True)
            if self.public_matches is not None:
                self.latest = self.public_matches.decorate(self.latest)
            if self.terminal is not None:
                self.terminal.observe(self.latest, time.perf_counter())
                self._sync_terminal_delivery()
            self.server.invalidated_at = max(self.server.invalidated_at, notice.captured_at)
            row = self.latest
            self.ready.set()
        # 通常の公開周期を待たずHOLDを確定し、SSE待機中の旧評価も失効させる。
        if not self.completion_driven:
            self._publish(row, 0, time.perf_counter())

    def input_pending(self, now: float | dict[str, Any]) -> None:
        with self.lock:
            if isinstance(now, dict):
                if self.terminal is not None:
                    self.terminal.input_changed(now)
                    self._sync_terminal_delivery()
                if self.completion_driven and self.input_calibration:
                    keys = ('phase', 'progress', 'epoch', 'score_roi_alignment')
                    if self.terminal is not None:
                        keys += ('device_id',)
                    if all(now.get(key) == self.input_calibration.get(key) for key in keys):
                        return
                self.input_calibration = dict(now)
                self.input_events.append(dict(now))
                if now['phase'] == 'ready':
                    self.offered_id = None
                    return
                now = now['at']
            else:
                self.input_calibration = dict(phase='verifying', progress=0)
                self.input_events.append(dict(self.input_calibration, at=now))
            self.server.invalidated_at = max(self.server.invalidated_at, now)
            previous = self.latest or {}
            self.latest = dict(frame=previous.get('frame', 0), t_sec=previous.get('t_sec', 0.0),
                game=previous.get('game', 0), captured_at=now, recognized_at=now,
                evaluated_at=now, queue_depth=0, raw_probability=None,
                hold=True, input_calibration=dict(self.input_calibration))
            if self.public_matches is not None:
                self.latest = self.public_matches.decorate(self.latest)
            self.ready.set()

    from .live_stage_timing import publication as _time_publication

    @_time_publication
    def _publish(self, row: dict[str, Any], revision: int, hold: float | None) -> None:
        now = time.perf_counter()
        from .live_stage_timing import locked
        with locked('publication_lock_wait', self.lock):
            if self.completion_driven and row is not self.latest:
                return
            # 状態変更と送出を直列化し、旧rowを取得済みの配信threadも較正ゲートへ従わせる。
            revision = max(revision, self.hub.latest.identity['reducer_revision']+1)
            if self.latest and self.latest.get('evaluation_error'):
                row = self.latest
            if self.input_calibration and self.input_calibration['phase'] != 'ready':
                row = dict(row, input_calibration=dict(self.input_calibration))
            if self.terminal is not None:
                row = self.terminal.project(row, now)
                self.terminal_visible = bool(row.get('_terminal_display_held'))
            if getattr(self,'ama_publication',None) is not None:
                row = self.ama_publication.project(row,now)
            snapshot = result_snapshot(self.initial, row, now, revision, hold,
                                       freshness_gate=self.freshness_gate)
            # 試験用の追加欄。row に debug_panel があるときだけ載せ、既存の値は変えない。
            snapshot = with_debug_panel(snapshot, row)
            from .live_winrate_switch import with_switch
            snapshot = with_switch(snapshot, row)
            published = self.hub.publish(snapshot)
            if published.fail_closed:
                raise ValueError(f'公開DTO不正: {published.validation_report}')
            self.state.publish(published.snapshot)
            if os.environ.get('PUYO_B3_TIME_GATE') == '1':
                from .live_b3_time_gate import published as record_publication
                record_publication(row, published.snapshot, now)
            # 計装も同じ公開時点で凍結し、直後の境界の値を混ぜない。
            terminal_visible = self.terminal_visible
            terminal_armed = self.terminal.armed if self.terminal else None
            terminal_deadline = self.terminal.deadline if self.terminal else None
        self.publications.append(dict(frame=row['frame'], published_at=now,
            evaluated_at=row['evaluated_at'], captured_at=row['captured_at'],
            recognized_at=row['recognized_at'], hold=(published.snapshot.display['status'] == 'hold'
                if self.freshness_gate else hold is not None),
            game=row['game'], source=published.snapshot.evaluations['practical'].get('source'),
            probability=published.snapshot.evaluations['practical']['p1_win_probability'],
            terminal_display_held=terminal_visible,
            terminal_hold_armed=terminal_armed,
            terminal_deadline=terminal_deadline,
            **({'public_game': row['public_game']} if 'public_game' in row else {})))

    def _run(self) -> None:
        if self.completion_driven:
            self._run_completed()
            return
        revision, previous, hold, hold_media = 0, None, None, None
        last_row = None
        next_publish = time.perf_counter()
        try:
            while True:
                if self.evaluation_driven and not self.ready.wait(PUBLISH_PERIOD_SEC):
                    # 入力待機中も状態を送出し、表示側の接続監視を維持する。
                    with self.lock:
                        if ((not self.input_calibration or self.input_calibration['phase'] == 'ready')
                                and not self.terminal_visible
                                and not (self.freshness_gate and evaluation_stale(
                                    self.latest, time.perf_counter()))):
                            continue
                time.sleep(max(0.0, next_publish-time.perf_counter()))
                with self.lock:
                    row = self.latest
                    self.ready.clear()
                if self.evaluation_driven and self.stop.is_set() and row is last_row:
                    break
                if row is not None:
                    stale = previous == row['frame'] or row['hold']
                    if stale and hold is None:
                        hold_media = row['t_sec']
                    hold = (hold or time.perf_counter()) if stale else None
                    revision += 1
                    self._publish(dict(row, hold_media_sec=hold_media), revision, hold)
                    previous = row['frame']
                    last_row = row
                next_publish = time.perf_counter() + PUBLISH_PERIOD_SEC
                if self.stop.is_set():
                    break
        except BaseException as error:
            self.error = error

    def _run_completed(self) -> None:
        """確定イベントだけで公開。接続維持は既存SSE heartbeatへ任せる。"""
        previous, expired = None, None
        try:
            while True:
                self.ready.wait(self._wait_timeout())
                with self.lock:
                    row = self.latest
                    self.ready.clear()
                if row is not None and row is not previous:
                    hold = time.perf_counter() if row.get('hold') else None
                    self._publish(row, 0, hold)
                    previous = row
                    expired = None
                elif (self.terminal_visible and self.terminal is not None
                      and not self.terminal.active(time.perf_counter())):
                    self._publish(row, 0, None)
                elif (self.freshness_gate and row is not expired
                      and evaluation_stale(row, time.perf_counter())):
                    self._publish(row, 0, None)
                    expired = row
                if self.stop.is_set():
                    break
        except BaseException as error:
            self.error = error

    def _wait_timeout(self) -> float | None:
        """完了通知がなくても決着の期限ちょうどに解除を公開する。"""
        timeout = PUBLISH_PERIOD_SEC if self.freshness_gate or self.terminal else None
        if self.terminal is not None and self.terminal_visible:
            deadline = self.terminal.deadline
            if deadline is None:
                return 0.
            return max(0., min(PUBLISH_PERIOD_SEC, deadline-time.perf_counter()))
        return timeout

    def _sync_terminal_delivery(self) -> None:
        """SSE送出待ちの前試合・前機器の確定を、新しい保持期限で復活させない。"""
        row = self.terminal.row
        key = (str(row.get('public_game', row['game'])), row['captured_at']) if row else None
        # 別threadが旧keyと新deadlineを組み合わせないよう、一組で差し替える。
        self.server.terminal_delivery = (key, self.terminal.deadline or float('-inf'))

    def close(self) -> None:
        self.stop.set()
        self.ready.set()
        self.thread.join(PUBLISH_PERIOD_SEC * PUBLISHER_JOIN_PERIODS)
        self.server.stop()
        if self.thread.is_alive():
            raise RuntimeError('配信workerが停止しません')
        if self.error is not None:
            raise RuntimeError('配信workerが失敗しました') from self.error
