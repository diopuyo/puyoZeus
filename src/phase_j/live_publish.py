"""B2専用のPhase J DTO生成と既存SSEサーバの接続。"""
from __future__ import annotations

from datetime import datetime, timezone
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Any
from uuid import uuid4

from src.stream_overlay import StreamOverlayServer, StreamState
from .contracts import OverlaySnapshot
from .display_projector import PublicationContext, project_snapshot
from .reducer import Bootstrap, initial_state
from .snapshot_hub import SnapshotHub

PUBLISH_PERIOD_SEC = 0.5
MILLISECONDS = 1000.0
EVEN_THRESHOLD = 3.0
HTTP_OK = 200
PUBLISHER_JOIN_PERIODS = 4
PROJECTED_SOURCES = frozenset({'S3_landing', 'unavoidable_death'})
HTML = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<title>Phase J ライブ</title><style>body{background:transparent;color:white;
font:28px sans-serif;text-shadow:1px 1px 3px black}</style><div id="value">待機中</div>
<script>const events=new EventSource('/events');events.addEventListener('analysis', e=>{
const d=JSON.parse(e.data), p=d.evaluations.practical;
document.getElementById('value').textContent=d.display.input_status&&d.display.input_status!=='ready'?'HOLD '+d.display.message:
d.display.visibility==='hidden'?'待機中':
`1P ${(p.p1_win_probability*100).toFixed(1)}% | ${p.source} | ${d.display.status}`+
(d.evaluations.counter_search?.pending?' | 応手 計算中'+
(d.evaluations.counter_search.result_generation===null?'（未取得）':'（直前値）'):'');
});events.onerror=()=>{document.getElementById('value').textContent='HOLD 接続待ち';};
</script></html>'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initial_snapshot(assets: dict[str, str]) -> OverlaySnapshot:
    state = initial_state(Bootstrap('live-' + uuid4().hex, assets=assets))
    return project_snapshot(state, PublicationContext(0, utc_now(), int(time.perf_counter() * MILLISECONDS)))


def result_snapshot(initial: OverlaySnapshot, row: dict[str, Any], now: float,
                    revision: int, hold_started: float | None) -> OverlaySnapshot:
    """既存DTOにsourceと単調時刻の観測列だけを追加する。"""
    payload = initial.to_mapping()
    payload['identity'].update(reducer_revision=revision, match_id=str(row['game']),
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
    if (calibration and calibration['phase'] != 'ready') or row.get('input_verifying'):
        payload['timing']['calculation_age_ms'] = None
        input_hold(payload, row, now, hold_started)
        return OverlaySnapshot.from_mapping(payload)
    if calibration:
        payload['display'].update(input_status='ready', calibration_progress=100)
    if 'display_layers' in row:
        payload['evaluations']['display_layers'] = row['display_layers']
    if 'counter_search' in row:
        payload['evaluations']['counter_search'] = row['counter_search']
    if row['raw_probability'] is None:
        payload['timing']['calculation_age_ms'] = None
        return OverlaySnapshot.from_mapping(payload)
    fill_evaluation(payload, row, now, hold_started)
    return OverlaySnapshot.from_mapping(payload)


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
        hold_started_ms=int(row.get('hold_media_sec', row['t_sec'])*MILLISECONDS) if hold else None,
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
        calibration_id='exchange-v2-existing-display', evaluated_positions=1, source=row['source'])


def input_hold(payload: dict[str, Any], row: dict[str, Any], now: float,
               hold_started: float | None) -> None:
    """未確認入力では架空の勝率を作らず、HOLDの文字だけを表示する。"""
    state = row.get('input_calibration', dict(phase='verifying', progress=0))
    phase, progress = state['phase'], state['progress']
    message = {'verifying': '入力確認中', 'no_puyo_screen': 'ぷよ画面なし',
               'calibrating': f'色を較正中 {progress}%'}[phase]
    payload['display'].update(visibility='visible', status='hold', input_status=phase,
        calibration_progress=progress, message=message, update_reason='hold_started',
        primary_hold_reason='recognition_unreliable', all_hold_reasons=['recognition_unreliable'],
        hold_started_ms=int(row['t_sec']*MILLISECONDS),
        hold_elapsed_ms=int((now-(hold_started or now))*MILLISECONDS))


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

    def __init__(self, state: LiveStreamState, host: str, port: int) -> None:
        super().__init__(state, host, port)
        self.sent: dict[int, dict[str, Any]] = {}
        self.sent_lock = Lock()

    def _make_handler_class(self) -> type:
        parent = super()._make_handler_class()
        owner = self

        class Handler(parent):
            last_analysis = 0.0

            def _serve_html(self) -> None:
                body = HTML.encode('utf-8')
                self.send_response(HTTP_OK)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_sse(self, event: str, data: dict[str, Any]) -> None:
                if event == 'analysis':
                    time.sleep(max(0.0, self.last_analysis + PUBLISH_PERIOD_SEC-time.perf_counter()))
                super()._write_sse(event, data)
                if event == 'analysis':
                    self.last_analysis = time.perf_counter()
                if event == 'analysis' and 'capture_monotonic_sec' in data['timing']:
                    with owner.sent_lock:
                        owner.sent.setdefault(data['identity']['stream_seq'],
                            dict(frame=data['timing']['source_available_frame'],
                                 sent_at=time.perf_counter(), **data['timing']))

        return Handler


class LivePublisher:
    """評価からはlatest一件だけ受け取り、SnapshotHubへ単一writerで公開する。"""

    def __init__(self, assets: dict[str, str], host: str, port: int) -> None:
        self.initial = initial_snapshot(assets)
        self.hub = SnapshotHub(self.initial)
        self.state = LiveStreamState()
        self.server = LiveOverlayServer(self.state, host, port)
        self.lock, self.stop = Lock(), Event()
        self.latest: dict[str, Any] | None = None
        self.input_calibration: dict[str, Any] | None = None
        self.publications: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self.thread = Thread(target=self._run, name='sse-publisher', daemon=True)

    def start(self) -> None:
        self.server.start()
        self.thread.start()

    def offer(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.latest = dict(row)
            if self.input_calibration:
                self.latest['input_calibration'] = dict(self.input_calibration)

    def input_pending(self, now: float | dict[str, Any]) -> None:
        with self.lock:
            if isinstance(now, dict):
                self.input_calibration = dict(now)
                if now['phase'] == 'ready':
                    return
                now = now['at']
            else:
                self.input_calibration = dict(phase='verifying', progress=0)
            previous = self.latest or {}
            self.latest = dict(frame=previous.get('frame', 0), t_sec=previous.get('t_sec', 0.0),
                game=previous.get('game', 0), captured_at=now, recognized_at=now,
                evaluated_at=now, queue_depth=0, raw_probability=None,
                hold=True, input_calibration=dict(self.input_calibration))

    def _publish(self, row: dict[str, Any], revision: int, hold: float | None) -> None:
        now = time.perf_counter()
        with self.lock:
            # 状態変更と送出を直列化し、旧rowを取得済みの配信threadも較正ゲートへ従わせる。
            if self.input_calibration and self.input_calibration['phase'] != 'ready':
                row = dict(row, input_calibration=dict(self.input_calibration))
            snapshot = result_snapshot(self.initial, row, now, revision, hold)
            published = self.hub.publish(snapshot)
            if published.fail_closed:
                raise ValueError(f'公開DTO不正: {published.validation_report}')
            self.state.publish(published.snapshot)
        self.publications.append(dict(frame=row['frame'], published_at=now,
            evaluated_at=row['evaluated_at'], captured_at=row['captured_at'],
            recognized_at=row['recognized_at'], hold=hold is not None))

    def _run(self) -> None:
        revision, previous, hold, hold_media = 0, None, None, None
        next_publish = time.perf_counter()
        try:
            while True:
                time.sleep(max(0.0, next_publish-time.perf_counter()))
                with self.lock:
                    row = self.latest
                if row is not None:
                    stale = previous == row['frame'] or row['hold']
                    if stale and hold is None:
                        hold_media = row['t_sec']
                    hold = (hold or time.perf_counter()) if stale else None
                    revision += 1
                    self._publish(dict(row, hold_media_sec=hold_media), revision, hold)
                    previous = row['frame']
                next_publish = time.perf_counter() + PUBLISH_PERIOD_SEC
                if self.stop.is_set():
                    break
        except BaseException as error:
            self.error = error

    def close(self) -> None:
        self.stop.set()
        self.thread.join(PUBLISH_PERIOD_SEC * PUBLISHER_JOIN_PERIODS)
        self.server.stop()
        if self.thread.is_alive():
            raise RuntimeError('配信workerが停止しません')
        if self.error is not None:
            raise RuntimeError('配信workerが失敗しました') from self.error
