"""ローカルの設定・確認画面と固定された表示資産だけを公開する。"""
from __future__ import annotations

import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import webbrowser
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from .distribution_config import MAX_PORT_ATTEMPTS, camera_preview, devices, feedback_site_url
from .distribution_runtime import DistributionRuntime
from .distribution_logs import MAX_REPORTS
from .overlay_config import configured_html
from .overlay_http import ASSETS, BINARY_ASSETS
from .local_http import LocalHTTPHandler, LocalHTTPServer

MAX_BODY = 64 * 1024
PORT_LIMIT = 65536
WINDOWS_BIND_CONFLICTS = (10013, 10048)
STREAM_TIMEOUT = 20
RECONNECT_SECONDS = 1
SOURCE = Path(__file__).parent
EXTRA = {'/overlay_i18n.js': 'overlay_i18n.js', '/overlay_mascot.js': 'overlay_mascot.js',
         '/distribution_settings.js': 'distribution_settings.js'}


class DistributionHandler(LocalHTTPHandler):
    runtime: DistributionRuntime

    def log_message(self, format: str, *args: Any) -> None:
        return

    def respond(self, body: bytes, mime: str = 'application/json', status: int = 200) -> None:
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        try:
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def json(self, data: Any, status: int = 200) -> None:
        self.respond(json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8', status)

    def local_request(self) -> bool:
        host = f'127.0.0.1:{self.runtime.port}'
        return self.headers.get('Host') == host and self.headers.get('Origin', f'http://{host}') == f'http://{host}'

    def do_GET(self) -> None:
        if not self.local_request():
            return self.json(dict(error='Local access only'), 403)
        path = urlsplit(self.path).path
        if path == '/api/settings/events':
            return self.settings_events()
        if path == '/api/input-preview.jpg':
            return self.input_preview()
        if path == '/api/state':
            return self.state()
        if path in ('/events', '/latest'):
            return self.live(path)
        if path.startswith('/mascot/') and path != '/mascot/battle-still.webp':
            return self.mascot(path)
        if path in BINARY_ASSETS:
            filename, mime = BINARY_ASSETS[path]
            return self.respond((SOURCE / filename).read_bytes(), mime)
        if path == '/setup':
            self.send_response(302)
            self.send_header('Location', '/settings')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if path == '/':
            filename, mime = 'overlay.html', 'text/html'
        elif path in ASSETS:
            filename, mime = ASSETS[path]
        elif path in EXTRA:
            filename, mime = EXTRA[path], 'text/javascript'
        else:
            return self.json(dict(error='Not found'), 404)
        options = {**self.runtime.settings['overlay'], 'language': self.runtime.settings['language'],
                   'os_language': self.runtime.os_language, 'app_started_at': self.runtime.started_at,
                   'distribution_mode': True}
        body = configured_html((SOURCE / filename).read_text(encoding='utf-8'), options)
        self.respond(body, mime + '; charset=utf-8')

    def state(self) -> None:
        if self.runtime.inner_port:
            self.snapshot()
        self.json(dict(settings={k: v for k, v in self.runtime.settings.items() if k != 'obs_password'},
            obs_password_saved=bool(self.runtime.settings.get('obs_password')),
            feedback_enabled=bool(feedback_site_url()),
            diagnostics=self.runtime.diagnostics(), os_language=self.runtime.os_language,
            selected_input=self.runtime.preferred_device,
            root=str(self.runtime.root), url=f'http://127.0.0.1:{self.runtime.port}/',
            port_conflict=getattr(self.runtime, 'port_conflict', None),
            error=self.runtime.error.get('type'), update=self.runtime.update,
            running=bool(self.runtime.child and self.runtime.child.poll() is None)))

    def input_preview(self) -> None:
        runtime = self.runtime
        if runtime.inner_port and runtime.child and runtime.child.poll() is None:
            try:
                with urlopen(f'http://127.0.0.1:{runtime.inner_port}/input-preview.jpg', timeout=1) as response:
                    image = response.read()
                    if image:
                        runtime.preview_image = image
            except OSError:
                pass
        self.respond(runtime.preview_image, 'image/jpeg', 200 if runtime.preview_image else 204)

    def settings_events(self) -> None:
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        previous = ''
        try:
            while not self.runtime.finished.is_set():
                options = {**self.runtime.settings['overlay'], 'language': self.runtime.settings['language'],
                           'app_started_at': self.runtime.started_at}
                current = json.dumps(options, ensure_ascii=False)
                payload = f'event: appearance\ndata: {current}\n\n' if current != previous else ': keepalive\n\n'
                self.wfile.write(payload.encode('utf-8'))
                self.wfile.flush()
                previous = current
                time.sleep(RECONNECT_SECONDS)
        except (OSError, ConnectionError):
            return

    def mascot(self, path: str) -> None:
        if path in {f'/mascot/{band}.webp' for band in (10, 30, 50, 70, 90)}:
            name = Path(path).name
            custom = self.runtime.root / 'mascot' / name
            asset = custom if custom.is_file() else SOURCE / 'brand' / ('mascot_' + name)
            return self.respond(asset.read_bytes(), 'image/webp')
        if path != '/mascot/overlay.webp':
            from .mascot_assets import mascot_asset
            try:
                return self.respond(mascot_asset(SOURCE / 'brand', path.removeprefix('/mascot/')), 'image/webp')
            except FileNotFoundError:
                return self.json(dict(error='Not found'), 404)
        # 利用者の差し替えを優先し、未指定時は同梱アニメーションを使う。
        directory = self.runtime.root / 'mascot'
        for extension, mime in (('.webp', 'image/webp'), ('.png', 'image/png'), ('.gif', 'image/gif')):
            target = directory / ('overlay' + extension)
            if target.is_file():
                return self.respond(target.read_bytes(), mime)
        self.respond((SOURCE / 'brand/overlay_mascot.webp').read_bytes(), 'image/webp')

    def open_report_destination(self, path: str) -> dict[str, Any]:
        """利用者が押した操作だけを、固定のフォルダまたはサイトへ渡す。"""
        if path == '/api/reports/open':
            os.startfile(str(self.runtime.store.directory.resolve()))
        else:
            site = feedback_site_url()
            if not site:
                raise ValueError('フォームは準備中です / Feedback form is not available yet')
            if not webbrowser.open(site + '/#feedback'):
                raise OSError('ブラウザを開けませんでした / Could not open the browser')
        return dict(opened=True)

    def do_POST(self) -> None:
        if not self.local_request() or self.headers.get('Content-Type') != 'application/json':
            return self.json(dict(error='Same-origin JSON required'), 403)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BODY:
                raise ValueError('Request size')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('JSON object required')
            with self.runtime.control_lock:
                result = self.action(urlsplit(self.path).path, data)
            self.json(result)
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.json(dict(error=str(error)), 400)
        except Exception:
            self.json(dict(error='処理に失敗しました / Internal error'), 500)

    def action(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        runtime = self.runtime
        if path == '/api/port/switch':
            from .distribution_ports import switch_port
            return switch_port(self.server, runtime)
        if path == '/api/input/select':
            runtime.switch_device(data.get('index'))
            return dict(selected=True)
        if path == '/api/settings':
            runtime.configure(data)
            return dict(saved=True, warning=runtime.shortcut_warning)
        if path == '/api/start':
            runtime.configure({'completed': True})
            runtime.request_start()
            return dict(started=bool(runtime.child and runtime.child.poll() is None), input=runtime.input_probe)
        if path == '/api/discover':
            return runtime.scan_inputs()
        if path == '/api/obs':
            from .distribution_obs import integrate
            runtime.obs_result = integrate(runtime.settings, f'http://127.0.0.1:{runtime.port}/', data.get('apply') is True)
            if (runtime.obs_result.get('applied') or runtime.obs_result.get('already_added')) and not runtime.settings['stream_obs_done']:
                runtime.configure({'stream_obs_done': True})
            return runtime.obs_result
        if path == '/api/devices':
            return dict(devices=devices())
        if path == '/api/camera-test':
            import base64
            index = data['index']
            if type(index) is not int or not 0 <= index <= 64:
                raise ValueError('Invalid device index')
            if runtime.child and runtime.child.poll() is None:
                raise ValueError('起動前にテストしてください / Test before starting')
            return dict(image='data:image/jpeg;base64,' + base64.b64encode(camera_preview(index, runtime.input_probe)).decode('ascii'))
        if path == '/api/logs/preview':
            return runtime.store.preview(runtime.report(), runtime.settings['relay_url'],
                                         max_reports=MAX_REPORTS if feedback_site_url() else None)
        if path == '/api/logs/send':
            sent = runtime.store.send(data.get('token', ''), data.get('confirmed', False))
            return dict(sent=sent, **runtime.store.last_send_result)
        if path in ('/api/reports/open', '/api/feedback/open'):
            return self.open_report_destination(path)
        if path == '/api/pause':
            runtime.stop(final=False)
            return dict(stopped=True)
        if path == '/api/stop':
            runtime.stop()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return dict(stopped=True)
        raise ValueError('Unknown action')

    def live(self, path: str) -> None:
        if path == '/latest':
            self.json(self.snapshot())
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        try:
            while not self.runtime.stopping:
                if self.runtime.inner_port:
                    try:
                        self.relay_events()
                        continue
                    except (OSError, ValueError):
                        pass
                snapshot = self.snapshot()
                self.wfile.write(('event: analysis\ndata: ' + json.dumps(snapshot) + '\n\n').encode())
                self.wfile.flush()
                time.sleep(RECONNECT_SECONDS)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def relay_events(self) -> None:
        """元のSSEの更新間隔を保ち、無更新を新しい観測に見せかけない。"""
        url = f'http://127.0.0.1:{self.runtime.inner_port}/events'
        with urlopen(url, timeout=STREAM_TIMEOUT) as response:
            event = b''
            while not self.runtime.stopping:
                line = response.readline(MAX_BODY * 8)
                if not line:
                    return
                if line.startswith(b'event:'):
                    event = line[len(b'event:'):].strip()
                if line.startswith(b'data: ') and event == b'analysis':
                    self.runtime.accept_snapshot(json.loads(line[len(b'data: '):]))
                if not line.strip():
                    event = b''
                self.wfile.write(line)
                self.wfile.flush()

    def snapshot(self) -> dict[str, Any]:
        runtime = self.runtime
        if runtime.inner_port:
            try:
                with urlopen(f'http://127.0.0.1:{runtime.inner_port}/latest', timeout=1) as response:
                    data = json.loads(response.read(MAX_BODY * 8))
                if data:
                    runtime.accept_snapshot(data)
                    return data
            except (OSError, ValueError):
                pass
        return dict(display=dict(input_status='checking', status='waiting'), evaluations={})


def make_server(runtime: DistributionRuntime, port: int) -> ThreadingHTTPServer:
    from .distribution_ports import inspect_conflict
    runtime.requested_port = port
    runtime.port_conflict = None
    handler = type('InstalledDistributionHandler', (DistributionHandler,), {'runtime': runtime})
    for candidate in range(port, min(PORT_LIMIT, port + MAX_PORT_ATTEMPTS)):
        try:
            server = LocalHTTPServer(('127.0.0.1', candidate), handler)
        except OSError as error:
            if error.errno != errno.EADDRINUSE and getattr(error, 'winerror', None) not in WINDOWS_BIND_CONFLICTS:
                raise
            if candidate == port:
                runtime.port_conflict = inspect_conflict(runtime.root, port)
            continue
        runtime.port = server.server_address[1]
        from .distribution_instance import record_port
        try:
            record_port(runtime.root, runtime.port)
        except OSError:
            server.server_close()
            raise
        return server
    raise OSError('HTTP のポートを確保できません / No available HTTP port')
