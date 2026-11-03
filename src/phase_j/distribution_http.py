"""ローカルの設定・確認画面と固定された表示資産だけを公開する。"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from .distribution_config import camera_preview, devices
from .distribution_runtime import DistributionRuntime
from .distribution_uninstall import launch_uninstaller
from .overlay_config import configured_html
from .overlay_http import ASSETS, BINARY_ASSETS

MAX_BODY = 64 * 1024
STREAM_TIMEOUT = 20
RECONNECT_SECONDS = 1
SOURCE = Path(__file__).parent
EXTRA = {'/overlay_i18n.js': 'overlay_i18n.js', '/overlay_mascot.js': 'overlay_mascot.js',
         '/distribution_settings.js': 'distribution_settings.js', '/setup.js': 'distribution_setup.js'}


class DistributionHandler(BaseHTTPRequestHandler):
    runtime: DistributionRuntime

    def log_message(self, format: str, *args: Any) -> None:
        return

    def respond(self, body: bytes, mime: str = 'application/json', status: int = 200) -> None:
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def json(self, data: Any, status: int = 200) -> None:
        self.respond(json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8', status)

    def local_request(self) -> bool:
        host = f'127.0.0.1:{self.runtime.port}'
        return self.headers.get('Host') == host and self.headers.get('Origin', f'http://{host}') == f'http://{host}'

    def do_GET(self) -> None:
        if not self.local_request():
            return self.json(dict(error='Local access only'), 403)
        path = urlsplit(self.path).path
        if path == '/api/state':
            if self.runtime.inner_port:
                self.snapshot()
            return self.json(dict(settings={k: v for k, v in self.runtime.settings.items() if k != 'obs_password'},
                diagnostics=self.runtime.diagnostics(), os_language=self.runtime.os_language,
                root=str(self.runtime.root), url=f'http://127.0.0.1:{self.runtime.port}/',
                error=self.runtime.error.get('type'), update=self.runtime.update,
                running=bool(self.runtime.child and self.runtime.child.poll() is None)))
        if path in ('/events', '/latest'):
            return self.live(path)
        if path.startswith('/mascot/'):
            return self.mascot(path)
        if path in BINARY_ASSETS:
            filename, mime = BINARY_ASSETS[path]
            return self.respond((SOURCE / filename).read_bytes(), mime)
        if path in ('/', '/setup'):
            filename, mime = ('overlay.html', 'text/html') if path == '/' else ('distribution_setup.html', 'text/html')
        elif path in ASSETS:
            filename, mime = ASSETS[path]
        elif path in EXTRA:
            filename, mime = EXTRA[path], 'text/javascript'
        else:
            return self.json(dict(error='Not found'), 404)
        options = {**self.runtime.settings['overlay'], 'language': self.runtime.settings['language'],
                   'os_language': self.runtime.os_language, 'app_started_at': self.runtime.started_at}
        body = configured_html((SOURCE / filename).read_text(encoding='utf-8'), options)
        self.respond(body, mime + '; charset=utf-8')

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

    def do_POST(self) -> None:
        if not self.local_request() or self.headers.get('Content-Type') != 'application/json':
            return self.json(dict(error='Same-origin JSON required'), 403)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BODY:
                raise ValueError('Request size')
            data = json.loads(self.rfile.read(size))
            with self.runtime.lock:
                result = self.action(urlsplit(self.path).path, data)
            self.json(result)
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.json(dict(error=str(error)), 400)

    def action(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        runtime = self.runtime
        if path == '/api/settings':
            runtime.configure(data)
            return dict(saved=True)
        if path == '/api/start':
            runtime.configure({'completed': True})
            runtime.start()
            return dict(started=bool(runtime.child and runtime.child.poll() is None), input=runtime.input_probe)
        if path == '/api/discover':
            return runtime.scan_inputs()
        if path == '/api/obs':
            from .distribution_obs import integrate
            runtime.obs_result = integrate(runtime.settings, f'http://127.0.0.1:{runtime.port}/', data.get('apply') is True)
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
            return runtime.store.preview(runtime.report(), runtime.settings['relay_url'])
        if path == '/api/logs/send':
            return dict(sent=runtime.store.send(data.get('token', ''), data.get('confirmed', False)))
        if path == '/api/uninstall':
            if data.get('confirmed') is not True:
                raise ValueError('確認してください / Confirmation required')
            launch_uninstaller(runtime.root)
            runtime.stop()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return dict(uninstalling=True)
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
            while not self.runtime.stopping:
                line = response.readline(MAX_BODY * 8)
                if not line:
                    return
                if line.startswith(b'data: '):
                    self.runtime.accept_snapshot(json.loads(line[len(b'data: '):]))
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
    handler = type('InstalledDistributionHandler', (DistributionHandler,), {'runtime': runtime})
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    server.daemon_threads = True
    runtime.port = server.server_address[1]
    return server
