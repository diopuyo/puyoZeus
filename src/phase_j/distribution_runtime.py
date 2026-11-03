"""初期設定を保持したまま認識プロセスを監督し、障害画面を残す。"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
from urllib.request import urlopen

from .distribution_config import available_port, load_settings, os_language, save_settings
from .distribution_logs import ReportStore, tail
from .distribution_job import assign_job, close_job, create_job
from .distribution_windows import shortcuts
from .release_config import distribution_user_config, release_environment

POLL_SECONDS = 1
MAX_RUNTIME_LOG_BYTES = 2 * 1024 * 1024
DEFAULT_VERSION = dict(app='development', model='v6', code_sha='unknown')


class DistributionRuntime:
    """各インストール先に閉じた状態。試験は独立したrootを渡す。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        configuration_error = ''
        try:
            self.settings = load_settings(self.root)
        except (OSError, ValueError) as error:
            from .distribution_config import validate_settings
            self.settings = validate_settings({})
            configuration_error = str(error)
        self.os_language = os_language()
        self.logs = self.root / 'logs'
        fallback = Path(os.environ['LOCALAPPDATA']) / 'PuyoLiveManualReports' if os.name == 'nt' else None
        self.store = ReportStore(self.logs / 'reports', fallback)
        self.version = DEFAULT_VERSION
        version = self.root / 'app' / 'RELEASE.json'
        if version.exists():
            self.version = json.loads(version.read_text(encoding='utf-8'))
        self.child: subprocess.Popen | None = None
        self.job: int | None = None
        self.inner_port = 0
        self.port = 0
        self.error: dict[str, Any] = {}
        self.latest: dict[str, Any] = {}
        self.input_probe: dict[str, Any] = {}
        self.update: dict[str, Any] = {}
        self.stopping = False
        self.started_at = time.time()
        self.last_update = None
        self.scanning = False
        self.obs_result: dict[str, Any] = {}
        self.lock = threading.RLock()
        if configuration_error:
            self.record_failure('configuration_error', configuration_error)

    def configure(self, changes: dict[str, Any]) -> None:
        from .distribution_config import validate_settings
        with self.lock:
            merged = {**self.settings, **changes}
            if 'overlay' in changes:
                merged['overlay'] = {**self.settings['overlay'], **changes['overlay']}
                merged['language'] = merged['overlay'].get('language', merged['language'])
            merged = validate_settings(merged)
            shortcuts(self.root, merged)
            save_settings(self.root, merged)
            self.settings = merged

    def scan_inputs(self) -> dict[str, Any]:
        from .distribution_config import devices
        from .distribution_input import discover
        if self.child and self.child.poll() is None:
            raise ValueError('入力使用中 / Input is in use')
        self.scanning = True
        try:
            self.input_probe = discover(devices())
            selected = self.input_probe.get('selected')
            if selected:
                self.configure(dict(device_index=selected['index'], device_name=selected['name']))
                self.input_probe.update(selected)
            return self.input_probe
        finally:
            self.scanning = False

    def accept_snapshot(self, data: dict[str, Any]) -> None:
        if data != self.latest:
            self.latest = data
            self.last_update = time.time()

    def diagnostics(self) -> dict[str, Any]:
        display = self.latest.get('display', {})
        return dict(scanning=self.scanning, input=self.input_probe,
            input_resolution=self.input_probe.get('input_resolution'), fps=self.input_probe.get('fps'),
            calibration=display.get('input_status', 'unknown'),
            display_status=display.get('status'), match_phase=display.get('match_phase'),
            delay_ms=self.latest.get('timing', {}).get('capture_to_publish_latency_ms'),
            last_update=self.last_update, started_at=self.started_at, obs=self.obs_result)

    def start(self) -> None:
        with self.lock:
            if self.child and self.child.poll() is None:
                return
            self.error = {}
            self.stopping = False
            if self.settings['auto_input'] and not self.scan_inputs().get('selected'):
                return
            if self.stopping:
                return
            self.inner_port = available_port(self.port + 1 if self.port < 65535 else 1024)
            config = distribution_user_config(self.settings, self.root, self.inner_port)
            config_path = self.root / 'puyo_live.json'
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
            executable = str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable
            command = [executable, '-m', 'src.phase_j.launcher', '--config', str(config_path), '--require-manifest']
            app = self.root / 'app'
            try:
                close_job(self.job)
                self.job = create_job()
                self.child = subprocess.Popen(command, cwd=app, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=release_environment({**os.environ, 'PYTHONUTF8': '1'}),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                assign_job(self.job, self.child)
            except Exception:
                self.record_failure('startup_failure', traceback.format_exc())
                return
            threading.Thread(target=self._watch_child, daemon=True).start()
            if self.settings['check_updates']:
                threading.Thread(target=self.check_update, daemon=True).start()

    def _watch_child(self) -> None:
        child = self.child
        path = self.logs / 'runtime.log'
        with path.open('ab') as stream:
            for line in iter(child.stdout.readline, b''):
                if stream.tell() > MAX_RUNTIME_LOG_BYTES:
                    stream.seek(0)
                    stream.truncate()
                stream.write(line)
                stream.flush()
        code = child.wait()
        if not self.stopping:
            kind = 'abnormal_exit' if self.latest else 'startup_failure'
            self.record_failure(kind, '\n'.join(tail(path)), code)

    def record_failure(self, kind: str, stack: str, code: int | None = None) -> None:
        locations = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', stack)
        location = ':'.join(locations[-1]) if locations else 'launcher.pipeline'
        self.error = dict(type=kind, location=location, stack_trace=stack, exit_code=code)
        self.store.save([self.report()])
        print(f'失敗 / Failure: {kind}. ログを送る / Send logs: http://127.0.0.1:{self.port}/settings', flush=True)

    def report(self) -> dict[str, Any]:
        display = self.latest.get('display', {})
        diagnostics = dict(input=display, timing=self.latest.get('timing', {}),
            telemetry=self.latest.get('telemetry', {}), input_resolution='unknown', fps='unknown',
            calibration=display.get('input_status', 'unknown'),
            process_delay=self.latest.get('timing', {}).get('capture_to_publish_latency_ms'),
            child_running=bool(self.child and self.child.poll() is None))
        diagnostics.update(self.input_probe)
        diagnostics.update(self.diagnostics())
        for line in reversed(tail(self.root / 'output/runtime.jsonl')):
            try:
                diagnostics['runtime'] = json.loads(line)
                break
            except ValueError:
                continue
        output = self.root / 'output'
        for name in ('status.json', 'summary.json'):
            path = output / name
            if path.exists() and path.stat().st_size < MAX_RUNTIME_LOG_BYTES:
                try:
                    diagnostics[name] = json.loads(path.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    pass
        settings = {k: v for k, v in self.settings.items() if k not in ('relay_url', 'releases_url', 'obs_password')}
        return self.store.report(self.version, settings, self.error or dict(type='manual', location='settings'),
            diagnostics, tail(self.logs / 'runtime.log'), tail(self.root / 'install.log'))

    def check_update(self) -> None:
        """更新一覧の取得のみ。インストールIDや利用情報は付けない。"""
        try:
            with urlopen(self.settings['releases_url'], timeout=5) as response:
                data = json.loads(response.read(64 * 1024))
            self.update = dict(available=data.get('version') != self.version['app'], version=data.get('version'))
        except Exception:
            self.update = dict(available=False, status='unavailable')

    def stop(self) -> None:
        self.stopping = True
        close_job(self.job)
        self.job = None
        if self.child and self.child.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(self.child.pid), '/T', '/F'],
                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.child.terminate()
            self.child.wait(timeout=10)
        (self.logs / 'session.running').unlink(missing_ok=True)

    def begin_session(self) -> None:
        """強制終了の痕跡は次回起動時にローカル記録だけを作る。"""
        marker = self.logs / 'session.running'
        if marker.exists():
            self.record_failure('abnormal_exit', 'Previous application session did not close cleanly')
        marker.write_text(str(time.time()), encoding='ascii')
