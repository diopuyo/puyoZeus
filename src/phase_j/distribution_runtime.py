"""初期設定を保持したまま認識プロセスを監督し、障害画面を残す。"""
from __future__ import annotations

import json
from .distribution_location import is_onedrive_root
from multiprocessing.shared_memory import SharedMemory
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any
from urllib.request import Request, build_opener

from .distribution_config import available_port, load_settings, os_language, save_settings
from .distribution_logs import SITE_USER_AGENT, NoRedirect, ReportStore, tail
from .distribution_summary import NumericSummary
from .distribution_config import feedback_site_url
from .distribution_job import assign_job, close_job, create_job
from .distribution_windows import shortcuts, system_tool
from .distribution_output import prepare_output
from .release_config import distribution_user_config, release_environment

POLL_SECONDS = 1
MAX_RUNTIME_LOG_BYTES = 2 * 1024 * 1024
DEFAULT_VERSION = dict(app='development', model='v6', code_sha='unknown')
AUTOSTART_RETRY_SECONDS = 5
CHILD_EXIT_TIMEOUT_SECONDS = 10
RESTART_WINDOW_SECONDS = 600
RESTART_LIMIT = 3
RESTART_DELAY_SECONDS = 5
PREVIEW_STOP_CHECK_SECONDS = 3


class DistributionRuntime:
    """各インストール先に閉じた状態。試験は独立したrootを渡す。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        configuration_error = ''
        try:
            self.settings = load_settings(self.root)
        except (OSError, ValueError) as error:
            from .distribution_config import validate_settings
            damaged = self.root / 'distribution.json'
            if damaged.exists():
                damaged.rename(self.root / f'distribution.invalid-{uuid.uuid4().hex}.json')
            self.settings = validate_settings({})
            configuration_error = str(error)
        self.os_language = os_language()
        self.logs = self.root / 'logs'
        fallback = Path(os.environ['LOCALAPPDATA']) / 'PuyoLiveManualReports' if os.name == 'nt' else None
        self.store = ReportStore(self.logs / 'reports', fallback, root=self.root)
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
        self.numeric_summary = NumericSummary()
        self.last_update = None
        self.scanning = False
        self.obs_result: dict[str, Any] = {}
        self.lock = threading.RLock()
        self.control_lock = threading.RLock()
        self.finished = threading.Event()
        self.cancel_start = threading.Event()
        self.input_waiting = False
        self.autostart_waiting = False
        self.restarting, self.restart_times = False, []
        self.retry_devices: list[dict[str, Any]] | None = None
        self.preview_image, self._preview_stop_checked, self._preview_stopped = b'', float('-inf'), False
        self.preview_memory: SharedMemory | None = None
        self.preferred_device: int | None = None
        self.start_thread: threading.Thread | None = None
        self.shortcut_warning = ''
        self.update_checked = False
        if configuration_error:
            self.record_failure('configuration_error', configuration_error)

    def configure(self, changes: dict[str, Any]) -> None:
        from .distribution_config import validate_settings
        with self.control_lock:
            merged = {**self.settings, **changes}
            if 'overlay' in changes:
                merged['overlay'] = {**self.settings['overlay'], **changes['overlay']}
                merged['language'] = merged['overlay'].get('language', merged['language'])
            merged = validate_settings(merged)
            shortcut_choice = bool(set(changes) & {'desktop_shortcut', 'start_menu_shortcut', 'autostart'})
            first_start = merged['completed'] and not self.settings['completed']
            save_settings(self.root, merged)
            self.settings = merged
            self.shortcut_warning = ''
            try:
                if shortcut_choice or first_start:
                    shortcuts(self.root, merged)
            except (OSError, subprocess.SubprocessError):
                self.shortcut_warning = '設定は保存しました。ショートカットを作成できません / Settings saved; could not create shortcuts'

    def close_preview(self) -> None:
        if self.preview_memory is not None:
            self.preview_memory.close()
            self.preview_memory.unlink()
            self.preview_memory = None

    def preview_environment(self) -> dict[str, str]:
        from .live_preview import PREVIEW_CAPACITY, PREVIEW_MEMORY_ENV
        self.close_preview()
        self.preview_memory = SharedMemory(create=True, size=PREVIEW_CAPACITY)
        return {**release_environment({**os.environ, 'PYTHONUTF8': '1'}),
                'PUYO_INPUT_PREVIEW': '1', PREVIEW_MEMORY_ENV: self.preview_memory.name}

    def take_preview(self) -> None:
        """検索結果の画像を診断・ログに混ぜず取り出す。"""
        import base64
        items = self.input_probe.get('devices', [])
        selected = self.input_probe.get('selected') or (items[0] if items else {})
        encoded = selected.get('preview', '')
        for item in items:
            item.pop('preview', None)
        selected.pop('preview', None)
        self.preview_image = base64.b64decode(encoded) if encoded else b''

    def switch_device(self, index: int | None) -> None:
        from .distribution_config import devices
        if index is not None and (type(index) is not int or index not in {d['index'] for d in devices()}):
            raise ValueError('入力が見つかりません / Input not found')
        self.preferred_device = index
        self.retry_devices = None
        self.preview_image = b''
        if self.input_waiting:
            return
        active = bool(self.child and self.child.poll() is None)
        self.stop(final=False)
        if active:
            self.request_start()
        else:
            self.scan_inputs()

    def scan_inputs(self) -> dict[str, Any]:
        from .distribution_config import devices
        from .distribution_input import discover
        if self.child and self.child.poll() is None:
            raise ValueError('入力使用中 / Input is in use')
        self.scanning = True
        try:
            items = self.retry_devices if self.input_waiting and self.retry_devices is not None else devices()
            if self.autostart_waiting:
                items = [item for item in devices() if item['name'].lower() == 'obs virtual camera']
            elif self.preferred_device is not None:
                items = [item for item in items if item['index'] == self.preferred_device]
            self.input_probe = discover(items)
            self.take_preview()
            if self.input_waiting and self.retry_devices is None:
                self.retry_devices = [item for item in self.input_probe['devices']
                    if self.preferred_device is not None or item.get('input_resolution') == [1920, 1080]
                    or 'obs virtual camera' in item['name'].lower()]
            selected = self.input_probe.get('selected')
            if selected:
                self.configure(dict(device_index=selected['index'], device_name=selected['name']))
                self.input_probe.update(selected)
            return self.input_probe
        finally:
            self.scanning = False

    def accept_snapshot(self, data: dict[str, Any]) -> None:
        with self.lock:
            self.restarting = False
            if data != self.latest:
                self.latest = data
                self.last_update = time.time()
                if feedback_site_url():
                    self.numeric_summary.observe(data)

    def diagnostics(self) -> dict[str, Any]:
        display = self.latest.get('display', {})
        probe = self.input_probe
        preview_stopped = self._preview_obs_stopped(
            display.get('input_status') == 'verifying' or display.get('calibration') == 'verifying')
        if display.get('input_status') in ('obs_stopped', 'overlay_feedback'):
            from .distribution_input import MESSAGES
            reason = 'feedback' if display['input_status'] == 'overlay_feedback' else 'obs_stopped'
            probe = dict(probe, reason=reason, message=MESSAGES[reason])
        elif preview_stopped:
            from .distribution_input import MESSAGES
            probe = dict(probe, reason='obs_stopped', message=MESSAGES['obs_stopped'])
        return dict(scanning=self.scanning or self.input_waiting, input=probe,
            restarting=self.restarting,
            onedrive=is_onedrive_root(self.root), warning=self.shortcut_warning, input_waiting=self.input_waiting,
            input_resolution=self.input_probe.get('input_resolution'), fps=self.input_probe.get('fps'),
            calibration=display.get('input_status', 'unknown'),
            display_status=display.get('status'), match_phase=display.get('match_phase'),
            delay_ms=self.latest.get('timing', {}).get('capture_to_publish_latency_ms'),
            last_update=self.last_update, started_at=self.started_at, obs=self.obs_result)

    def _preview_obs_stopped(self, verifying: bool) -> bool:
        """設定画面の案内だけを更新し、映像側の状態通知には触れない。"""
        from .live_preview import shared_image
        with self.lock:
            if not verifying:
                self._preview_stop_checked, self._preview_stopped = float('-inf'), False
                return False
            now = time.monotonic()
            if now - self._preview_stop_checked < PREVIEW_STOP_CHECK_SECONDS:
                return self._preview_stopped
            self._preview_stop_checked, self._preview_stopped = now, False
            try:
                import cv2
                import numpy as np
                from .distribution_input import is_obs_stopped
                encoded = (shared_image(self.preview_memory) if self.preview_memory is not None
                           else self.preview_image)
                if encoded:
                    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self._preview_stopped = is_obs_stopped(frame)
            except Exception:
                # 途中の共有メモリや壊れたJPEGで設定窓の状態取得を止めない。
                self._preview_stopped = False
            return self._preview_stopped

    def start(self) -> None:
        with self.control_lock:
            if self.child and self.child.poll() is None:
                return
            self.error = {}
            self.stopping = False
            try:
                if (self.settings['auto_input'] or self.autostart_waiting) and not self.scan_inputs().get('selected'):
                    return
                if self.stopping:
                    return
                self.inner_port = available_port(self.port + 1 if self.port < 65535 else 1024)
                config = distribution_user_config(self.settings, self.root, self.inner_port)
                config_path = self.root / 'puyo_live.json'
                config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
                executable = str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable
                command = [executable, '-s', '-m', 'src.phase_j.launcher', '--config', str(config_path), '--require-manifest']
                prepare_output(self.root)
                close_job(self.job)
                self.job = create_job()
                self.child = subprocess.Popen(command, cwd=self.root / 'app', stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=self.preview_environment(),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                assign_job(self.job, self.child)
                threading.Thread(target=self._watch_child, args=(self.child,), daemon=True).start()
            except Exception:
                failure = traceback.format_exc()
                try:
                    self._discard_failed_child()
                finally:
                    self.close_preview()
                    self.record_failure('startup_failure', failure)

    def _discard_failed_child(self) -> None:
        """Jobへの割当より前に失敗した子も、パイプとともに回収する。"""
        child, job = self.child, self.job
        self.child, self.job, self.inner_port = None, None, 0
        try:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=CHILD_EXIT_TIMEOUT_SECONDS)
        finally:
            if child is not None and child.stdout is not None:
                child.stdout.close()
            close_job(job)

    def request_start(self) -> None:
        """HTTPへすぐ戻し、入力の待機中も停止ボタンを使えるようにする。"""
        with self.control_lock:
            if self.input_waiting or (self.child and self.child.poll() is None):
                return
            if self.finished.is_set():
                return
            self.stopping = False
            self.cancel_start.clear()
            self.retry_devices = None
            self.input_waiting = True
            self.start_thread = threading.Thread(target=self._wait_for_input, daemon=True)
            self.start_thread.start()

    def start_autostart(self) -> None:
        """ログイン時はOBSだけを低頻度で調べ、停止まで待つ。"""
        with self.control_lock:
            if self.stopping or self.finished.is_set() or self.input_waiting or self.child:
                return
            self.cancel_start.clear()
            self.retry_devices = None
            self.autostart_waiting = True
            self.input_waiting = True
        try:
            self._wait_for_input()
        finally:
            self.autostart_waiting = False

    def _wait_for_input(self, deadline: float | None = None) -> None:
        """12コマの検索を繰り返し、入力確定または停止まで待つ。"""
        from .distribution_input import MESSAGES
        try:
            while not self.cancel_start.is_set():
                with self.control_lock:
                    if self.cancel_start.is_set():
                        return
                    self.start()
                if self.child or self.error or (not self.settings['auto_input'] and not self.autostart_waiting):
                    return
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return
                reason = self.input_probe.get('reason')
                if self.autostart_waiting and reason not in ('feedback', 'wrong_size'):
                    self.input_probe.update(reason='waiting', message='OBS の仮想カメラが始まるのを待っています / Waiting for OBS Virtual Camera to start')
                elif reason not in ('obs_stopped', 'feedback', 'wrong_size'):
                    self.input_probe['message'] = MESSAGES['waiting']
                delay = AUTOSTART_RETRY_SECONDS if remaining is None else min(AUTOSTART_RETRY_SECONDS, remaining)
                self.cancel_start.wait(delay)
        except Exception:
            self.record_failure('startup_failure', traceback.format_exc())
        finally:
            self.input_waiting = False

    def _watch_child(self, child: subprocess.Popen | None = None) -> None:
        child = child or self.child
        path = self.logs / 'runtime.log'
        with path.open('ab') as stream:
            for line in iter(child.stdout.readline, b''):
                if stream.tell() > MAX_RUNTIME_LOG_BYTES:
                    stream.seek(0)
                    stream.truncate()
                stream.write(line)
                stream.flush()
        code = child.wait()
        with self.control_lock:
            if child is not self.child or self.stopping or self.cancel_start.is_set():
                return
            if code != 0 and self._schedule_restart(code):
                retry = True
            else:
                retry = False
                kind = 'abnormal_exit' if self.latest else 'startup_failure'
                self.record_failure(kind, '\n'.join(tail(path)), code)
        if retry:
            self._restart_after_exit()

    def _schedule_restart(self, code: int) -> bool:
        """直近10分の異常終了だけを数え、上限後は通常の障害表示へ戻す。"""
        now = time.monotonic()
        self.restart_times = [stamp for stamp in self.restart_times
                              if now - stamp < RESTART_WINDOW_SECONDS]
        allowed = len(self.restart_times) < RESTART_LIMIT
        if allowed:
            self.restart_times.append(now)
        with (self.logs / 'runtime.log').open('a', encoding='utf-8') as stream:
            stream.write(f'\n解析再開: reason=abnormal_exit exit_code={code} '
                         f'attempt={len(self.restart_times)}/{RESTART_LIMIT} allowed={allowed}\n')
        self.restarting = allowed
        if allowed:
            self.error = {}
            self.latest = {}
            self._discard_failed_child()
        return allowed

    def _restart_after_exit(self) -> None:
        """待機中の停止を優先し、再検索から解析を再開する。"""
        delay = RESTART_DELAY_SECONDS * len(self.restart_times)
        if self.cancel_start.wait(delay):
            self.restarting = False
            return
        with self.control_lock:
            if self.stopping or self.cancel_start.is_set() or self.finished.is_set():
                self.restarting = False
                return
            self.input_waiting = True
        self._wait_for_input()
        if self.error:
            self.restarting = False

    def record_failure(self, kind: str, stack: str, code: int | None = None) -> None:
        locations = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', stack)
        location = ':'.join(locations[-1]) if locations else 'launcher.pipeline'
        self.error = dict(type=kind, location=location, stack_trace=stack, exit_code=code)
        self.store.save([self.report()])
        print(f'失敗 / Failure: {kind}. ログを送る / Send logs: http://127.0.0.1:{self.port}/settings', flush=True)

    def report_summary(self) -> dict[str, Any] | None:
        with self.lock:
            return (self.numeric_summary.snapshot(time.time() - self.started_at)
                    if feedback_site_url() else None)

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
            diagnostics, tail(self.logs / 'runtime.log'), tail(self.root / 'install.log'),
            numeric_summary=self.report_summary())

    def check_update(self) -> None:
        """更新一覧の取得のみ。インストールIDや利用情報は付けない。"""
        with self.lock:
            if self.update_checked:
                return
            self.update_checked = True
        try:
            from packaging.version import Version
            request = Request(self.settings['releases_url'], headers={'User-Agent': SITE_USER_AGENT})
            with build_opener(NoRedirect()).open(request, timeout=5) as response:
                data = json.loads(response.read(64 * 1024))
            version = data.get('version') if isinstance(data, dict) else None
            if not isinstance(version, str) or not version:
                self.update = dict(available=False, status='unavailable')
                return
            self.update = dict(available=Version(version) > Version(self.version['app']), version=version)
        except Exception:
            self.update = dict(available=False, status='unavailable')

    def stop(self, final: bool = True) -> None:
        self.cancel_start.set()
        self.restarting = False
        self.stopping = True
        if final:
            self.finished.set()
        with self.control_lock:
            close_job(self.job)
            self.job = None
            if self.child and self.child.poll() is None:
                if os.name == 'nt':
                    subprocess.run([system_tool('taskkill'), '/PID', str(self.child.pid), '/T', '/F'],
                        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    self.child.terminate()
                self.child.wait(timeout=10)
            self.child = None
            self.inner_port = 0
            self.close_preview()
            self.latest = {}
            self.numeric_summary.last_match = None
            self.last_update = None
            if final:
                (self.logs / 'session.running').unlink(missing_ok=True)


    def begin_session(self) -> None:
        """強制終了の痕跡は次回起動時にローカル記録だけを作る。"""
        marker = self.logs / 'session.running'
        if marker.exists():
            self.record_failure('abnormal_exit', 'Previous application session did not close cleanly')
        marker.write_text(str(time.time()), encoding='ascii')
