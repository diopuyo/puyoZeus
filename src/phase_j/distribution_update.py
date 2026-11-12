"""画面の操作と別プロセス更新を接続する。初期化だけでは通信しない。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Callable
import tempfile
import uuid

from . import update_download as fetch
from . import update_transaction as transaction


def enabled() -> bool:
    from src.production_config import DISTRIBUTION_UPDATE_BUTTON
    return os.environ.get('PUYO_UPDATE_BUTTON', str(int(DISTRIBUTION_UPDATE_BUTTON))) == '1'


class UpdateController:
    def __init__(self, runtime: Any, local_origin: str | None = None) -> None:
        self.runtime = runtime
        self.local_origin = local_origin
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.info: dict = {}
        self.state: dict = dict(status='idle', received=0, total=0)
        self.stage: Path | None = None
        self.result_mtime = 0
        self.read_result()

    def read_result(self) -> None:
        """別プロセスの完了通知も起動後の画面へ反映する。"""
        result = self.runtime.root / transaction.RESULT
        try:
            mtime = result.stat().st_mtime_ns
            if mtime != self.result_mtime:
                self.state = json.loads(result.read_text(encoding='utf8'))
                self.result_mtime = mtime
        except (OSError, ValueError):
            pass

    def snapshot(self) -> dict:
        with self.lock:
            if not self.busy():
                self.read_result()
            return dict(self.state)

    def busy(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def check(self) -> None:
        with self.lock:
            if self.busy():
                raise ValueError('更新の処理中です / Update is in progress')
            self.state = dict(status='checking')
            self.thread = threading.Thread(target=self._check, daemon=True)
            self.thread.start()

    def _check(self) -> None:
        self.runtime.check_update(force=True)
        with self.lock:
            self.info = dict(self.runtime.update)
            self.state = dict(status=self.info.get('status', 'unavailable'))

    def start(self, shutdown: Callable[[], None]) -> None:
        with self.lock:
            if self.busy() or self.state.get('status') == 'restarting':
                raise ValueError('更新の処理中です / Update is in progress')
            self.info = dict(self.runtime.update)
            if not self.info.get('available'):
                raise ValueError('先に更新を確認してください / Check for updates first')
            self.cancel.clear()
            self.state = dict(status='downloading', received=0, total=0)
            self.thread = threading.Thread(target=self._run, args=(shutdown,), daemon=True)
            self.thread.start()

    def abort(self) -> None:
        with self.lock:
            if self.state.get('status') in ('downloading', 'extracting', 'preparing'):
                self.cancel.set()

    def progress(self, received: int, total: int) -> None:
        with self.lock:
            self.state.update(received=received, total=total)

    def _run(self, shutdown: Callable[[], None]) -> None:
        handed_off = False
        try:
            root = self.runtime.root
            self.stage = Path(tempfile.mkdtemp(prefix=f'.{root.name}-update-', dir=root.parent))
            archive = self.stage / 'download.zip'
            fetch.download(self.info, archive, self.cancel, self.progress, self.local_origin)
            self.state = dict(status='extracting')
            payload = fetch.extract(archive, self.stage, self.info['version'], self.cancel)
            self.state = dict(status='preparing')
            self._prepare_worker(payload)
            if self.cancel.is_set():
                raise fetch.Cancelled()
            self._spawn_worker(payload)
            handed_off = True
            self.state = dict(status='restarting')
            shutdown()
        except fetch.Cancelled:
            self.state = dict(status='cancelled')
        except Exception as error:
            self.state = dict(status='failed', message=str(error))
        finally:
            if not handed_off and self.stage:
                # 作成した同じ親の専用フォルダだけを片付ける。
                if self.stage.resolve().parent == self.runtime.root.resolve().parent:
                    shutil.rmtree(transaction.filesystem_path(self.stage), ignore_errors=True)

    def _prepare_worker(self, payload: Path) -> None:
        assert self.stage is not None
        for relative in (*transaction.BOOTSTRAP, 'python/python.exe'):
            if not (payload / relative).is_file():
                raise ValueError('更新用ファイルが不足しています / Update files are missing')
        shutil.copyfile(Path(transaction.__file__), self.stage / 'worker.py')
        transaction.prepare(self.runtime.root, payload, self.stage, self.info['version'], uuid.uuid4().hex)

    def _spawn_worker(self, payload: Path) -> None:
        assert self.stage is not None
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0
        with (self.stage / 'worker.log').open('ab') as log:
            subprocess.Popen([str(payload / 'python/python.exe'), str(self.stage / 'worker.py'),
                              str(self.stage / 'transaction.json'), str(os.getpid())],
                             cwd=self.stage, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             creationflags=flags, close_fds=True)


def acknowledge_start(runtime: Any) -> None:
    token = os.environ.get('PUYO_UPDATE_TOKEN')
    if token:
        transaction.write_json(runtime.root / transaction.HEALTH,
                               dict(token=token, version=runtime.version['app'],
                                    port=runtime.port, pid=os.getpid()))
