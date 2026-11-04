"""インストーラーとzip共通の起動口。終了時は自身の子だけを止める。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from typing import Any, BinaryIO

from .distribution_http import make_server
from .distribution_runtime import DistributionRuntime
from .distribution_windows import migrate_shortcuts, shortcuts
from .distribution_window import AppWindow, request_existing


def acquire_instance(root: Path) -> BinaryIO | None:
    """同じ展開先の二重起動を防ぎ、プロセス終了時にOSがロックを解放する。"""
    stream = (root / 'logs/instance.lock').open('a+b')
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        return None
    return stream


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--port', type=int)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--autostart', action='store_true')
    parser.add_argument('--remove-shortcuts', action='store_true')
    return parser.parse_args()


EXIT_TIMEOUT_SECONDS = 10


def active_server(server: Any) -> Any:
    while getattr(server, 'next_server', None) is not None:
        server = server.next_server
    return server


def serve_application(server: Any, window: AppWindow, runtime: DistributionRuntime,
                      deadline: threading.Timer) -> None:
    """切替時以外は直ちに待受を閉じ、GUIが止まらない場合にも終了する。"""
    try:
        while True:
            server.serve_forever()
            replacement = getattr(server, 'next_server', None)
            server.server_close()
            if replacement is None:
                break
            server = replacement
            window.default_url = f'http://127.0.0.1:{runtime.port}/settings'
            window.url = window.default_url
    finally:
        deadline.start()
        window.finished.set()
        server.server_close()


def run_application(runtime: DistributionRuntime, options: argparse.Namespace) -> None:
    """HTTPは別スレッド、WindowsのGUIはメインスレッドで動かす。"""
    try:
        migrate_shortcuts(runtime.root, runtime.settings)
    except (OSError, subprocess.SubprocessError):
        runtime.shortcut_warning = '自動起動の登録を更新できません / Could not update startup shortcuts'
    server = make_server(runtime, options.port or runtime.settings['port'])
    runtime.begin_session()
    if runtime.settings['check_updates']:
        threading.Thread(target=runtime.check_update, daemon=True).start()
    url = f'http://127.0.0.1:{runtime.port}'
    page = '/settings'
    window = AppWindow(url + page, runtime.root,
                       runtime.settings['distribution_app_window'], options.no_browser or getattr(options, 'autostart', False))
    def stop(signum: int = 0, frame: object = None) -> None:
        threading.Thread(target=active_server(server).shutdown, daemon=True).start()
    deadline = threading.Timer(EXIT_TIMEOUT_SECONDS, lambda: os._exit(0))
    deadline.daemon = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from .distribution_tray import start_tray
    tray = start_tray(runtime, stop, window.open)
    worker = threading.Thread(target=serve_application, args=(server, window, runtime, deadline), daemon=True)
    worker.start()
    from .distribution_obs import refresh_on_startup
    refresh_on_startup(runtime.settings)
    try:
        if runtime.settings['completed'] and not options.setup and getattr(options, 'autostart', False):
            threading.Thread(target=runtime.start_autostart, daemon=True).start()
        window.run()
    finally:
        active_server(server).shutdown()
        worker.join()
        window.finished.set()
        runtime.stop()
        if tray:
            tray.stop()
        server.server_close()
        deadline.cancel()


def main() -> None:
    options = parse_options()
    if sys.stdout is None or sys.stderr is None:
        (options.root / 'logs').mkdir(parents=True, exist_ok=True)
        stream = (options.root / 'logs/launcher.log').open('a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = stream
    runtime = DistributionRuntime(options.root)
    if options.remove_shortcuts:
        shortcuts(runtime.root, runtime.settings, remove=True)
        return
    instance = acquire_instance(runtime.root)
    if instance is None:
        request_existing(runtime.root)
        print('同じフォルダのpuyoZeus Liveは起動済みです / Already running', flush=True)
        return
    try:
        from .distribution_memory import watch_memory
        threading.Thread(target=watch_memory, args=(runtime.logs, runtime.finished), daemon=True).start()
        run_application(runtime, options)
    finally:
        runtime.finished.set()
        instance.close()


if __name__ == '__main__':
    main()
