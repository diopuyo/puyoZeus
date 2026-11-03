"""インストーラーとzip共通の起動口。終了時は自身の子だけを止める。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import threading
from typing import BinaryIO

from .distribution_config import available_port
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
    parser.add_argument('--remove-shortcuts', action='store_true')
    return parser.parse_args()


def run_application(runtime: DistributionRuntime, options: argparse.Namespace) -> None:
    """HTTPは別スレッド、WindowsのGUIはメインスレッドで動かす。"""
    migrate_shortcuts(runtime.root, runtime.settings)
    port = available_port(options.port or runtime.settings['port'])
    server = make_server(runtime, port)
    runtime.begin_session()
    url = f'http://127.0.0.1:{runtime.port}'
    page = '/setup' if options.setup or not runtime.settings['completed'] else '/settings'
    window = AppWindow(url + page, runtime.root,
                       runtime.settings['distribution_app_window'], options.no_browser)
    def stop(signum: int = 0, frame: object = None) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()
    def serve() -> None:
        try:
            server.serve_forever()
        finally:
            window.finished.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from .distribution_tray import start_tray
    tray = start_tray(runtime, stop, window.open)
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        if runtime.settings['completed'] and not options.setup:
            threading.Thread(target=runtime.start, daemon=True).start()
        window.run()
    finally:
        server.shutdown()
        worker.join()
        window.finished.set()
        runtime.stop()
        if tray:
            tray.stop()
        server.server_close()


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
        run_application(runtime, options)
    finally:
        instance.close()


if __name__ == '__main__':
    main()
