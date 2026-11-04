"""配布版のトレイ。ログ送信はブラウザの内容確認画面を開くだけ。"""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
import webbrowser
from .distribution_windows import system_tool

POLL_SECONDS = 2


def status(runtime: Any) -> str:
    if runtime.error:
        return 'エラー / Error'
    if getattr(runtime, 'input_waiting', False) or getattr(runtime, 'restarting', False):
        return '試合待ち / Waiting for a match'
    if runtime.scanning:
        return '入力確認中 / Checking input'
    if not runtime.child or runtime.child.poll() is not None:
        return '入力なし / No input'
    display = runtime.latest.get('display', {})
    phase = display.get('input_status')
    if phase in ('overlay_feedback', 'obs_stopped'):
        return 'エラー / Error'
    if phase in (None, 'checking', 'verifying'):
        return '入力なし / No input'
    playing = phase == 'ready' and display.get('match_phase') == 'playing'
    return '判定中 / Analyzing' if playing else '試合待ち / Waiting for a match'


def copy_url(url: str) -> None:
    """Windowsのクリップボードへ、シェル文字列を介さずコピーする。"""
    import subprocess
    subprocess.run([system_tool('clip')], input=url.encode('utf-16le'), check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def start_tray(runtime: Any, stop: Callable, open_window: Callable | None = None) -> Any:
    if os.name != 'nt':
        return None
    import pystray
    from PIL import Image
    url = f'http://127.0.0.1:{runtime.port}/'
    menu = pystray.Menu(
        pystray.MenuItem('開く / Open', lambda: open_window() if open_window else webbrowser.open(url + 'settings'), default=True),
        pystray.MenuItem('URL をコピー / Copy URL', lambda: copy_url(url)),
        pystray.MenuItem('ログを送る / Send logs', lambda: open_window('/settings#logs') if open_window else webbrowser.open(url + 'settings#logs')),
        pystray.MenuItem('終了 / Exit', stop))
    icon = pystray.Icon('puyoZeus Live', Image.open(Path(__file__).parent / 'brand/app_logo.ico'),
                        'puyoZeus Live', menu)
    threading.Thread(target=icon.run, daemon=True).start()
    def refresh() -> None:
        while not runtime.finished.is_set():
            icon.title = 'puyoZeus Live — ' + status(runtime)
            runtime.finished.wait(POLL_SECONDS)
    threading.Thread(target=refresh, daemon=True).start()
    return icon
