"""設定ページを専用の窓に表示し、閉じる操作では通知領域へ隠す。"""
from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any
import webbrowser

TITLE = 'puyoZeus Live'
WINDOW_WIDTH, WINDOW_HEIGHT = 1100, 800
POLL_SECONDS = 0.2
GUI_READY_TIMEOUT = 5
REQUEST_FILE = 'window.request'
LOG = logging.getLogger(__name__)


def request_existing(root: Path) -> None:
    """同じ展開先の起動済みプロセスだけに表示を依頼する。"""
    (root / 'logs' / REQUEST_FILE).touch()


class AppWindow:
    def __init__(self, url: str, root: Path, enabled: bool,
                 hidden: bool = False) -> None:
        self.url, self.root = url, root
        self.default_url = url
        self.enabled, self.hidden = enabled, hidden
        self.window: Any = None
        self.finished = threading.Event()
        self.ready = threading.Event()
        self.pending_open = threading.Event()
        self.exiting = False
        self.pending_path: str | None = None

    def open(self, path: str | None = None) -> None:
        if path is not None:
            from urllib.parse import urljoin
            self.pending_path = urljoin(self.default_url, path)
        elif self.pending_path is None:
            self.pending_path = self.default_url
        if self.enabled and not self.ready.is_set():
            self.pending_open.set()
        elif self.window is not None:
            if self.pending_path is not None:
                self.window.load_url(self.pending_path)
                self.pending_path = None
            self.window.show()
            self.window.restore()
        else:
            webbrowser.open(self.pending_path or self.default_url)
            self.pending_path = None

    def closing(self) -> bool:
        if self.exiting:
            return True
        self.window.hide()
        return False

    def shown(self) -> None:
        self.ready.set()
        if self.finished.is_set():
            self.exiting = True
            self.window.destroy()

    def set_icon(self) -> None:
        """pywebview 6.1のicon引数はWinForms非対応なのでnativeで設定する。"""
        from System.Drawing import Icon
        self.window.native.Icon = Icon(str(Path(__file__).parent / 'brand/app_logo.ico'))

    def watch(self) -> None:
        """終了と二重起動を監視する。GUI初期化前の要求も保持する。"""
        request = self.root / 'logs' / REQUEST_FILE
        previous = 0
        while not self.finished.wait(POLL_SECONDS):
            try:
                modified = request.stat().st_mtime_ns
            except FileNotFoundError:
                modified = previous
            if modified != previous:
                previous = modified
                self.pending_open.set()
            if self.ready.is_set() and self.pending_open.is_set():
                self.pending_open.clear()
                self.open()
        self.exiting = True
        if self.window is not None and self.ready.wait(GUI_READY_TIMEOUT):
            self.window.destroy()

    def run(self) -> None:
        if self.finished.is_set():
            return
        threading.Thread(target=self.watch, daemon=True).start()
        if self.enabled:
            try:
                import webview
                # 6.1はEdge指定でも不在時にMSHTMLへ降りるため、作成前に確認する。
                if webview.initialize('edgechromium').renderer != 'edgechromium':
                    raise RuntimeError('Edge WebView2 がありません')
                if self.finished.is_set():
                    return
                self.window = webview.create_window(
                    TITLE, self.url, width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                    hidden=self.hidden)
                self.window.events.closing += self.closing
                self.window.events.before_show += self.set_icon
                self.window.events.shown += self.shown
                if self.finished.is_set():
                    return
                webview.start(func=None if self.hidden else self.open, gui='edgechromium',
                              icon=str(Path(__file__).parent / 'brand/app_logo.ico'))
                return
            except Exception:
                LOG.exception('専用の窓を開けないためブラウザに切り替えます')
                self.window = None
                self.enabled = False
        self.ready.set()
        if not self.hidden and not self.finished.is_set():
            webbrowser.open(self.url)
        self.finished.wait()
