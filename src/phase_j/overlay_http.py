"""表示ページ専用の固定ルート。設定ファイルや任意パスは公開しない。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .overlay_config import configured_html

HTTP_OK = 200
BINARY_ASSETS = {'/brand/app_logo.png': ('brand/app_logo.png', 'image/png'),
    '/favicon.ico': ('brand/app_logo.ico', 'image/x-icon'),
    '/mascot/overlay.webp': ('brand/overlay_mascot.webp', 'image/webp'),
    '/mascot/battle-still.webp': ('brand/overlay_mascot_still.webp', 'image/webp')}
BINARY_ASSETS.update({f'/mascot/{band}.webp': (f'brand/mascot_{band}.webp', 'image/webp')
                      for band in (10, 30, 50, 70, 90)})
BINARY_ASSETS.update({f'/fonts/{name}.woff2': (f'fonts/{name}.woff2', 'font/woff2')
                      for name in ('DelaGothicOne-Regular', 'NotoSansJP')})
ASSETS = {'/overlay_expressions.js': ('overlay_expressions.js', 'text/javascript'),
    '/distribution_ux.js': ('distribution_ux.js', 'text/javascript'),
    '/distribution_ux.css': ('distribution_ux.css', 'text/css'),
    '/overlay_24h.js': ('overlay_24h.js', 'text/javascript'),
    '/overlay_battle.css': ('overlay_battle.css', 'text/css'),
    '/overlay_battle.js': ('overlay_battle.js', 'text/javascript'),
    '/overlay_winrate_switch.js': ('overlay_winrate_switch.js', 'text/javascript'),
    '/overlay_ama.js': ('overlay_ama.js', 'text/javascript'),
    '/overlay_v49f.js': ('overlay_v49f.js', 'text/javascript'),
    '/overlay_x_prefire.js': ('overlay_x_prefire.js', 'text/javascript'),
    '/overlay_position.js': ('overlay_position.js', 'text/javascript'),
    '/overlay_contents.js': ('overlay_contents.js', 'text/javascript'),
    '/overlay_i18n.js': ('overlay_i18n.js', 'text/javascript'),
    '/overlay_mascot.js': ('overlay_mascot.js', 'text/javascript'),
    '/distribution_settings.js': ('distribution_settings.js', 'text/javascript'),
    '/overlay_themes.css': ('overlay_themes.css', 'text/css'),
    '/overlay_themes.js': ('overlay_themes.js', 'text/javascript'),
    '/settings': ('overlay_settings.html', 'text/html'),
    '/overlay_options.js': ('overlay_options.js', 'text/javascript'),
    '/overlay_layout.js': ('overlay_layout.js', 'text/javascript'),
    '/overlay_debug.js': ('overlay_debug.js', 'text/javascript'),
    '/overlay_settings.js': ('overlay_settings.js', 'text/javascript'),
    '/overlay_layout.css': ('overlay_layout.css', 'text/css')}


def x_prefire_display() -> bool:
    """暫定 X の表示フラグ (live_x_prefire)。判定は段3 と同じ `flags()` を使う。"""
    from .live_v49f_prefire import flags
    return flags()[1]


class OverlayPagesMixin:
    """既存SSEハンドラに表示用GETだけを足す。"""

    overlay_options: dict[str, Any]
    overlay_html: str

    def do_GET(self) -> None:
        path = self.path.split('?')[0]
        if path == '/input-preview.jpg':
            from .live_preview import current_image
            self._serve_asset(current_image(), 'image/jpeg')
        elif path in BINARY_ASSETS:
            filename, mime = BINARY_ASSETS[path]
            self._serve_asset((Path(__file__).parent / filename).read_bytes(), mime)
        elif path.startswith('/mascot/'):
            from .mascot_assets import mascot_asset
            try:
                data = mascot_asset(Path(__file__).parent / 'brand', path.removeprefix('/mascot/'))
                self._serve_asset(data, 'image/webp')
            except FileNotFoundError:
                self.send_error(404)
        elif path in ASSETS:
            filename, mime = ASSETS[path]
            text = Path(__file__).with_name(filename).read_text(encoding='utf-8')
            self._serve_asset(configured_html(text, self.overlay_options), mime)
        else:
            super().do_GET()

    def _serve_html(self) -> None:
        import os
        html = self.overlay_html
        unified = os.environ.get('PUYO_LIVE_24H') == '1'
        switch = os.environ.get('PUYO_LIVE_WINRATE_SWITCH') == '1' and not unified
        if unified:
            html += '\n<script src="overlay_24h.js"></script>\n'
        if switch:
            html += '\n<script src="overlay_winrate_switch.js"></script>\n'
        if not unified and not switch and os.environ.get('PUYO_LIVE_V49F') == '1':
            html += '\n<script src="overlay_v49f.js"></script>\n'
        if not unified and not switch and os.environ.get('PUYO_LIVE_AMA_PLAIN') == '1':
            html += '\n<script src="overlay_ama.js"></script>\n'
        if not unified and x_prefire_display():
            # 段3 の暫定 X の 1 行。表示フラグが OFF のときは script 自体を足さず、ページを変えない。
            html += '\n<script src="overlay_x_prefire.js"></script>\n'
        self._serve_asset(configured_html(html, self.overlay_options), 'text/html')

    def _serve_asset(self, body: bytes, mime: str) -> None:
        self.send_response(HTTP_OK)
        self.send_header('Content-Type', mime if mime.startswith(('image/', 'font/')) else mime + '; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
