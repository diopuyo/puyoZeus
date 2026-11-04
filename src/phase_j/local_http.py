"""常駐窓とOBSからの同一オリジン接続を受けるHTTP共通処理。"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket


class LocalHTTPHandler(BaseHTTPRequestHandler):
    def handle(self) -> None:
        """要求の読み取りや表示資産の送信中に相手が切った場合も静かに終える。"""
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def version_string(self) -> str:
        return 'puyoZeus Live'

    def end_headers(self) -> None:
        self.send_header('Content-Security-Policy', "frame-ancestors 'self'")
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        super().end_headers()

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        host = f'127.0.0.1:{self.server.server_address[1]}'
        valid = (self.headers.get_all('Host') == [host]
                 and self.headers.get('Origin', f'http://{host}') == f'http://{host}'
                 and self.headers.get('Sec-Fetch-Site') != 'cross-site')
        if not valid:
            self.send_error(403, 'Local access only')
        return valid


class LocalHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()
