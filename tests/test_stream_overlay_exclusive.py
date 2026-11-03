"""HTTP 待受の排他 bind (exclusive_bind) を検査する。

Windows の ThreadingHTTPServer は既定で SO_REUSEADDR を立てるため同じ port への 2 つ目の bind が黙って成功し、
接続は先の 1 つにしか届かない (2026-10-24 の二重起動の原因)。exclusive_bind=True は 2 つ目の bind を失敗させる。
既定 (exclusive_bind=False) は従来どおりで、OFF の挙動は変えない。
"""
from __future__ import annotations

import http.client
import socket

import pytest

from src.stream_overlay import DEFAULT_HOST, StreamOverlayServer

WINDOWS = hasattr(socket, 'SO_EXCLUSIVEADDRUSE')


@pytest.fixture
def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((DEFAULT_HOST, 0))
        return probe.getsockname()[1]


def fetch(port: int, path: str = '/latest') -> tuple[int, bytes]:
    conn = http.client.HTTPConnection(DEFAULT_HOST, port, timeout=3.0)
    try:
        conn.request('GET', path)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


@pytest.mark.skipif(not WINDOWS, reason='Windows 専用の実測: 既定の bind が二重に成功する')
def test_default_bind_silently_allows_a_second_server_on_windows(free_port: int) -> None:
    """不具合の原因の固定。これが変わったら (成功しなくなったら) 二重起動の検出の前提を見直す。"""
    first, second = StreamOverlayServer(port=free_port), StreamOverlayServer(port=free_port)
    first.start()
    try:
        second.start()     # 例外にならない = 2 つ目は誰にも配信できないまま動く
        second.stop()
    finally:
        first.stop()


def test_exclusive_second_bind_fails_and_leaves_first_serving(free_port: int) -> None:
    first = StreamOverlayServer(port=free_port, exclusive_bind=True)
    second = StreamOverlayServer(port=free_port, exclusive_bind=True)
    first.start()
    try:
        with pytest.raises(OSError):
            second.start()
        assert second._server is None and second._thread is None        # 失敗した側は何も残さない
        assert fetch(free_port)[0] == 200                               # 先の側は配信を続ける
    finally:
        first.stop()


@pytest.mark.skipif(not WINDOWS, reason='Windows 専用: 排他を知らない旧版 (既定の bind) が後から来る場合')
def test_exclusive_holder_also_blocks_a_default_bind_started_later(free_port: int) -> None:
    first = StreamOverlayServer(port=free_port, exclusive_bind=True)
    late_default = StreamOverlayServer(port=free_port)
    first.start()
    try:
        with pytest.raises(OSError):
            late_default.start()
    finally:
        first.stop()


def test_bind_then_start_serves_and_rebinds_right_after_stop(free_port: int) -> None:
    """bind() だけを先に行える (認識を始める前に失敗させるため)。停止直後の再 bind も通る。"""
    server = StreamOverlayServer(port=free_port, exclusive_bind=True)
    server.bind()
    server.bind()                                    # 冪等
    assert server._thread is None and server.address() == (DEFAULT_HOST, free_port)
    server.start()
    try:
        assert fetch(free_port)[0] == 200
        with pytest.raises(RuntimeError):
            server.start()                           # 従来どおり起動中の二重 start は拒否
    finally:
        server.stop()
    again = StreamOverlayServer(port=free_port, exclusive_bind=True)
    again.start()                                    # 接続済みの port でも停止直後に取り直せる
    try:
        assert fetch(free_port)[0] == 200
    finally:
        again.stop()


def test_default_server_is_unchanged_and_start_alone_still_works(free_port: int) -> None:
    server = StreamOverlayServer(port=free_port)
    assert server._exclusive_bind is False
    server.start()
    try:
        assert fetch(free_port)[0] == 200
    finally:
        server.stop()


def test_stop_without_start_after_bind_releases_the_port(free_port: int) -> None:
    server = StreamOverlayServer(port=free_port, exclusive_bind=True)
    server.bind()
    server.stop()
    assert server._server is None
    with socket.socket() as probe:                   # 解放されていれば素の bind も通る
        probe.bind((DEFAULT_HOST, free_port))


def test_live_overlay_server_follows_the_env_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_instance_guard import ALLOW_SECOND_INSTANCE_ENV
    from src.phase_j.live_publish import LiveOverlayServer, LiveStreamState
    monkeypatch.delenv(ALLOW_SECOND_INSTANCE_ENV, raising=False)
    assert LiveOverlayServer(LiveStreamState(), DEFAULT_HOST, 0)._exclusive_bind is True       # 既定: 排他
    monkeypatch.setenv(ALLOW_SECOND_INSTANCE_ENV, '1')
    assert LiveOverlayServer(LiveStreamState(), DEFAULT_HOST, 0)._exclusive_bind is False     # 許可時は従来
    assert LiveOverlayServer(LiveStreamState(), DEFAULT_HOST, 0, exclusive_bind=True)._exclusive_bind is True
