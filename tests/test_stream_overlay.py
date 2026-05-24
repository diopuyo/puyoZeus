"""
stream_overlay.py のテスト

StreamState の並行操作と HTTP サーバの基本動作を検証する。
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from src.analyzer import Analyzer
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_RED, Board
from src.stream_overlay import (
    DEFAULT_HOST,
    SSE_EVENT_UPDATE,
    StreamOverlayServer,
    StreamState,
)


# ============================
# ヘルパー
# ============================


def make_result():
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    grid[12][3] = COLOR_RED
    b1 = Board.from_list(grid)
    b2 = Board.from_list(
        [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    )
    return Analyzer().analyze_boards(b1, b2)


@pytest.fixture
def free_port() -> int:
    """OS から空きポートを取得する。"""
    import socket
    with socket.socket() as s:
        s.bind((DEFAULT_HOST, 0))
        return s.getsockname()[1]


# ============================
# StreamState
# ============================


class TestStreamState:
    def test_initial_latest_is_none(self):
        state = StreamState()
        assert state.latest() is None

    def test_update_stores_payload(self):
        state = StreamState()
        state.update(make_result())
        latest = state.latest()
        assert latest is not None
        assert "score" in latest

    def test_subscribe_receives_future_updates(self):
        state = StreamState()
        q = state.subscribe()
        state.update(make_result())
        payload = q.get(timeout=1.0)
        assert "score" in payload

    def test_subscribe_receives_current_state_immediately(self):
        state = StreamState()
        state.update(make_result())
        q = state.subscribe()
        # 現在の state を即時受信
        payload = q.get(timeout=1.0)
        assert "score" in payload

    def test_unsubscribe_removes_subscriber(self):
        state = StreamState()
        q = state.subscribe()
        assert state.subscriber_count() == 1
        state.unsubscribe(q)
        assert state.subscriber_count() == 0

    def test_multiple_subscribers_all_receive(self):
        state = StreamState()
        q1 = state.subscribe()
        q2 = state.subscribe()
        state.update(make_result())
        p1 = q1.get(timeout=1.0)
        p2 = q2.get(timeout=1.0)
        assert p1 == p2

    def test_concurrent_updates_are_safe(self):
        state = StreamState()
        result = make_result()

        def worker():
            for _ in range(50):
                state.update(result)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert state.latest() is not None


# ============================
# StreamOverlayServer - HTTP
# ============================


class TestStreamOverlayServerHttp:
    def _get(self, port: int, path: str, timeout: float = 2.0) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection(DEFAULT_HOST, port, timeout=timeout)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def test_html_served_at_root(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            status, body = self._get(free_port, "/")
            assert status == 200
            assert b"EventSource" in body
        finally:
            srv.stop()

    def test_latest_returns_json(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            # 未投入なら空オブジェクト
            status, body = self._get(free_port, "/latest")
            assert status == 200
            assert json.loads(body) == {}

            # 更新後
            srv.state.update(make_result())
            status, body = self._get(free_port, "/latest")
            data = json.loads(body)
            assert "score" in data
        finally:
            srv.stop()

    def test_unknown_path_404(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            status, _ = self._get(free_port, "/nope")
            assert status == 404
        finally:
            srv.stop()

    def test_address_returns_port(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            host, port = srv.address()
            assert port == free_port
        finally:
            srv.stop()

    def test_double_start_raises(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            with pytest.raises(RuntimeError):
                srv.start()
        finally:
            srv.stop()


# ============================
# StreamOverlayServer - SSE
# ============================


class TestStreamOverlayServerSse:
    def test_events_stream_receives_update(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            # SSE 接続は生でソケット読みで検証する
            conn = http.client.HTTPConnection(
                DEFAULT_HOST, free_port, timeout=3.0,
            )
            conn.request("GET", "/events")
            resp = conn.getresponse()
            assert resp.status == 200
            # 接続確立後に update
            time.sleep(0.2)
            srv.state.update(make_result())

            # 少量読み出し (タイムアウトあり)
            resp.fp.readline()  # event: analysis\n を期待
            received_event = resp.fp.readline().decode(errors="ignore")
            # 何かイベントが届くこと
            conn.close()
            assert "data:" in received_event or received_event.startswith("data")
        finally:
            srv.stop()
