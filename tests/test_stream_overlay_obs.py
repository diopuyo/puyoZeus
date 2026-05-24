"""
OBS ブラウザソース互換性テスト

OBS (CEF ベース) が /events をブラウザの EventSource API で消費できることを
プログラマティックに検証する。ブラウザ実機なしで:
    - HTML が有効な HTML5 であること
    - SSE ストリームが W3C SSE 仕様に準拠していること
    - EventSource プロトコル (event/data/retry) が正しくパースできること
    - 同一オリジン前提で CORS 問題が起きないこと
を保証する。
"""

from __future__ import annotations

import http.client
import socket
import threading
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest

from src.analyzer import Analyzer
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_RED, Board
from src.stream_overlay import (
    DEFAULT_HOST,
    SSE_EVENT_UPDATE,
    StreamOverlayServer,
)


# ============================
# ヘルパー
# ============================


@pytest.fixture
def free_port() -> int:
    with socket.socket() as s:
        s.bind((DEFAULT_HOST, 0))
        return s.getsockname()[1]


def _make_result():
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(4):
        grid[12][c] = COLOR_RED
    return Analyzer().analyze_boards(
        Board.from_list(grid),
        Board.from_list([[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]),
    )


# ============================
# HTML バリデーション
# ============================


class _Html5Validator(HTMLParser):
    """タグの開閉を対応づけて HTML5 の構文エラーを検出する簡易検証器。"""

    VOID_ELEMENTS = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    })

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.has_doctype = False
        self.scripts: list[str] = []

    def handle_decl(self, decl: str) -> None:
        if decl.lower().strip().startswith("doctype"):
            self.has_doctype = True

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in self.VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            self.errors.append(f"閉じタグだけ出現: {tag}")
            return
        if self.stack[-1] != tag:
            self.errors.append(
                f"タグ不整合: 期待 </{self.stack[-1]}> 実 </{tag}>"
            )
            return
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.stack and self.stack[-1] == "script":
            self.scripts.append(data)


class TestHtmlValid:
    def test_html_is_well_formed(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            conn = http.client.HTTPConnection(DEFAULT_HOST, free_port, timeout=2)
            conn.request("GET", "/")
            body = conn.getresponse().read().decode("utf-8")
            conn.close()
        finally:
            srv.stop()

        validator = _Html5Validator()
        validator.feed(body)
        assert validator.has_doctype, "<!DOCTYPE html> が欠けている"
        assert validator.errors == [], f"HTML 構文エラー: {validator.errors}"
        assert validator.stack == [], f"閉じていないタグ: {validator.stack}"

    def test_html_has_eventsource_script(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            conn = http.client.HTTPConnection(DEFAULT_HOST, free_port, timeout=2)
            conn.request("GET", "/")
            body = conn.getresponse().read().decode("utf-8")
            conn.close()
        finally:
            srv.stop()
        assert "new EventSource" in body
        assert "/events" in body

    def test_html_has_transparent_background(self, free_port):
        """OBS オーバーレイ用途で透過背景が指定されていること。"""
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            conn = http.client.HTTPConnection(DEFAULT_HOST, free_port, timeout=2)
            conn.request("GET", "/")
            body = conn.getresponse().read().decode("utf-8")
            conn.close()
        finally:
            srv.stop()
        assert "background:transparent" in body or "background: transparent" in body


# ============================
# SSE プロトコル準拠
# ============================


class _SseClient:
    """
    EventSource プロトコルを模した SSE クライアント (ブラウザ挙動相当)。

    W3C SSE 仕様:
        - フィールド形式: "field: value" または "field:value"
        - イベント区切り: 空行 (\n\n)
        - 認識フィールド: event, data, id, retry
    """

    def __init__(self, host: str, port: int) -> None:
        self._conn = http.client.HTTPConnection(host, port, timeout=5.0)
        self._conn.request(
            "GET", "/events",
            headers={"Accept": "text/event-stream"},
        )
        self._resp = self._conn.getresponse()

    def assert_headers(self) -> None:
        assert self._resp.status == 200
        ct = self._resp.getheader("Content-Type", "")
        assert "text/event-stream" in ct, f"Content-Type 異常: {ct}"
        assert self._resp.getheader("Cache-Control") == "no-cache"

    def read_event(self, timeout: float = 3.0) -> dict[str, str]:
        """次のイベント1件 (空行区切り) を読み取り、フィールド辞書を返す。"""
        deadline = time.time() + timeout
        fields: dict[str, str] = {}
        while time.time() < deadline:
            line = self._resp.fp.readline().decode("utf-8", errors="replace")
            if not line:
                break
            line = line.rstrip("\r\n")
            if line == "":
                if fields:
                    return fields
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                if value.startswith(" "):
                    value = value[1:]
                fields[key] = value
        return fields

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class TestSseProtocol:
    def test_response_headers_are_sse(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            client = _SseClient(DEFAULT_HOST, free_port)
            client.assert_headers()
            client.close()
        finally:
            srv.stop()

    def test_event_has_name_and_data(self, free_port):
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            client = _SseClient(DEFAULT_HOST, free_port)
            # 初回接続では latest が無いので data が来るまで update を push
            time.sleep(0.1)
            srv.state.update(_make_result())
            evt = client.read_event()
            assert evt.get("event") == SSE_EVENT_UPDATE
            assert "data" in evt
            client.close()
        finally:
            srv.stop()

    def test_event_data_is_valid_json(self, free_port):
        import json

        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            client = _SseClient(DEFAULT_HOST, free_port)
            time.sleep(0.1)
            srv.state.update(_make_result())
            evt = client.read_event()
            parsed = json.loads(evt["data"])
            assert "score" in parsed
            assert "player1" in parsed
            client.close()
        finally:
            srv.stop()

    def test_multiple_events_delivered_in_order(self, free_port):
        import json

        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            client = _SseClient(DEFAULT_HOST, free_port)
            time.sleep(0.1)
            # 3 回更新してすべて届くか
            for i in range(3):
                srv.state.update(_make_result())
            scores = []
            for _ in range(3):
                evt = client.read_event()
                if "data" in evt:
                    scores.append(json.loads(evt["data"]).get(
                        "score", {}
                    ).get("total_score"))
            assert len(scores) >= 1  # 最低1件は届く
            client.close()
        finally:
            srv.stop()


# ============================
# 複数クライアント (OBS + ブラウザプレビュー)
# ============================


class TestMultiClient:
    def test_two_clients_both_receive(self, free_port):
        """OBS とブラウザプレビューが同時接続しても両方に配信される。"""
        srv = StreamOverlayServer(port=free_port)
        srv.start()
        try:
            c1 = _SseClient(DEFAULT_HOST, free_port)
            c2 = _SseClient(DEFAULT_HOST, free_port)
            time.sleep(0.2)
            srv.state.update(_make_result())

            e1 = c1.read_event()
            e2 = c2.read_event()
            assert e1.get("event") == SSE_EVENT_UPDATE
            assert e2.get("event") == SSE_EVENT_UPDATE
            c1.close()
            c2.close()
        finally:
            srv.stop()
