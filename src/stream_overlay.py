"""
配信リアルタイムオーバーレイ

OBS Studio のブラウザソース向けに HTTP/SSE でオーバーレイ情報を配信する。
外部依存 (flask, websockets) を避け、Python 標準ライブラリのみで実装する。

エンドポイント:
    GET /          -> オーバーレイ HTML (EventSource で /events を購読)
    GET /events    -> Server-Sent Events で最新 AnalysisResult を push
    GET /latest    -> 最新 AnalysisResult を一回だけ JSON で返す

使い方:
    state = StreamState()
    server = StreamOverlayServer(state=state, host="127.0.0.1", port=8765)
    server.start()
    # ... 別スレッド/ループで ...
    state.update(analysis_result)
    server.stop()
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.analyzer import AnalysisResult

# ============================
# 定数定義
# ============================

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8765

# SSE 接続の heartbeat 間隔 (秒)
HEARTBEAT_INTERVAL_SEC: float = 15.0

# /events のストリーム待機タイムアウト (秒)
EVENT_WAIT_TIMEOUT_SEC: float = 1.0

# SSE メッセージのイベント名
SSE_EVENT_UPDATE: str = "analysis"
SSE_EVENT_HEARTBEAT: str = "heartbeat"

# HTTP ステータス
HTTP_OK: int = 200
HTTP_NOT_FOUND: int = 404

# サーバ停止時のシグナル値
_SHUTDOWN_SENTINEL: object = object()


# ============================
# HTML テンプレート
# ============================

OVERLAY_HTML_TEMPLATE: str = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<title>ぷよぷよ分析オーバーレイ</title>
<style>
  body { margin:0; background:transparent; color:#fff;
         font-family: sans-serif; }
  #score { font-size: 48px; text-align:center; padding:12px;
           text-shadow: 2px 2px 4px #000; }
  .bar { height:24px; background:#222; margin:0 auto; width:40%;
         border-radius:6px; overflow:hidden; position:relative; }
  .fill { height:100%; }
  .fill.p1 { background:#46a0ff; }
  .fill.p2 { background:#ff4646; }
  .center { position:absolute; left:50%; top:0; bottom:0;
            width:2px; background:#fff; }
</style></head>
<body>
<div id="score">-</div>
<div class="bar"><div id="fill" class="fill"></div>
<div class="center"></div></div>
<script>
const es = new EventSource("/events");
es.addEventListener("analysis", (e) => {
  const d = JSON.parse(e.data);
  const s = d.score.total_score;
  document.getElementById("score").textContent =
      d.score.advantage + "  " + s.toFixed(1);
  const fill = document.getElementById("fill");
  const ratio = Math.max(-1, Math.min(1, s / 100));
  fill.className = "fill " + (s >= 0 ? "p1" : "p2");
  fill.style.width = Math.abs(ratio) * 50 + "%";
  fill.style.marginLeft = s >= 0 ? "50%" : (50 - Math.abs(ratio) * 50) + "%";
});
</script></body></html>
"""


# ============================
# StreamState
# ============================


@dataclass
class StreamState:
    """
    最新の分析結果を保持するスレッドセーフな状態コンテナ。

    複数の SSE クライアントに同時配信するため、各クライアント用
    キューを登録し、update() で一斉配信する。
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _latest_payload: dict[str, Any] | None = field(default=None, init=False)
    _subscribers: list[queue.Queue] = field(default_factory=list, init=False)

    def update(self, result: AnalysisResult) -> None:
        """新しい分析結果を保存し、全購読者にブロードキャストする。"""
        payload = result.to_dict()
        with self._lock:
            self._latest_payload = payload
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass  # 遅いクライアントは落としてでも他を優先

    def latest(self) -> dict[str, Any] | None:
        """最新ペイロードを返す (初回は None)。"""
        with self._lock:
            return dict(self._latest_payload) if self._latest_payload else None

    def subscribe(self) -> queue.Queue:
        """
        新しい購読キューを生成して登録する。

        Returns:
            queue.Queue: 購読用キュー。None が投入されたら終了シグナル。
        """
        q: queue.Queue = queue.Queue(maxsize=16)
        with self._lock:
            self._subscribers.append(q)
            if self._latest_payload is not None:
                q.put_nowait(self._latest_payload)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        """購読キューを解除する。"""
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def subscriber_count(self) -> int:
        """現在の購読数。"""
        with self._lock:
            return len(self._subscribers)


# ============================
# HTTP ハンドラ
# ============================


class _OverlayRequestHandler(BaseHTTPRequestHandler):
    """
    BaseHTTPRequestHandler のサブクラス。
    StreamOverlayServer からインスタンス変数 state を受け取る。
    """

    # サーバから動的に注入される
    state: "StreamState" = None  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """標準エラーへのロギングを抑制する (テスト/本番で静かに)。"""
        return

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            self._serve_html()
        elif path == "/latest":
            self._serve_latest()
        elif path == "/events":
            self._serve_events()
        else:
            self.send_error(HTTP_NOT_FOUND, "Not Found")

    # ============================
    # ルート別実装
    # ============================

    def _serve_html(self) -> None:
        body = OVERLAY_HTML_TEMPLATE.encode("utf-8")
        self.send_response(HTTP_OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_latest(self) -> None:
        payload = self.state.latest() or {}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTP_OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        self.send_response(HTTP_OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q = self.state.subscribe()
        try:
            self._pump_events(q)
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            self.state.unsubscribe(q)

    def _pump_events(self, q: queue.Queue) -> None:
        """購読キューから取り出して SSE メッセージを送信する。"""
        last_heartbeat = time.time()
        while True:
            try:
                payload = q.get(timeout=EVENT_WAIT_TIMEOUT_SEC)
                if payload is _SHUTDOWN_SENTINEL:
                    return
                self._write_sse(SSE_EVENT_UPDATE, payload)
            except queue.Empty:
                pass

            if time.time() - last_heartbeat > HEARTBEAT_INTERVAL_SEC:
                self._write_sse(SSE_EVENT_HEARTBEAT, {"ts": time.time()})
                last_heartbeat = time.time()

    def _write_sse(self, event: str, data: dict[str, Any]) -> None:
        """SSE プロトコルに従い1メッセージ送信する。"""
        lines = [
            f"event: {event}\n",
            f"data: {json.dumps(data, ensure_ascii=False)}\n",
            "\n",
        ]
        self.wfile.write("".join(lines).encode("utf-8"))
        self.wfile.flush()


# ============================
# StreamOverlayServer
# ============================


class StreamOverlayServer:
    """
    配信オーバーレイ HTTP+SSE サーバ。

    Usage:
        state = StreamState()
        srv = StreamOverlayServer(state=state)
        srv.start()
        ...
        srv.stop()
    """

    def __init__(
        self,
        state: StreamState | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        """
        Args:
            state: 共有状態 (None なら新規作成)。
            host: バインドアドレス。
            port: ポート番号。
        """
        self.state: StreamState = state or StreamState()
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ============================
    # 公開メソッド
    # ============================

    def start(self) -> None:
        """サーバをバックグラウンドスレッドで起動する。"""
        if self._server is not None:
            raise RuntimeError("サーバは既に起動中です")
        handler_cls = self._make_handler_class()
        self._server = ThreadingHTTPServer(
            (self._host, self._port), handler_cls,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="StreamOverlayServer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """サーバを停止する。"""
        if self._server is None:
            return
        # 既存 SSE 接続を解除するためシグナル送信
        self._broadcast_shutdown()
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._server = None
        self._thread = None

    def address(self) -> tuple[str, int]:
        """実際にバインドされた (host, port) を返す。"""
        if self._server is None:
            return (self._host, self._port)
        return self._server.server_address[:2]  # type: ignore[return-value]

    # ============================
    # 内部メソッド
    # ============================

    def _make_handler_class(self) -> type[_OverlayRequestHandler]:
        """state を注入したハンドラクラスを動的に作る。"""
        state_ref = self.state

        class Handler(_OverlayRequestHandler):
            state = state_ref  # type: ignore[misc]

        return Handler

    def _broadcast_shutdown(self) -> None:
        """全購読者に停止シグナルを送る。"""
        with self.state._lock:  # 内部: 直接触る
            subs = list(self.state._subscribers)
        for q in subs:
            try:
                q.put_nowait(_SHUTDOWN_SENTINEL)
            except queue.Full:
                pass
