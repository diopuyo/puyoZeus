"""停止の割り込みを計算中へ決定的に差し込み、未返信1件と最後の吸い出しを検査する。"""
from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
import signal
from types import SimpleNamespace
from typing import Any

import pytest
from src.phase_j import live_v49f, live_v49f_async, live_v49f_engine, live_v49f_origins
from src.b3_async import GRACEFUL_STOP_ENV

PACKET_ID = 9533


class Connection:
    def __init__(self) -> None:
        self.messages = [dict(id=PACKET_ID, game=0, t=1., raw=[], origins=[], submitted_at=0.), None]
        self.sent: list[dict] = []
        self.closed = False

    def recv(self) -> Any:
        return self.messages.pop(0)

    def send(self, value: dict) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


def interrupt_current_handler() -> None:
    """他processへ信号を送らず、計算中に現在のSIGINTハンドラを呼ぶ。"""
    handler = signal.getsignal(signal.SIGINT)
    if handler != signal.SIG_IGN:
        handler(signal.SIGINT, None)


@pytest.mark.parametrize('graceful', [False, True])
def test_interrupt_after_input_is_drained_only_when_graceful(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, graceful: bool) -> None:
    """新規5旗なしで、入力受信→計算中SIGINT→親のNoneという順番を固定する。"""
    monkeypatch.setenv(GRACEFUL_STOP_ENV, str(int(graceful)))
    monkeypatch.setenv('PUYO_B3_M3', '0')
    monkeypatch.setenv('PUYO_B3_NET_LEDGER_COLS', '0')
    monkeypatch.setenv('PUYO_LIVE_X_DIRECT_PUBLISH', '0')
    for name in ('PUYO_LIVE_RECOG_SPEED_PY', 'PUYO_B3_SHARE_SCORE_READ', 'PUYO_LIVE_NATIVE_READ_BOARD',
                 'PUYO_LIVE_RECOG_SPEED_PY_PAIR_LAZY', 'PUYO_B3_ENSEMBLE'):
        monkeypatch.setenv(name, '0')
    calls: list[str] = []
    def observe(*args: Any) -> tuple[None, None]:
        calls.append('compute_enter')
        interrupt_current_handler()
        calls.append('compute_complete')
        return None, None
    engine = SimpleNamespace(base=SimpleNamespace(policy={}, native=None), observe=observe,
                             close=lambda: calls.append('engine_close'))
    monkeypatch.setattr(live_v49f_engine, 'load_engine', lambda *args: engine)
    monkeypatch.setattr(live_v49f_origins, 'OriginStream',
                        lambda *args: SimpleNamespace(observe=lambda *a: []))
    connection, before = Connection(), signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        live_v49f_async.worker(connection, (), tmp_path)
    finally:
        signal.signal(signal.SIGINT, before)
    results = [row for row in connection.sent if 'id' in row]
    assert [row['id'] for row in results] == ([PACKET_ID] if graceful else [])
    assert any('fatal' in row for row in connection.sent) is (not graceful)
    assert ('compute_complete' in calls) is graceful
    assert connection.closed and calls[-1] == 'engine_close'
    assert len((tmp_path/'v49f_updates.jsonl').read_text().splitlines()) == int(graceful)


def test_close_collects_final_reply_after_threads_finish(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """受信threadの最後の通知も吸い出してから終了票を数える。"""
    monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', '1')
    view = object.__new__(live_v49f.LiveV49F)
    view.inputs, view.outputs = Queue(), Queue()
    view.output, view.submitted, view.received, view.failure = tmp_path, 1, 1, None
    order: list[str] = []
    def receiver_join(timeout: float) -> None:
        assert order == ['sender_stopped']
        view.outputs.put(dict(fatal='final-reply-error'))
        order.append('receiver_stopped')
    view.thread = SimpleNamespace(join=lambda t: order.append('sender_stopped'), is_alive=lambda: False)
    view.receiver = SimpleNamespace(join=receiver_join, is_alive=lambda: False)
    view.worker = SimpleNamespace(close=lambda: order.append('worker_closed'))
    view.wake = SimpleNamespace(close=lambda: None)
    view.close_prefire = lambda: None
    view.x_wake_counts = lambda: {}
    view.close_direct()
    status = json.loads((tmp_path/'v49f_display_status.json').read_text())
    assert view.outputs.empty()
    assert status['failure'] == 'final-reply-error'
    assert order == ['sender_stopped', 'receiver_stopped', 'worker_closed']

def test_worker_resets_model_environment_before_importing_column_specs(
        tmp_path: Path) -> None:
    """親のC2設定を継いだfresh processで、v49fの既存OFF列仕様を保持する。"""
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    script = r'''
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.phase_j import live_v49f_async
engine = types.SimpleNamespace(base=types.SimpleNamespace(policy={}, native=None), close=lambda: None)
engine_module = types.ModuleType('src.phase_j.live_v49f_engine')
engine_module.load_engine = lambda *args: engine
origin_module = types.ModuleType('src.phase_j.live_v49f_origins')
origin_module.OriginStream = lambda *args: object()
sys.modules[engine_module.__name__] = engine_module
sys.modules[origin_module.__name__] = origin_module
class Connection:
    def send(self, value: dict) -> None:
        assert not value.get('fatal'), value
    def recv(self) -> None:
        return None
    def close(self) -> None:
        pass
live_v49f_async.worker(Connection(), (), Path(sys.argv[2]))
from src.b3_column_spec import COLUMN_NAMES, NET_LEDGER_COLUMN_NAMES
assert not set(COLUMN_NAMES).intersection(NET_LEDGER_COLUMN_NAMES), '親の正味列がv49fへ混入'
assert 'sl_inflight_p1' not in COLUMN_NAMES, '親のM3列がv49fへ混入'
'''
    environment = dict(os.environ, PUYO_B3_M3='1', PUYO_B3_NET_LEDGER_COLS='1',
                       PUYO_B3_WORKER_GRACEFUL_STOP='1', PUYO_LIVE_X_DIRECT_PUBLISH='0')
    result = subprocess.run([sys.executable, '-B', '-c', script, str(root), str(tmp_path)],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
