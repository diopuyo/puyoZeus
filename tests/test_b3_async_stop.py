"""B3 作業 process の停止の回収 (finished=null の原因) を検査する。

原因: Ctrl+C はコンソールを共有する全 process に届く。作業 process (state / compute) が KeyboardInterrupt で
先に死ぬと、親が close() で終了の合図 (None) を送っても done を返せず、b3_async_status.json の finished が null になる
(2026-10-24 の手で止めた 18 件が全て null。smoke_NEW の b3_updates.jsonl に {"fatal": "KeyboardInterrupt: "})。
対策 (b3_worker_graceful_stop、既定 OFF): 作業 process が SIGINT を無視し、親の合図か親の消滅でだけ止まる。
OFF では停止の動作は変えず、回収の事実 (終了コード・回収できなかった process・null の理由) だけを status に足す。
"""
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from typing import Any

import pytest

from src import b3_async
from src.b3_async import AsyncB3, GRACEFUL_STOP_ENV
from src.b3_timing import TimingTable


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GRACEFUL_STOP_ENV, raising=False)


# ---------------------------------------------------------------- 理由の名前

@pytest.mark.parametrize('fatal,forced,expected', [
    (['KeyboardInterrupt: '], [], b3_async.MISSING_INTERRUPTED),
    (['KeyboardInterrupt: ', 'RuntimeError: x'], ['b3-state'], b3_async.MISSING_INTERRUPTED),
    (['RuntimeError: x'], [], b3_async.MISSING_FATAL),
    ([], ['b3-compute'], b3_async.MISSING_FORCED),
    ([], [], b3_async.MISSING_NO_DONE)])
def test_missing_reason_names_the_cause(fatal: list[str], forced: list[str], expected: str) -> None:
    assert b3_async.missing_reason(fatal, forced) == expected


# ---------------------------------------------------------------- 作業 process の入口 (fake)

class FakeConnection:
    def __init__(self, messages: list[Any]) -> None:
        self.messages, self.sent, self.closed = list(messages), [], False

    def poll(self, timeout: float = 0.0) -> bool:
        return bool(self.messages)

    def recv(self) -> Any:
        return self.messages.pop(0)

    def send(self, message: Any) -> None:
        self.sent.append(message)

    def close(self) -> None:
        self.closed = True


class FakeQueue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)

    def get(self, timeout: float = 0.0) -> Any:
        raise Empty


def test_keyboard_interrupt_in_state_worker_becomes_fatal_and_never_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """原因の再現 (OFF = 従来): Ctrl+C を受けた作業 process は done を返さず fatal(KeyboardInterrupt) だけを返して終わる。"""
    def interrupted(*args: Any) -> None:
        raise KeyboardInterrupt
    monkeypatch.setattr(b3_async, '_StateLoop', lambda *args: object())
    monkeypatch.setattr(b3_async, '_pump', interrupted)
    outputs, connection = FakeQueue(), FakeConnection([{'ready': True}])
    b3_async._state_worker(FakeQueue(), outputs, connection, TimingTable(), 'live', True, False, None, False)
    assert outputs.items == [{'ready': True}, {'fatal': 'KeyboardInterrupt: '}]
    assert not any('done' in item for item in outputs.items)
    assert connection.sent == [None] and connection.closed       # compute 側へは終了を伝えて閉じる


def test_interrupt_handler_is_untouched_by_default_and_ignored_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    before = signal.getsignal(signal.SIGINT)
    try:
        b3_async._protect_from_interrupt()
        assert signal.getsignal(signal.SIGINT) is before                 # 既定 OFF: 何も変えない
        monkeypatch.setenv(GRACEFUL_STOP_ENV, '1')
        b3_async._protect_from_interrupt()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, before)


def test_default_is_off_and_env_is_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert b3_async.B3_WORKER_GRACEFUL_STOP_DEFAULT is False and b3_async.graceful_stop_enabled() is False
    monkeypatch.setenv(GRACEFUL_STOP_ENV, '1')
    assert b3_async.graceful_stop_enabled() is True


def run_pump(monkeypatch: pytest.MonkeyPatch, graceful: bool, parent_gone: bool, end_after: int) -> list[Any]:
    """_pump を回す。inputs は end_after 回目の取得で終端 (None) を返す。親の生存確認は毎回行う設定にする。"""
    if graceful:
        monkeypatch.setenv(GRACEFUL_STOP_ENV, '1')
    monkeypatch.setattr(b3_async, 'PARENT_CHECK_SEC', 0.0)
    checks: list[bool] = []
    def gone() -> bool:
        checks.append(True)
        return parent_gone
    monkeypatch.setattr(b3_async, '_parent_gone', gone)
    class Inputs:
        calls = 0
        def get(self, timeout: float = 0.0) -> Any:
            self.calls += 1
            if self.calls >= end_after:
                return None
            raise Empty
    state = SimpleNamespace(pending=[], counts=dict(inputs=0))
    outputs = FakeQueue()
    b3_async._pump(Inputs(), outputs, FakeConnection([]), state, False)
    return outputs.items + [dict(parent_checks=len(checks))]


def test_pump_without_graceful_never_asks_about_the_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    items = run_pump(monkeypatch, graceful=False, parent_gone=True, end_after=5)
    assert items == [{'done': dict(inputs=0)}, dict(parent_checks=0)]


def test_pump_with_graceful_stops_when_the_parent_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl+C を無視する作業 process が、親の強制終了後に孤児として残らない。"""
    with pytest.raises(RuntimeError, match='親process'):
        run_pump(monkeypatch, graceful=True, parent_gone=True, end_after=10 ** 6)


def test_pump_with_graceful_keeps_running_while_the_parent_lives(monkeypatch: pytest.MonkeyPatch) -> None:
    items = run_pump(monkeypatch, graceful=True, parent_gone=False, end_after=5)
    assert items[0] == {'done': dict(inputs=0)} and items[1]['parent_checks'] >= 1


def test_parent_gone_is_false_without_a_parent_process() -> None:
    assert b3_async._parent_gone() is False        # pytest 本体 (親 process 情報なし) は常に「親は居る」


# ---------------------------------------------------------------- status (回収の記録)

class FakeWorker:
    def __init__(self, name: str, exitcode: int | None, pid: int | None = 1234) -> None:
        self.name, self.exitcode, self.pid = name, exitcode, pid


def stopped_b3(output: Path, process: FakeWorker, compute: FakeWorker, summary: dict | None,
               fatals: list[str], submitted: int = 7) -> AsyncB3:
    b3 = object.__new__(AsyncB3)
    b3.process, b3.compute, b3.summary, b3.fatals = process, compute, summary, list(fatals)
    b3.submitted, b3.output = submitted, output
    return b3


def test_status_keeps_existing_keys_and_adds_exit_codes(tmp_path: Path) -> None:
    b3 = stopped_b3(tmp_path, FakeWorker('b3-state', 0), FakeWorker('b3-compute', 0), dict(inputs=7), [])
    status = b3.stop_status([])
    assert status['submitted'] == 7 and status['finished'] == dict(inputs=7) and status['forced_termination'] == []
    assert status['exit_codes'] == {'b3-state': 0, 'b3-compute': 0} and status['unreaped'] == []
    assert 'finished_missing_reason' not in status and status['graceful_stop'] is False


def test_status_names_the_reason_when_finished_is_null_after_ctrl_c(tmp_path: Path) -> None:
    b3 = stopped_b3(tmp_path, FakeWorker('b3-state', 0), FakeWorker('b3-compute', 0), None, ['KeyboardInterrupt: '])
    status = b3.stop_status([])
    assert status['finished'] is None
    assert status['finished_missing_reason'] == b3_async.MISSING_INTERRUPTED
    assert status['worker_fatal'] == ['KeyboardInterrupt: ']


def test_status_names_workers_that_could_not_be_reaped(tmp_path: Path) -> None:
    b3 = stopped_b3(tmp_path, FakeWorker('b3-state', None), FakeWorker('b3-compute', -15), None, [])
    status = b3.stop_status(['b3-state'])
    assert status['unreaped'] == ['b3-state'] and status['exit_codes'] == {'b3-state': None, 'b3-compute': -15}
    assert status['finished_missing_reason'] == b3_async.MISSING_FORCED
    never_started = stopped_b3(tmp_path, FakeWorker('b3-state', None, pid=None), FakeWorker('b3-compute', None, pid=None),
                               None, [])
    assert never_started.stop_status([])['unreaped'] == []        # 起動していない process は未回収と数えない


class ScriptedOutputs:
    def __init__(self, items: list[Any]) -> None:
        self.items = list(items)

    def get_nowait(self) -> Any:
        if not self.items:
            raise Empty
        return self.items.pop(0)


def test_poll_collects_fatal_messages_even_when_the_consumer_reads_them_first() -> None:
    """消費側 (LiveB3) が poll で fatal を先に受け取っても、停止の記録 (status) に残る。"""
    b3 = object.__new__(AsyncB3)
    b3.outputs, b3.finished, b3.fatals = ScriptedOutputs([{'fatal': 'KeyboardInterrupt: '}]), True, []
    b3.process = b3.compute = SimpleNamespace(is_alive=lambda: True)
    assert b3.poll() == [{'fatal': 'KeyboardInterrupt: '}]
    assert b3.fatals == ['KeyboardInterrupt: ']


# ---------------------------------------------------------------- 実 process (POSIX のみ)
# Windows の Ctrl+C はコンソール共有の全 process への CTRL_C_EVENT で、試験 process 自身も巻き込むため、
# ここでは SIGINT を作業 process の pid へ直接送る POSIX だけで検査する (Windows は手動の実コンソール試験で確認した)。

def spawn_and_interrupt(tmp_path: Path) -> dict:
    import joblib
    from src.b3_train_model import TwoStateModel
    model = tmp_path / 'model.joblib'
    joblib.dump(TwoStateModel(static=None, exchange=None), model)
    b3 = AsyncB3(model, TimingTable(), output=tmp_path)
    try:
        for worker in (b3.process, b3.compute):
            os.kill(worker.pid, signal.SIGINT)          # Ctrl+C が作業 process にも届いた状態
        import time
        time.sleep(1.0)
    finally:
        b3.close()
    return json.loads((tmp_path / 'b3_async_status.json').read_text(encoding='utf-8'))


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX の SIGINT を作業 process へ直接送る検査')
def test_real_workers_off_lose_done_on_interrupt(tmp_path: Path) -> None:
    status = spawn_and_interrupt(tmp_path)
    assert status['finished'] is None                              # 不具合の再現 (従来の動作)
    assert status['finished_missing_reason'] == b3_async.MISSING_INTERRUPTED
    assert status['forced_termination'] == [] and status['unreaped'] == []
    assert set(status['exit_codes']) == {'b3-state', 'b3-compute'}


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX の SIGINT を作業 process へ直接送る検査')
def test_real_workers_graceful_finish_and_report_done_on_interrupt(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GRACEFUL_STOP_ENV, '1')                     # spawn 子へ環境変数で届く
    status = spawn_and_interrupt(tmp_path)
    assert status['finished'] == dict(inputs=0, triggers=0, discarded=0, computed=0)
    assert status['finished']['inputs'] == status['submitted'] == 0     # 母数 0 でも done が返った (null でない)
    assert 'finished_missing_reason' not in status and status['graceful_stop'] is True
    assert status['exit_codes'] == {'b3-state': 0, 'b3-compute': 0} and status['unreaped'] == []
    assert status['forced_termination'] == []
