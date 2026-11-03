"""待ち置換でも履歴を残し、未来・期限超過・旧世代を採用しない。"""
from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

import pytest

from src.phase_j import live_ex_async as A


def test_latest_request_keeps_all_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    observed, sent = [], []
    engine = SimpleNamespace(observe=lambda p: (observed.append(p['t']) or
                             SimpleNamespace(key=(1, 2, (3, None)), fight=True)), close=lambda: None)
    monkeypatch.setattr(A, 'create_engine', lambda *_: engine)
    connection = SimpleNamespace(send=sent.append, poll=lambda: False)
    state = A._State({}, connection, Queue())
    state.busy = True
    for serial in range(1, 4):
        state.observe(dict(packet=dict(t=serial), epoch=1, serial=serial,
                           captured_at=float(serial), input_t=float(serial)))
        state.pump()
    assert observed == [1, 2, 3] and sent == []
    state.busy = False
    state.pump()
    assert len(sent) == 1 and sent[0]['serial'] == 3
    assert [p['t'] for p in sent[0]['packets']] == observed
    assert state.counts['replaced'] == 2
    assert state.counts['requested'] == 3 and state.counts['executed'] == 1
    assert state.counts['sync_batches'] == 1


def owner() -> A.AsyncEx:
    result = object.__new__(A.AsyncEx)
    result.outputs = Queue()
    result.serial, result.epoch, result.error = 1, 4, None
    result.key = dict(serial=1, key=(4, 2, (3, None)), fight=True)
    result.latest = dict(epoch=4, key=result.key['key'], fight=True,
                         captured_at=10., completed_at=10.1, input_t=3.)
    return result


@pytest.mark.parametrize(('now', 'input_t', 'reason'), [
    (10.5, 3., 'available'), (10.500001, 3., 'stale'),
    (10.09, 3., 'future'), (10.2, 2.9, 'future'), (9.9, 3., 'future')])
def test_freshness(now: float, input_t: float, reason: str) -> None:
    assert owner().available(now, input_t)[1] == reason


def test_old_epoch_and_unacknowledged_input() -> None:
    subject = owner()
    subject.epoch += 1
    assert subject.available(10.2, 3.)[1] == 'key_mismatch'
    subject = owner()
    subject.serial += 1
    assert subject.available(10.2, 3.)[1] == 'not_completed'


def test_ipc_arrival_is_not_backdated(monkeypatch: pytest.MonkeyPatch) -> None:
    subject = owner()
    subject.outputs.put(dict(kind='result', value=dict(subject.latest)))
    monkeypatch.setattr(A, 'perf_counter', lambda: 10.2)
    assert subject.available(10.15, 3.)[1] == 'future'
    assert subject.available(10.2, 3.)[1] == 'available'


def test_join_draining_lets_flooding_child_exit() -> None:
    """出力を受け取らずに待つと、子は送り出しが詰まって終われない (10-28 24j の停止で 61.6 秒の強制終了)。受け取りながら待てば終わる。"""
    from queue import Empty
    import multiprocessing as mp
    from tests.mp_flood_helper import ITEMS, flood
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    child = ctx.Process(target=flood, args=(queue,))
    child.start()
    child.join(3.)
    assert child.is_alive()   # 受け取らない待ちでは終わらない (旧 close と同じ待ち方)
    received = []

    def drain() -> None:
        while True:
            try:
                received.append(queue.get_nowait())
            except Empty:
                return
    A.join_draining(child, drain, 10.)
    drain()
    assert not child.is_alive() and child.exitcode == 0
    assert len(received) == ITEMS


def test_stop_does_not_run_queued_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []
    connection = SimpleNamespace(send=sent.append, close=lambda: None)
    worker = SimpleNamespace(start=lambda: None, is_alive=lambda: True,
                             join=lambda _: None, terminate=lambda: None)
    context = SimpleNamespace(Pipe=lambda: (connection, connection),
                              Process=lambda **_: worker)
    monkeypatch.setattr(A.mp, 'get_context', lambda _: context)
    inputs = SimpleNamespace(get=lambda **_: pytest.fail('OFF後の待ち入力を実行'))
    outputs = Queue()
    A._serve(inputs, outputs, {}, SimpleNamespace(is_set=lambda: True))
    assert sent == [None]
    assert outputs.get_nowait() == dict(kind='counts', counts={})
