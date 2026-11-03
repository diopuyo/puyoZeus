"""認識の次通知を待たず、B3結果だけで評価所有者が起床する。"""
from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

from src.phase_j.live_completion import receive_completion


def test_completion_wakes_without_recognition() -> None:
    context = mp.get_context('spawn')
    recognition, outputs = context.Queue(), context.Queue()
    received: list[dict] = []
    callbacks: list[bool] = []
    alive = SimpleNamespace(is_alive=lambda: True)

    def consume() -> bool:
        received.append(outputs.get())
        return True

    b3 = SimpleNamespace(completion_driven=True, async_failed=False,
        async_engine=SimpleNamespace(outputs=outputs, process=alive, compute=alive),
        receive_completed=consume)
    bridge = SimpleNamespace(b3=b3, queue=recognition, worker=alive,
                             on_b3_completed=lambda: callbacks.append(True))
    try:
        outputs.put(dict(raw_probability=.8))
        assert not receive_completion(bridge, timeout=2.)
        assert received == [dict(raw_probability=.8)]
        assert callbacks == [True]
        assert not receive_completion(bridge, timeout=0.)
        assert callbacks == [True]
    finally:
        for queue in (recognition, outputs):
            queue.close()
            queue.join_thread()


def test_disabled_keeps_original_receive_path() -> None:
    assert receive_completion(SimpleNamespace(), timeout=0.)


def test_rejected_completion_does_not_publish() -> None:
    context = mp.get_context('spawn')
    recognition, outputs = context.Queue(), context.Queue()
    alive = SimpleNamespace(is_alive=lambda: True)
    calls: list[bool] = []

    def reject() -> bool:
        outputs.get()
        return False

    b3 = SimpleNamespace(completion_driven=True, async_failed=False,
        async_engine=SimpleNamespace(outputs=outputs, process=alive, compute=alive), receive_completed=reject)
    bridge = SimpleNamespace(b3=b3, queue=recognition, worker=alive,
                             on_b3_completed=lambda: calls.append(True))
    try:
        outputs.put(dict(epoch='旧世代'))
        assert not receive_completion(bridge, timeout=2.)
        assert not calls
    finally:
        for queue in (recognition, outputs):
            queue.close()
            queue.join_thread()
