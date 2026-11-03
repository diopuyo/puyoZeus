"""認識とB3完了の到着を待ち、評価状態は既存の単一所有者で更新する。"""
from __future__ import annotations

from multiprocessing.connection import wait
from typing import Any


def x_wake(b3: Any) -> Any:
    """段1 直接公開 (既定OFF) の起床信号。OFF・v4.9f 欄なしでは None。"""
    return getattr(getattr(b3, 'v49f', None), 'wake', None)


def publish_x(bridge: Any, wake: Any, published: bool) -> None:
    """X の初出で起きた。同じ起床で B3 完了を公開済みなら、その公開に X も載っている。"""
    wake.drain()
    if not published and getattr(bridge, 'b3_context', None) is not None:
        bridge.on_b3_completed()


def receive_completion(bridge: Any, timeout: float) -> bool:
    """周期を短縮せずIPC到着で起床する。戻り値は認識通知の読取可否。"""
    b3 = getattr(bridge, 'b3', None)
    if b3 is None or not getattr(b3, 'completion_driven', False):
        return True
    if b3.async_engine is None or b3.async_failed:
        return True
    # multiprocessing.Queueの受信端はこの所有者だけが読む。
    # 別threadによるpollと競合させず、Windowsのpipe待機にも対応する。
    recognition = bridge.queue._reader
    completed = b3.async_engine.outputs._reader
    wake = x_wake(b3)
    ready = wait((recognition, completed) if wake is None else (recognition, completed, wake.reader),
                 timeout=timeout)
    published = False
    if (completed in ready or not b3.async_engine.process.is_alive()
            or not b3.async_engine.compute.is_alive()):
        if b3.receive_completed():
            bridge.on_b3_completed()
            published = True
    if wake is not None and wake.reader in ready:
        publish_x(bridge, wake, published)
    return recognition in ready or not bridge.worker.is_alive()
