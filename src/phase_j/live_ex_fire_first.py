"""新しい撃ち合いを空いている計算子へ渡す、既定OFFの実行器。"""
from __future__ import annotations

import multiprocessing as mp
from queue import Empty
from time import perf_counter
from typing import Any

from .live_ex_async import CLOSE_SEC, POLL_SEC, _State, _compute

COMPUTE_WORKERS = 2


class FireFirstState(_State):
    """各計算子に未送信の全観測を残し、同じ鍵は一方でだけ計算する。"""

    def __init__(self, settings: dict, connections: list[Any], outputs: Any) -> None:
        super().__init__(settings, connections[0], outputs)
        self.connections = connections
        self.histories: list[list[dict]] = [[] for _ in connections]
        self.running: list[dict | None] = [None for _ in connections]
        self.active = 0
        self.dispatched_identity: tuple | None = None

    def observe(self, request: dict) -> None:
        if self.epoch != request['epoch']:
            self.histories = [[] for _ in self.connections]
        super().observe(request)
        for history in self.histories:
            history.append(request['packet'])
        self.packets = []

    def receive(self) -> None:
        for index, connection in enumerate(self.connections):
            if self.running[index] is None or not connection.poll():
                continue
            result = connection.recv()
            self.running[index] = None
            if result.get('sync_only'):
                self.counts['standby_synced'] += 1
                continue
            self.counts['completed'] += int(result.get('fight', False))
            self.outputs.put(dict(kind='result', value=result))
            if result.get('error'):
                raise RuntimeError(result['error'])

    def pump(self) -> None:
        self.receive()
        if self.pending is None:
            return
        identity = (self.pending['epoch'], self.pending['key'], self.pending['fight'])
        new_fight = self.pending['fight'] and identity != self.dispatched_identity
        index = self.active
        if self.running[index] is not None:
            free = next((i for i, request in enumerate(self.running) if request is None), None)
            if not new_fight or free is None:
                self.sync_standby()
                return
            index = free
            self.counts['fire_bypass'] += 1
        request = dict(self.pending, packets=self.histories[index],
                       dispatched_at=perf_counter(), compute_worker=index)
        self.connections[index].send(request)
        self.running[index] = request
        self.histories[index], self.pending = [], None
        self.active, self.dispatched_identity = index, identity
        self.counts['sync_batches'] += 1
        self.counts['executed'] += int(request['fight'])
        self.sync_standby()

    def sync_standby(self) -> None:
        """空き側も計算せず観測を進め、発火時の初期化と履歴再生を減らす。"""
        current = self.running[self.active] or self.pending
        if current is None:
            return
        for index, connection in enumerate(self.connections):
            if index == self.active or self.running[index] is not None or not self.histories[index]:
                continue
            # 最新の鍵は pending にあり、実行中の古い鍵とは異なることがある。
            source = self.pending or current
            request = {k: v for k, v in source.items() if k != 'packets'}
            request.update(packets=self.histories[index], sync_only=True, compute_worker=index)
            connection.send(request)
            self.running[index], self.histories[index] = request, []


def start_workers(settings: dict) -> tuple[list[Any], list[Any]]:
    ctx = mp.get_context('spawn')
    connections, workers = [], []
    for index in range(COMPUTE_WORKERS):
        local, remote = ctx.Pipe()
        worker = ctx.Process(target=_compute, args=(remote, settings), name=f'live-ex-compute-{index}')
        worker.start()
        remote.close()
        connections.append(local)
        workers.append(worker)
    return connections, workers


def close_workers(state: FireFirstState, workers: list[Any]) -> None:
    if state.pending is not None and state.pending['fight']:
        state.counts['cancelled_pending'] += 1
    if state.engine is not None:
        state.engine.close()
    # WindowsのPipeでは結果送信中の子へsendすると双方が詰まる。
    # 全計算子の未受信結果を先に読み、待受中の子にだけ停止を送る。
    deadline = perf_counter() + CLOSE_SEC
    while any(state.running) and perf_counter() < deadline:
        for index, connection in enumerate(state.connections):
            if state.running[index] is not None and connection.poll():
                try:
                    connection.recv()
                except EOFError:
                    pass
                state.running[index] = None
        for worker in workers:
            worker.join(POLL_SEC)
    for index, (worker, connection) in enumerate(zip(workers, state.connections)):
        if worker.is_alive() and state.running[index] is None:
            connection.send(None)
    deadline = perf_counter() + CLOSE_SEC
    for worker, connection in zip(workers, state.connections):
        worker.join(max(0., deadline-perf_counter()))
        if worker.is_alive():
            worker.terminate()
            worker.join(CLOSE_SEC)
        connection.close()
    state.outputs.put(dict(kind='counts', counts=dict(state.counts)))


def serve(inputs: Any, outputs: Any, settings: dict, stop: Any) -> None:
    from src.phase_j.live_power import apply_power_policy
    apply_power_policy()
    import os
    os.environ['PUYO_EX_RETR'] = '0'
    connections, workers = start_workers(settings)
    state = FireFirstState(settings, connections, outputs)
    try:
        while not stop.is_set():
            try:
                request = inputs.get(timeout=POLL_SEC)
            except Empty:
                request = False
            if request is None:
                break
            if request is not False:
                state.observe(request)
            if stop.is_set():
                break
            state.pump()
            if any(not worker.is_alive() for worker in workers):
                raise RuntimeError('EX計算子の終了')
    except Exception as exc:
        outputs.put(dict(kind='error', error=repr(exc)))
    finally:
        close_workers(state, workers)
