"""全観測を順に進め、探索は実行中1件と待ち最新1件に限る。"""
from __future__ import annotations

from collections import Counter
import multiprocessing as mp
import os
from queue import Empty
from time import perf_counter
from typing import Any

from .live_ex_runtime import create_engine
from .live_ex_fresh import EX_FRESH_SEC_DEFAULT, validate_ex_fresh_sec

POLL_SEC = 0.01
CLOSE_SEC = 5.0
MAX_AGE_SEC = EX_FRESH_SEC_DEFAULT


def _compute(connection: Any, settings: dict) -> None:
    from src.phase_j.live_power import apply_power_policy
    apply_power_policy()
    from .live_24h_config import validate_retr_second
    second = validate_retr_second(settings.get('retr_second', False), settings.get('retr', False))
    os.environ['PUYO_RETR_SECOND'] = str(int(second))
    from .live_retr_v2_config import configure_child
    configure_child(settings)
    os.environ['PUYO_EX_RETR'] = '1' if settings.get('retr', False) else '0'
    os.environ['PUYO_EX_XC'] = '1' if settings.get('xc', False) else '0'
    os.environ['PUYO_RETR_H_FORECAST'] = '1' if settings.get('retr_h_forecast', False) else '0'
    engine, epoch = None, None
    try:
        if settings.get('xc', False) and not settings.get('retr', False):
            raise ValueError('live_24h_xcにはlive_24h_retrが必要です')
        while True:
            request = connection.recv()
            if request is None:
                break
            received_at = perf_counter()
            if epoch != request['epoch']:
                if engine is not None:
                    engine.close()
                epoch = request['epoch']
                engine = create_engine(settings, epoch)
            for packet in request['packets']:
                probe = engine.observe(packet)
            started_at = perf_counter()
            result = (dict(key=probe.key, fight=probe.fight) if request.get('sync_only')
                      else engine.calculate())
            if result['key'] != request['key'] or result['fight'] != request['fight']:
                raise RuntimeError('EX観測状態と計算状態の鍵が不一致')
            result.update({k: v for k, v in request.items() if k != 'packets'})
            result['completed_at'] = perf_counter()
            result.update(compute_received_at=received_at, compute_started_at=started_at)
            connection.send(result)
    except Exception as exc:
        connection.send(dict(error=repr(exc)))
    finally:
        if engine is not None:
            engine.close()
        connection.close()


class _State:
    """状態更新は間引かず、未実行の探索要求だけを置換する。"""

    def __init__(self, settings: dict, connection: Any, outputs: Any) -> None:
        self.settings, self.connection, self.outputs = settings, connection, outputs
        self.engine, self.epoch, self.pending = None, None, None
        self.packets: list[dict] = []
        self.busy = False
        self.counts: Counter = Counter()

    def observe(self, request: dict) -> None:
        if self.epoch != request['epoch']:
            if self.engine is not None:
                self.engine.close()
            self.epoch = request['epoch']
            self.engine = create_engine(self.settings, self.epoch)
            self.outputs.put(dict(kind='assets', epoch=self.epoch, receipt=getattr(self.engine, 'receipt', {})))
            self.packets, self.pending = [], None
        probe = self.engine.observe(request['packet'])
        xc = {}
        if self.settings.get('xc_same_time', False) and probe.fight:
            from .live_ex_xc_same_time import observe_xc
            xc = observe_xc(self.engine.engine)
        self.packets.append(request['packet'])
        self.counts['observations'] += 1
        self.counts['requested'] += int(probe.fight)
        self.outputs.put(dict(kind='key', serial=request['serial'], key=probe.key,
                              epoch=self.epoch, fight=probe.fight,
                              input_t=request['input_t'], captured_at=request['captured_at'],
                              observed_at=perf_counter(), requested_at=request.get('requested_at'),
                              resets=self.engine.take_resets() if hasattr(self.engine, 'take_resets') else [], **xc))
        if self.pending is not None and self.pending['fight']:
            self.counts['replaced'] += 1
        self.pending = {k: v for k, v in request.items() if k != 'packet'}
        self.pending.update(key=probe.key, fight=probe.fight)

    def pump(self) -> None:
        if self.busy and self.connection.poll():
            result = self.connection.recv()
            self.busy = False
            self.counts['completed'] += int(result.get('fight', False))
            self.outputs.put(dict(kind='result', value=result))
            if result.get('error'):
                raise RuntimeError(result['error'])
        if not self.busy and self.pending is not None:
            request = dict(self.pending, packets=self.packets)
            self.connection.send(request)
            self.pending, self.packets, self.busy = None, [], True
            self.counts['sync_batches'] += 1
            self.counts['executed'] += int(request['fight'])


def _serve(inputs: Any, outputs: Any, settings: dict, stop: Any) -> None:
    from src.phase_j.live_power import apply_power_policy
    apply_power_policy()
    os.environ['PUYO_EX_RETR'] = '0'
    ctx = mp.get_context('spawn')
    local, remote = ctx.Pipe()
    worker = ctx.Process(target=_compute, args=(remote, settings), name='live-ex-compute')
    worker.start()
    remote.close()
    state = _State(settings, local, outputs)
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
            if not worker.is_alive():
                raise RuntimeError(f'EX計算子の終了: {worker.exitcode}')
    except Exception as exc:
        outputs.put(dict(kind='error', error=repr(exc)))
    finally:
        if state.pending is not None and state.pending['fight']:
            state.counts['cancelled_pending'] += 1
        state.pending, state.packets = None, []
        if state.engine is not None:
            state.engine.close()
        if worker.is_alive():
            local.send(None)
        worker.join(CLOSE_SEC)
        if worker.is_alive():
            worker.terminate()
            worker.join(CLOSE_SEC)
        local.close()
        outputs.put(dict(kind='counts', counts=dict(state.counts)))


def join_draining(process: Any, drain: Any, timeout: float) -> None:
    """子の終了を timeout 秒まで待つ。待つ間も drain で子の出力を受け取り続ける。"""
    deadline = perf_counter() + timeout
    while process.is_alive() and perf_counter() < deadline:
        drain()
        process.join(POLL_SEC)


class AsyncEx:
    """専用の状態子と探索子。OFF時はこの所有者を作らない。"""

    fresh_sec: float = MAX_AGE_SEC

    def __init__(self, settings: dict) -> None:
        self.fresh_sec = validate_ex_fresh_sec(settings.get('ex_fresh_sec', MAX_AGE_SEC))
        ctx = mp.get_context('spawn')
        self.inputs, self.outputs = ctx.Queue(), ctx.Queue()
        self.stop = ctx.Event()
        target = _serve
        if settings.get('ex_fire_first', False):
            from .live_ex_fire_first import serve
            target = serve
        self.process = ctx.Process(target=target, args=(self.inputs, self.outputs, settings, self.stop),
                                   name='live-ex-state')
        self.process.start()
        self.serial, self.epoch = 0, None
        self.key, self.latest, self.error = None, None, None
        self.events: list[dict] = []
        self.closed = False

    def submit(self, packet: dict, epoch: int, captured_at: float) -> None:
        self.serial += 1
        self.epoch = epoch
        self.inputs.put(dict(packet=packet, epoch=epoch, serial=self.serial,
                             captured_at=captured_at, input_t=packet['t']))

    def poll(self) -> list[dict]:
        rows = []
        while True:
            try:
                item = self.outputs.get_nowait()
            except Empty:
                break
            rows.append(item)
            if item['kind'] == 'key' and item['serial'] == self.serial:
                self.key = item
            elif item['kind'] == 'result':
                self.latest = item['value']
                self.latest['available_at'] = perf_counter()
                if 'extra' in self.latest:
                    for name in ('av_ex_p1', 'av_ex_p2'):
                        self.latest['extra'][name] = self.latest['available_at']
            elif item['kind'] == 'error':
                self.error = item['error']
        if hasattr(self, 'events'):
            self.events.extend(rows)
        if hasattr(self, 'process') and not self.closed and not self.process.is_alive() and not self.error:
            self.error = f'worker_exit:{self.process.exitcode}'
        return rows

    def take_events(self) -> list[dict]:
        events, self.events = self.events, []
        return events

    def available(self, now: float, input_t: float) -> tuple[dict | None, str]:
        self.poll()
        if self.error:
            return None, 'worker_failed'
        if not self.latest or not self.key or self.key['serial'] != self.serial:
            return None, 'not_completed'
        row = self.latest
        if row.get('epoch') != self.epoch or row.get('key') != self.key['key']:
            return None, 'key_mismatch'
        if not self.key['fight'] or not row.get('fight'):
            return None, 'outside_exchange'
        if row.get('available_at', row['completed_at']) > now or row['input_t'] > input_t:
            return None, 'future'
        if not 0 <= now - row['captured_at'] <= self.fresh_sec:
            return None, 'stale'
        return row, 'available'

    def close(self) -> None:
        self.closed = True
        self.stop.set()
        self.inputs.put(None)
        # 子が読まずに終えた入力が pipe (Windows は 8KB) に残っても、この process の終了で送り出しを待たない。
        self.inputs.cancel_join_thread()
        # 待つ間も出力を受け取る。受け取らないと子は出力の送り出しが詰まって終われず、この process も終了時に子を待って止まる
        # (10-28 24j の制限なしの煙試験の停止で、評価と EX の状態の process が 61.6 秒止まらず強制終了)。
        join_draining(self.process, self.poll, CLOSE_SEC * 2)
        if self.process.is_alive():
            self.error = 'state_close_timeout'
            # 子孫を残す強制終了はしない。停止状態を監査に残す。
        else:
            self.poll()
            self.inputs.close()
            self.outputs.close()
