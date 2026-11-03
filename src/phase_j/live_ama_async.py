"""ama専用process。観測は全件、重い読みは実行中1件と最新待機1件に限定する。"""
from __future__ import annotations
from contextlib import nullcontext, closing
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
from time import perf_counter
from typing import Any
import traceback

START_TIMEOUT = 120.
OBSERVATION_CAPACITY = 256


def input_audit(output: Path) -> Any:
    """切分け時だけ、認識から実際に届いた原票を保存する。通常は既定OFF。"""
    import os
    if os.environ.get('PUYO_LIVE_INPUT_AUDIT') == '1':
        return (output/'ama_input_audit.jsonl').open('x',encoding='utf-8',buffering=1)
    return nullcontext()


def replace_latest(queue: Queue, value: Any) -> int:
    dropped = 0
    while True:
        try:
            queue.put_nowait(value)
            return dropped
        except Full:
            try:
                queue.get_nowait()
                dropped += 1
            except Empty:
                pass


def input_status(inputs: Any, item: dict | None, stamp: float) -> dict | None:
    """計算要求が作れない台帳欠測も表示へ伝える。評価入力の値は補完しない。"""
    if item is not None:
        return dict(input_sec=item['request']['input_sec'], missing=item.get('missing'))
    if inputs.missing_reason:
        return dict(input_sec=stamp, missing=inputs.missing_reason)
    return None


def calculator(jobs: Queue, done: Queue, binary: Path, model: Path) -> None:
    from .live_ama_engine import AmaEngine
    engine = None
    try:
        engine = AmaEngine(binary, model)
        while (job := jobs.get()) is not None:
            began = perf_counter()
            result = engine.evaluate(job['item'])
            done.put(dict(job, result=result, began_at=began, completed_at=perf_counter()))
    except BaseException as exc:
        done.put(dict(fatal=f'{type(exc).__name__}: {exc}', stack=traceback.format_exc()))
    finally:
        if engine is not None:
            engine.close()


def collect_results(done: Queue, audit: Any, compact: bool = False) -> list[dict]:
    results = []
    while True:
        try:
            value = done.get_nowait()
        except Empty:
            return results
        value['collected_at'] = perf_counter()
        audit.write(json.dumps(value, ensure_ascii=False)+'\n')
        # 原票には入力全体を残し、表示所有者へは未使用の入力複製を送らない。
        results.append({k: v for k, v in value.items() if k != 'item'} if compact else value)


def worker(connection: Any, assets: tuple[Path, ...], output: Path) -> None:
    from src.phase_j.live_power import apply_power_policy
    apply_power_policy()
    import os
    from .live_obs_cpu import configure_ama
    configure_ama()
    os.environ['PUYO_B3_M3'] = '0'
    os.environ['PUYO_B3_NET_LEDGER_COLS'] = '0'   # live_v49f_async と同じ理由 (M3=0 の子へ正味の列を持ち込まない)
    from .live_ama_inputs import AmaInputs
    from .live_obs_audit import open_audit
    from src.ama_tsumo_seed_v5 import load_table
    import puyo_core
    jobs, done = Queue(1), Queue()
    thread = Thread(target=calculator, args=(jobs, done, *assets[:2]), daemon=True)
    try:
        table = load_table(assets[2])
        thread.start()
        connection.send(dict(ready=True))
        identity, inputs, dropped = None, None, 0
        with closing(open_audit(output/'ama_updates.jsonl', 'x')) as audit, input_audit(output) as trace:
            while (packet := connection.recv()) is not None:
                if trace is not None:
                    trace.write(json.dumps(packet,ensure_ascii=False)+'\n')
                now = perf_counter()
                current = (packet['generation'], packet['game'])
                if current != identity:
                    identity, inputs = current, AmaInputs(puyo_core, table)
                item = inputs.observe(packet)
                if item is not None:
                    job = dict(item=item, generation=packet['generation'], game=packet['game'],
                               frame=packet['id'], requested_at=packet['submitted_at'],
                               state_received_at=now, state_completed_at=perf_counter())
                    dropped += replace_latest(jobs, job)
                results = collect_results(done, audit, os.environ.get('PUYO_OBS_FAST_AUDIT') == '1')
                connection.send(dict(results=results, dropped=dropped, waiting=jobs.qsize(),
                    state_wait_ms=(now-packet['submitted_at'])*1000,
                    states=inputs.projection.states,
                    input_status=input_status(inputs, item, packet['t'])))
    except BaseException as exc:
        connection.send(dict(fatal=f'{type(exc).__name__}: {exc}', stack=traceback.format_exc()))
    finally:
        replace_latest(jobs, None)
        thread.join(10.)
        connection.close()


class AsyncAma:
    def __init__(self, assets: tuple[Path, ...], output: Path) -> None:
        self.status_path = output/'ama_async_status.json'
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, assets, output), daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(START_TIMEOUT):
            self.close()
            raise TimeoutError('ama入力processの起動待ち')
        response = self.connection.recv()
        if response.get('fatal'):
            self.close()
            raise RuntimeError(response)
        self.inputs, self.outputs = Queue(OBSERVATION_CAPACITY), Queue()
        self.failure_path = output/'ama_failure.json'
        self.thread = Thread(target=self.pump, daemon=True)
        self.thread.start()

    def pump(self) -> None:
        try:
            while (packet := self.inputs.get()) is not None:
                self.connection.send(packet)
                reply = self.connection.recv()
                for job in reply.get('results', []):
                    job['received_at'] = perf_counter()
                self.outputs.put(dict(reply, generation=packet['generation'], game=packet['game']))
                if reply.get('fatal'):
                    try:
                        self.failure_path.write_text(json.dumps(dict(reply, packet=packet),
                            ensure_ascii=False, indent=2), encoding='utf-8')
                    except OSError:
                        pass
                    return
        except Exception as exc:
            self.outputs.put(dict(fatal=f'{type(exc).__name__}: {exc}'))

    def submit(self, packet: dict) -> None:
        self.inputs.put_nowait(dict(packet, submitted_at=perf_counter()))

    def poll(self) -> list[dict]:
        values = []
        while True:
            try:
                values.append(self.outputs.get_nowait())
            except Empty:
                return values

    def close(self) -> None:
        if hasattr(self, 'inputs'):
            self.inputs.put(None, timeout=5.)
            self.thread.join(10.)
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(15.)
        forced = self.process.is_alive()
        if forced:
            self.process.terminate()
            self.process.join(5.)
        self.connection.close()
        if os.environ.get('PUYO_OBS_FAST_AUDIT') == '1':
            self.status_path.write_text(json.dumps(dict(
                forced_termination=forced, exit_code=self.process.exitcode,
                pump_alive=hasattr(self, 'thread') and self.thread.is_alive()), indent=2), encoding='utf-8')
