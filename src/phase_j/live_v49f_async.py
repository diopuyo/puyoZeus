"""v4.9f専用process。通知はFIFO、公開可能時刻は実際の計算完了後に記録する。"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
from pathlib import Path
from queue import Empty
from time import perf_counter
import traceback
from typing import Any
from contextlib import closing

MS = 1000.
START_TIMEOUT = 120.
QUEUE_CAPACITY = 256
CLOSE_TIMEOUT = 10.
AUDIT_CLOSE_TIMEOUT = 120.


def protect_worker_interrupt() -> None:
    """既存の停止保護がONなら、Ctrl+Cで先に死なず親のFIFO終端まで処理する。"""
    from src.b3_async import graceful_stop_enabled
    if graceful_stop_enabled():
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def worker(connection: Any, assets: tuple[Path, ...], output: Path) -> None:
    """原票と実際の待ち時間を別ファイルに逐次保存する。"""
    from src.phase_j.live_power import apply_power_policy
    apply_power_policy()
    os.environ['PUYO_B3_M3'] = '0'
    # 正味の列は M3=1 が前提。親から受け継ぐと列仕様の組み立てが例外で止まるので、M3=0 のこの子では明示的に OFF。
    os.environ['PUYO_B3_NET_LEDGER_COLS'] = '0'
    protect_worker_interrupt()   # 列仕様を読むimportは、v49f子の環境を確定した後にする。
    from threadpoolctl import threadpool_limits
    from .live_v49f_engine import load_engine
    from .live_v49f_origins import OriginStream
    from .live_obs_audit import open_audit
    from .live_x_direct import direct_publish_enabled
    # 段1 (既定OFF): 予測時間の例外門を記録だけにする。OFF は従来どおり超過で例外。
    record_budget = direct_publish_enabled()
    engine = None
    try:
        engine = load_engine(*assets)
        origins = OriginStream(engine.base.policy, engine.base.native, record_budget)
        generation = None
        connection.send({'ready': True})
        with threadpool_limits(limits=1), closing(open_audit(output/'v49f_updates.jsonl', 'x')) as audit:
            while (packet := connection.recv()) is not None:
                began = perf_counter()
                # B3の採用境界では発火入力はリセットされない。入力所有者の世代に合わせる。
                current_generation = packet.get('input_epoch', packet.get('generation', 0))
                if generation is not None and generation != current_generation:
                    engine.close()
                    engine = load_engine(*assets)
                    origins = OriginStream(engine.base.policy, engine.base.native, record_budget)
                generation = current_generation
                forecasts = origins.observe(packet['game'], packet['t'], packet.get('origins', []))
                origin_ms = (perf_counter()-began)*MS
                base, value = engine.observe(packet['t'], packet['game'], packet['raw'], forecasts,
                                             packet.get('firepower_ledgers'), packet.get('adopted_game'))
                completed = perf_counter()
                result = dict(id=packet['id'], game=packet['game'], t=packet['t'],
                    base=base, value=value, origins=forecasts, completed_at=completed,
                    origin_ms=origin_ms, compute_ms=(completed-began)*MS,
                    wait_ms=(began-packet['submitted_at'])*MS)
                if record_budget and (exceeded := origins.take_budget_exceeded()):
                    result['budget_exceeded_ms'] = exceeded
                audit.write(json.dumps(result)+'\n')
                connection.send(result)
    except BaseException as exc:
        connection.send(dict(fatal=f'{type(exc).__name__}: {exc}', stack=traceback.format_exc()))
    finally:
        if engine is not None:
            engine.close()
        connection.close()


class AsyncV49F:
    """送信キューの所有者。実装検証では同じprocessへ保存通知を順次配送する。"""

    def __init__(self, assets: tuple[Path, ...], output: Path) -> None:
        self.status_path = output/'v49f_async_status.json'
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, assets, output), daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(START_TIMEOUT):
            self.close()
            raise TimeoutError('v4.9f評価processの起動待ち')
        ready = self.connection.recv()
        if ready.get('fatal'):
            self.close()
            raise RuntimeError(ready)

    def submit(self, packet: dict) -> None:
        self.connection.send(dict(packet, submitted_at=perf_counter()))

    def poll(self) -> list[dict]:
        values = []
        while self.connection.poll():
            value = self.connection.recv()
            if value.get('fatal'):
                raise RuntimeError(value)
            values.append(value)
        return values

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(AUDIT_CLOSE_TIMEOUT if os.environ.get('PUYO_OBS_FAST_AUDIT') == '1'
                              else CLOSE_TIMEOUT)
        forced = self.process.is_alive()
        if forced:
            self.process.terminate()
            self.process.join(5.)
        self.connection.close()
        if os.environ.get('PUYO_OBS_FAST_AUDIT') == '1':
            self.status_path.write_text(json.dumps(dict(
                forced_termination=forced, exit_code=self.process.exitcode), indent=2), encoding='utf-8')
