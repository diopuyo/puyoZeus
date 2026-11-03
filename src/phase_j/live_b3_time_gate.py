"""B3_PREREG §8の事後集計用原票。既定OFF、時間計測と判定入力を分離する。"""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import platform
from threading import Event, Thread, Lock
from time import perf_counter, time
from typing import Any
from . import live_stage_timing as timing

ACTIVE = None


def published(row: dict, snapshot: Any, now: float) -> None:
    if ACTIVE is not None:
        payload = snapshot.to_mapping()
        extra = {}
        if 'winrate_switch' in payload['evaluations']:
            extra['winrate_switch'] = payload['evaluations']['winrate_switch']
        ACTIVE.write(dict(kind='publication', published_at=now, frame=row['frame'],
            t_sec=row['t_sec'], game=row.get('game'), captured_at=row.get('captured_at'),
            state_t_sec=row.get('b3_state_t_sec'), state_frame=row.get('b3_state_frame'),
            match_active=row.get('match_active', True), queue_depth=row.get('queue_depth'),
            display=payload['display'], practical=payload['evaluations']['practical'],
            ama=payload['evaluations'].get('ama_plain'), **extra))


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                *[(name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
                  'page_available', 'virtual_total', 'virtual_available', 'extended')]]


class TimeGate:
    def __init__(self, output: Path) -> None:
        global ACTIVE
        self.lock = Lock()
        ACTIVE = self
        from .live_obs_audit import open_audit
        self.log = open_audit(output/'b3_time_gate.jsonl', 'x')
        self.stop = Event()
        self.thread = Thread(target=self.resources, args=(output,), daemon=True)
        self.thread.start()
        self.write(dict(kind='start', pid=os.getpid(), platform=platform.platform(),
                        columns='full row including requested H / cutoff / null masks',
                        reference='B3_PREREG_2026-10-05 section 8', gpu_memory_source='nvidia-smi if available'))

    def write(self, row: dict) -> None:
        with timing.locked('time_gate_lock_wait', self.lock):
            if not self.log.closed:
                payload = timing.measure('time_gate_json', json.dumps, dict(monotonic=perf_counter(), **row),
                    default=lambda x: x.item() if hasattr(x, 'item') else str(x))+'\n'
                timing.measure('time_gate_write', self.log.write, payload)

    def result(self, result: dict, accepted: bool) -> None:
        self.write(dict(kind='result', accepted=accepted, **result))

    def publication(self, row: dict, engine: Any) -> None:
        self.write(dict(kind='evaluation_ready', t_sec=row['t_sec'], frame=row['frame'],
            captured_at=row.get('captured_at'), raw_probability=row.get('raw_probability'),
            source=row.get('source'), hold=row.get('hold'), match_active=row.get('match_active', True),
            state_t_sec=row.get('b3_state_t_sec'), state_frame=row.get('b3_state_frame'),
            queue_depth=row.get('queue_depth'),
            submitted=engine.submitted if engine else None,
            parent_queue=engine.inputs.qsize() if engine else 0))

    def resources(self, output: Path) -> None:
        from .live_process_stats import process_tree, gpu_memory
        with (output/'b3_resources.jsonl').open('x', encoding='utf-8', buffering=1) as stream:
            while not self.stop.is_set():
                value = dict(monotonic=perf_counter(), unix=time(), pid=os.getpid(),
                             cpu_times=list(os.times()), processes=process_tree(), gpu=gpu_memory())
                if os.name == 'nt':
                    memory = MemoryStatus()
                    memory.length = ctypes.sizeof(memory)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                        value.update(free_gib=memory.available/1024**3, total_gib=memory.total/1024**3)
                stream.write(json.dumps(value)+'\n')
                self.stop.wait(5.)

    def close(self) -> None:
        global ACTIVE
        ACTIVE = None
        self.stop.set()
        self.thread.join(6.)
        self.write(dict(kind='end'))
        self.log.close()
