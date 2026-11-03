"""ama-plainの配信・保存入力検収で共用する評価口。各側1回の読み。"""
from __future__ import annotations
import json
import os
from pathlib import Path
from time import perf_counter, process_time
from src.ama_plain import calculate
from src.ama_plain_reference import Reference
from src.prefire_worker import PrefireInput
from src.production_config import RELEASE_LIVE_ENV_DEFAULTS

DEFAULT_RUST_THREADS = 2


def configured_rust_threads(explicit: int | None = None) -> int:
    if explicit is not None:
        if explicit < 0:
            raise ValueError('rust_threadsは0以上が必要（0は既存exe）')
        return explicit
    if os.environ.get('PUYO_OBS_AMA_RUST') != '1':
        return 0
    threads = int(os.environ.get('PUYO_OBS_AMA_RUST_THREADS', str(DEFAULT_RUST_THREADS)))
    if threads < 1:
        raise ValueError('PUYO_OBS_AMA_RUST_THREADSは1以上が必要')
    return threads


def request_from_saved(value: dict) -> PrefireInput:
    fields = dict(value)
    for name in ('evidence_times', 'colors', 'pending', 'stable'):
        fields[name] = tuple(fields[name])
    fields['boards'] = tuple(tuple(b) for b in fields['boards'])
    fields['known'] = tuple(tuple(tuple(p) for p in q) for q in fields['known'])
    return PrefireInput(**fields)


class AmaEngine:
    def __init__(self, binary: Path, model: Path, rust_threads: int | None = None) -> None:
        self.model = json.loads(model.read_text(encoding='utf-8'))
        self.rust_threads = configured_rust_threads(rust_threads)
        override = ''
        if not self.rust_threads and not binary.is_file():
            self.rust_threads = int(RELEASE_LIVE_ENV_DEFAULTS['PUYO_OBS_AMA_RUST_THREADS'])
            override = ' reference executable absent; override disabled Rust'
        if self.rust_threads:
            from src.ama_rust_reference import NativeReference
            self.references = [NativeReference(self.rust_threads), NativeReference(self.rust_threads)]
            print(f'AMA_BACKEND native threads={self.rust_threads}{override}', flush=True)
            return
        self.references = [Reference(binary), Reference(binary)]
        if os.name == 'nt':
            from .live_obs_cpu import configure_ama
            for reference in self.references:
                configure_ama(int(reference.process._handle))

    def cpu_time(self) -> float:
        if self.rust_threads:
            return process_time()
        if os.name != 'nt':
            return process_time()+sum(r.cpu_sec for r in self.references)
        from .live_process_stats import details
        return process_time()+sum(details(dict(pid=r.process.pid)).get('cpu_sec',0.) for r in self.references)

    def evaluate(self, item: dict) -> dict:
        began = perf_counter()
        req = item['request']
        result = dict(index=item['index'], input_sec=req['input_sec'], states=item['states'],
                      projection=item['projection'], source_kind=item['source_kind'])
        if item.get('missing'):
            return dict(result, missing=item['missing'])
        cpu_before = self.cpu_time()
        value = calculate(request_from_saved(req), item['carry'], self.model,
                          enabled=True, references=self.references, all_clear=item['all_clear'])
        return dict(result, value={k: value[k] for k in ('plans', 'pending', 'probability')},
                    timings={k: value[k] for k in ('setup_sec', 'read_pair_wall_sec', 'conversion_sec',
                        'read_wall_sec', 'read_cpu_sec', 'reused', 'reference_calls')},
                    compute_ms=(perf_counter()-began)*1000,
                    cpu_sec=self.cpu_time()-cpu_before)

    def close(self) -> None:
        for ref in self.references:
            ref.close()
