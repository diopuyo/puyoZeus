"""保存済み前史から現行/高速版を独立実行し、途中CP復元もbit単位で照合する。"""
from __future__ import annotations

from contextlib import nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Any

from image_free_audit_checkpoint import cache_provenance
from image_free_audit_reuse import reuse_audit
from image_free_reference_fast import fast_reference

PROOF_FRAMES = 180
MID_FRAMES = 90


def differences(left: Any, right: Any, prefix: str = '') -> list[str]:
    if type(left) is not type(right):
        return [prefix+':type']
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [prefix+':keys']
        return [item for k in left for item in differences(left[k], right[k], prefix+'/'+str(k))]
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix+':length']
        return [item for i, (a, b) in enumerate(zip(left, right)) for item in differences(a, b, prefix+'/'+str(i))]
    if isinstance(left, float):
        return [] if struct.pack('d', left) == struct.pack('d', right) else [prefix]
    return [] if left == right else [prefix]


def original_state(job: dict, config: dict) -> dict:
    from joblib.externals import cloudpickle
    spec = config['cases'][job['id']]
    with Path(spec['seed']).open('rb') as file:
        state = cloudpickle.load(file)
    runtime = state['runtimes'][spec['seed_side']]
    if hasattr(runtime, 'cached') and runtime.cached is not None:
        raise ValueError('キャッシュ読出し途中のCPは検証対象にできません')
    runtime = getattr(runtime, 'runtime', runtime)
    return dict(frame=state['frame'], runtimes=[runtime], fingerprint=state['fingerprint'])


def calculate(job: dict, state: dict, end: int, timing: Any, guard: Any) -> list[dict]:
    from scripts.image_free_stream import decoded, detach_logs
    from scripts.image_free_timing import instrument
    runtime = state['runtimes'][0]
    target = getattr(runtime, 'runtime', runtime)
    detach_logs(target)
    rows = []
    with instrument(target, timing):
        for index, frame, fps, stamp in decoded(dict(job, end_frame=end), state['frame'], timing):
            guard.checkpoint()
            value = runtime.step(frame, index, fps)
            rows.append(dict(frame=index, fps=fps, pts_ms=stamp,
                pixel_sha256=hashlib.sha256(frame.data).hexdigest(), value=value))
            state['frame'] = index+1
    return rows


def candidate(job: dict, config: dict, timing: Any, guard: Any) -> tuple[list[dict], dict]:
    from scripts import image_free_stream as stream
    output = Path(config['output'])/job['id']
    spec = config['cases'][job['id']]
    with cache_provenance(config['control_sha256'], 'history-stage3', tuple(config['predecessors'])):
        signature, key = stream.fingerprint(job, config, output)
        state = stream.restore(job, config, output, signature, key)
        start = state['frame']
        with fast_reference():
            first = calculate(job, state, start+MID_FRAMES, timing, guard)
            stream.comparison.checkpoint(output, state, first)
            state = stream.restore(job, config, output, signature, key)
            second = calculate(job, state, start+PROOF_FRAMES, timing, guard)
            stream.comparison.checkpoint(output, state, second)
    return first+second, dict(initial=start, own_checkpoint=start+MID_FRAMES,
        final=state['frame'], migrated=bool(spec.get('continuous_cp')))


def run(job: dict, config: dict, guard: Any) -> None:
    from scripts import image_free_stream as stream, compare_image_free_all as comparison
    from scripts.image_free_timing import Timings
    stream.protect_outputs(config['protected_outputs'])
    output = Path(config['output'])/job['id']
    phases, records = {}, {}
    for mode in ('baseline', 'candidate'):
        timing, started = Timings(), time.perf_counter()
        reused = mode == 'candidate' or config.get('baseline_audit', True)
        audit = reuse_audit() if job['route'] == 'old' and reused else nullcontext()
        with comparison.mode(job['route'] == 'new'), audit:
            if mode == 'baseline':
                state = original_state(job, config)
                began = time.perf_counter()
                rows = calculate(job, state, state['frame']+PROOF_FRAMES, timing, guard)
                del state
                import gc
                gc.collect()
            else:
                began = time.perf_counter()
                rows, resume = candidate(job, config, timing, guard)
        seconds = sum(timing.seconds.values())
        phases[mode] = dict(frames=len(rows), compute_seconds=seconds, fps=len(rows)/seconds,
            wall_seconds=time.perf_counter()-started, timings=timing.snapshot(), feature_rows=sum(r['value']['b3_features'] is not None for r in rows))
        with gzip.open(output/(mode+'.json.gz'), 'wt') as file:
            json.dump(rows, file, ensure_ascii=False)
        records[mode] = rows
    bad = [dict(frame=a['frame'], columns=differences(a, b)) for a, b in zip(records['baseline'], records['candidate']) if differences(a, b)]
    passed = not bad and all(p['frames'] == PROOF_FRAMES for p in phases.values())
    comparison.save_json(output/'receipt.json', dict(passed=passed, phases=phases, resume=resume,
        mismatches=bad, source_unchanged=True, float_comparison='IEEE754 binary64 bits', route=job['route']))
    comparison.save_json(output/'progress.json', dict(frames=resume['final'], processed_frames=PROOF_FRAMES*2,
        complete=True, pid=__import__('os').getpid(), proof_passed=passed))
    if not passed:
        raise ValueError('段3全項目照合が不一致です')
