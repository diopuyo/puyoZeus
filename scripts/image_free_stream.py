"""動画の連続履歴を保つ旧/新単独ワーカー。全出力と復号同一性を保存する。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

from scripts import compare_image_free_all as comparison
from scripts.image_free_cache import CachedOld, cache_key, identity, file_sha
from scripts.image_free_timing import Timings, instrument

FRAME_SIZE = (1920, 1080)


def protect_outputs(paths: list[str]) -> None:
    roots = [Path(p).resolve() for p in paths]
    def audit(event: str, args: tuple) -> None:
        targets = []
        if event == 'open' and (args[2] or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            targets = [args[0]]
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime'):
            targets = [args[0]]
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            targets = list(args[:2])
        for value in targets:
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(value)).resolve()
                if any(path.is_relative_to(root) for root in roots):
                    raise PermissionError(f'既存出力への書込禁止: {path}')
    sys.addaudithook(audit)


def detach_logs(runtime: Any) -> None:
    """復元CPの旧出力先へ追記しない。判断に使われない補助sinkだけを外す。"""
    recorder = getattr(runtime.state['pipe'], '_placement_reconcile', None)
    if recorder is not None:
        recorder.sink = lambda row: None


def load_runtime(job: dict, config: dict, output: Path) -> Any:
    from scripts.image_free_runtime import Runtime
    from joblib.externals import cloudpickle
    if job.get('seed'):
        seed = Path(job['seed'])
        if file_sha(seed) != job['seed_sha256']:
            raise ValueError('前史CPのSHAが違います')
        with seed.open('rb') as stream:
            saved = cloudpickle.load(stream)
        if saved['frame'] != job['start_frame']:
            raise ValueError('前史CPの再開フレームが違います')
        runtime = saved['runtimes'][job.get('seed_side', int(job['route'] == 'new'))]
    else:
        if job['start_frame'] != 0:
            raise ValueError('通し比較の途中開始には前史CPが必要です')
        runtime = Runtime(Path(job['path']), output, config)
    detach_logs(runtime)
    return runtime


def decoded(job: dict, start: int, timing: Timings) -> Iterator[tuple]:
    import cv2
    cap = cv2.VideoCapture(job['path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if fps <= 0 or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
        raise RuntimeError('復号開始位置が違います')
    try:
        for index in range(start, job['end_frame']):
            with timing.measure('decode'):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'復号失敗: {index}')
                frame = cv2.resize(frame, FRAME_SIZE)
                stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield index, frame, fps, stamp
    finally:
        cap.release()


def fingerprint(job: dict, config: dict, output: Path) -> tuple[str, str]:
    record = identity(Path(job['path']), Path(__file__).resolve().parents[1], config)
    record.update(format='continuous_full_outputs_v1', origin=job['start_frame'],
                  seed_sha256=job.get('seed_sha256'))
    key = cache_key(record)
    comparison.save_json(output/'old_cache_identity.json', dict(key=key, **record))
    combined = hashlib.sha256((key+comparison.identity(job, config)).encode()).hexdigest()
    return combined, key


def restore(job: dict, config: dict, output: Path, signature: str, old_key: str) -> dict:
    from joblib.externals import cloudpickle
    manifest = output/'checkpoint.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['fingerprint'] != signature:
            raise ValueError('通し経路のコード・モデル・入力が変更されました')
        with (output/saved['state']).open('rb') as stream:
            state = cloudpickle.load(stream)
    else:
        with comparison.mode(job['route'] == 'new'):
            runtime = load_runtime(job, config, output)
        if job['route'] == 'old':
            runtime = CachedOld(runtime, Path(config['old_cache'])/old_key, old_key)
        state = dict(frame=job['start_frame'], runtimes=[runtime], fingerprint=signature,
                     seconds=0., processed_frames=0)
    return state


def report(job: dict, output: Path, state: dict, timing: Timings, complete: bool) -> None:
    comparison.save_json(output/'progress.json', dict(frames=state['frame'],
        expected_frames=job['end_frame'], start_frame=job['start_frame'],
        processed_frames=state['processed_frames'], seconds=state['seconds'],
        complete=complete, comparison_mode='continuous_routes', route=job['route'],
        video_id=job['video_id'], continuous_history=True, aggregate_pending=True,
        candidate_image_reads=0, timings=timing.snapshot(), old_cache=comparison.cache_metrics(state),
        pid=os.getpid(), updated_at=time.time()))


def run(job: dict, config: dict, guard: Any) -> None:
    protect_outputs(config.get('protected_outputs', []))
    output = Path(config['output'])/job['id']
    output.mkdir(exist_ok=True)
    signature, old_key = fingerprint(job, config, output)
    state = restore(job, config, output, signature, old_key)
    timing, rows = Timings(), []
    runtime = state['runtimes'][0]
    target = runtime.runtime if isinstance(runtime, CachedOld) else runtime
    detach_logs(target)
    began, last, prior = time.monotonic(), time.monotonic(), state['seconds']
    if not (output/'checkpoint.json').exists():
        comparison.checkpoint(output, state, [])
    with comparison.mode(job['route'] == 'new'), instrument(target, timing):
        for index, frame, fps, stamp in decoded(job, state['frame'], timing):
            guard.checkpoint()
            active = runtime.runtime if isinstance(runtime, CachedOld) else runtime
            detach_logs(active)
            value = runtime.step(frame, index, fps)
            with timing.measure('audit'):
                pixel_sha = hashlib.sha256(frame.data).hexdigest()
            rows.append(dict(frame=index, fps=fps, pts_ms=stamp, pixel_sha256=pixel_sha, value=value))
            state.update(frame=index+1, processed_frames=state['processed_frames']+1,
                         seconds=prior+time.monotonic()-began)
            if state['frame'] % comparison.CHECKPOINT_FRAMES == 0:
                with timing.measure('save'):
                    comparison.checkpoint(output, state, rows)
                rows.clear()
            if time.monotonic()-last >= comparison.PROGRESS_SECONDS:
                report(job, output, state, timing, False)
                last = time.monotonic()
    with timing.measure('save'):
        comparison.checkpoint(output, state, rows)
    state['seconds'] = prior+time.monotonic()-began
    report(job, output, state, timing, True)
