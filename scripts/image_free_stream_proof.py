"""同居版の全項目を基準にし、別プロセスと途中CP再開を検証する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

from scripts import compare_image_free_all as comparison
from scripts.image_free_cache import file_sha, write_chunk
from scripts.image_free_metadata import leaf_differences
from scripts.image_free_segment_merge import rows
from scripts.image_free_stream import decoded, detach_logs, protect_outputs
from scripts.image_free_timing import Timings


def baseline(job: dict, config: dict, guard: Any) -> None:
    from joblib.externals import cloudpickle
    protect_outputs(config.get('protected_outputs', []))
    output = Path(config['output'])/job['id']
    output.mkdir(exist_ok=True)
    seed = Path(job['seed'])
    if file_sha(seed) != job['seed_sha256']:
        raise ValueError('基準CPのSHAが違います')
    with seed.open('rb') as stream:
        state = cloudpickle.load(stream)
    if state['frame'] != job['start_frame']:
        raise ValueError('基準の開始位置が違います')
    for runtime in state['runtimes']:
        detach_logs(runtime)
    state['fingerprint'] = job['seed_sha256']
    saved, timing, began, features = [], Timings(), time.monotonic(), [0, 0]
    for index, frame, fps, stamp in decoded(job, state['frame'], timing):
        guard.checkpoint()
        values = []
        for side, runtime in enumerate(state['runtimes']):
            with comparison.mode(bool(side)):
                values.append(runtime.step(frame, index, fps))
            features[side] += int(values[-1]['b3_features'] is not None)
        saved.append(dict(frame=index, fps=fps, pts_ms=stamp,
            pixel_sha256=hashlib.sha256(frame.data).hexdigest(), old=values[0], new=values[1]))
        state['frame'] = index+1
        if state['frame'] == job['mid_frame']:
            write_chunk(output/'mid.pkl', state)
        if state['frame'] % comparison.CHECKPOINT_FRAMES == 0:
            comparison.checkpoint(output, state, saved)
            saved.clear()
            comparison.save_json(output/'progress.json', dict(frames=state['frame'], complete=False,
                processed_frames=state['frame']-job['start_frame'], pid=__import__('os').getpid()))
    comparison.checkpoint(output, state, saved)
    comparison.save_json(output/'progress.json', dict(frames=state['frame'], complete=True,
        processed_frames=state['frame']-job['start_frame'], seconds=time.monotonic()-began,
        feature_rows=features, pid=__import__('os').getpid(), aggregate_pending=True))


def verify(config: dict) -> dict:
    output = Path(config['output'])
    reference = {r['frame']: r for r in rows(output/config['baseline_id'])}
    cases = []
    for job in config['videos']:
        if job['route'] == 'both':
            continue
        bad, checked, features = [], [], 0
        for row in rows(output/job['id']):
            expected = reference[row['frame']]
            target = dict(value=expected[job['route']], **{k: expected[k]
                for k in ('frame', 'fps', 'pts_ms', 'pixel_sha256')})
            differences = leaf_differences(target, row)
            if differences:
                bad.append(dict(frame=row['frame'], columns=differences))
            checked.append(row['frame'])
            features += int(row['value']['b3_features'] is not None)
        complete = checked == list(range(job['start_frame'], job['end_frame']))
        progress = json.loads((output/job['id']/'progress.json').read_text())
        cases.append(dict(id=job['id'], route=job['route'], frames=len(checked),
            start=job['start_frame'], end=job['end_frame'], feature_rows=features,
            mismatches=bad, passed=complete and not bad and features > 0,
            seconds=progress['seconds'], fps=len(checked)/progress['seconds'], timings=progress['timings']))
    return dict(comparison_mode='continuous_routes', passed=len(cases) >= 4 and all(c['passed'] for c in cases),
                cases=cases, continuity='同居版の連続前史CP→連続実計算を基準に、独立復号・旧新分離・途中CP復元を照合')
