"""助走を捨て、区間の所有フレームだけを全項目で永続化する比較ワーカー。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

from scripts import compare_image_free_all as compare
from scripts.image_free_cache import CachedOld, cache_key, identity
from scripts.image_free_split_verify import counters, frames


class RecordedRuntime:
    """キャッシュ命中中も、実際に保存した各フレームの番号を返す。"""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def step(self, frame: Any, index: int, fps: float) -> dict:
        value = self.runtime.step(frame, index, fps)
        return dict(value=value, counters=counters(self.runtime))


def segment_identity(video: dict, config: dict, output: Path | None = None) -> str:
    record = identity(Path(video['path']), Path(__file__).resolve().parents[1], config)
    # 終端と所有範囲は既計算の旧出力に影響しない。段3の前半も本走行へ再利用する。
    record['segment'] = dict(start_frame=video['start_frame'])
    record['format'] = 'paired_outputs_with_counters_v2'
    key = cache_key(record)
    if output is not None:
        compare.save_json(output/'old_cache_identity.json', dict(key=key, **record))
    return key


def initial(video: dict, config: dict, output: Path, fingerprint: str, old_key: str) -> dict:
    from scripts.image_free_runtime import Runtime
    runtimes = []
    for candidate, label in ((False, 'old'), (True, 'new')):
        directory = output / label
        directory.mkdir(exist_ok=True)
        with compare.mode(candidate):
            runtime = RecordedRuntime(Runtime(Path(video['path']), directory, config))
        if not candidate:
            runtime = CachedOld(runtime, Path(config['old_cache']) / old_key, old_key)
        runtimes.append(runtime)
    return dict(frame=video['start_frame'], runtimes=runtimes, fingerprint=fingerprint,
                last_counters=[dict(lean_rows=0, games=0, adopted_games=0) for _ in range(2)],
                owned_counters=None, processed_frames=0, seconds=0.)


def restore(video: dict, config: dict, output: Path, fingerprint: str, old_key: str) -> dict:
    from joblib.externals import cloudpickle
    path = output / 'checkpoint.json'
    if not path.exists():
        return initial(video, config, output, fingerprint, old_key)
    manifest = json.loads(path.read_text())
    if manifest['fingerprint'] != fingerprint:
        raise ValueError('分割区間・コード・資産が変更されています')
    with (output/manifest['state']).open('rb') as stream:
        return cloudpickle.load(stream)


def step(state: dict, frame: Any, index: int, fps: float, owned_start: int) -> dict | None:
    if index == owned_start:
        state['owned_counters'] = state['last_counters']
    envelopes = []
    for candidate, runtime in enumerate(state['runtimes']):
        with compare.mode(bool(candidate)):
            envelopes.append(runtime.step(frame, index, fps))
    state['last_counters'] = [e['counters'] for e in envelopes]
    state['frame'] = index + 1
    if index < owned_start:
        return None
    state['processed_frames'] += 1
    return dict(frame=index, old=envelopes[0]['value'], new=envelopes[1]['value'])


def report(output: Path, video: dict, state: dict, complete: bool) -> None:
    compare.save_json(output/'progress.json', dict(frames=state['frame'],
        start_frame=video['start_frame'], owned_start=video['owned_start'],
        expected_frames=video['end_frame'], processed_frames=state['processed_frames'],
        owned_counters=state['owned_counters'], end_counters=state['last_counters'],
        seconds=state['seconds'], complete=complete, differences={}, candidate_image_reads=0,
        display_changed_scenes=0, display_text_changed_scenes=0,
        canonical_aggregation_pending=True, old_cache=compare.cache_metrics(state),
        comparison_mode=state.get('comparison_mode'), continuous_equivalence=False,
        disclaimer='通しとの同等性は不成立・対の比較' if state.get('comparison_mode') == 'paired_cold' else None,
        pid=__import__('os').getpid(), updated_at=time.time()))


def run_segment(video: dict, config: dict, guard: Any) -> None:
    output = Path(config['output']) / video['id']
    output.mkdir(exist_ok=True)
    if not 0 <= video['start_frame'] <= video['owned_start'] < video['end_frame']:
        raise ValueError('分割区間が不正です')
    old_key = segment_identity(video, config, output)
    fingerprint = hashlib.sha256((old_key+compare.identity(video, config)).encode()).hexdigest()
    state = restore(video, config, output, fingerprint, old_key)
    state['comparison_mode'] = config.get('comparison_mode')
    began, last, rows = time.monotonic(), time.monotonic(), []
    prior_seconds = state['seconds']
    if not (output/'checkpoint.json').exists():
        compare.checkpoint(output, state, [])
    for index, frame, fps in frames(video['path'], state['frame'], video['end_frame']):
        guard.checkpoint()
        row = step(state, frame, index, fps, video['owned_start'])
        if row is not None:
            rows.append(row)
        state['seconds'] = prior_seconds + time.monotonic()-began
        if state['frame'] % compare.CHECKPOINT_FRAMES == 0:
            compare.checkpoint(output, state, rows)
            rows.clear()
        if time.monotonic()-last >= compare.PROGRESS_SECONDS:
            report(output, video, state, False)
            last = time.monotonic()
    compare.checkpoint(output, state, rows)
    report(output, video, state, True)
