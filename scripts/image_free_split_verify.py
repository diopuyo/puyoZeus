"""通しチェックポイントから継続した全項目と、助走付き分割を照合する。"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

FRAME_SIZE = (1920, 1080)
PROGRESS_FRAMES = 60
REFERENCE_SCHEMA = 2


def reference_seed(case: dict, output: Path) -> tuple:
    from joblib.externals import cloudpickle
    partial = output / (case['video']+'.reference.resume.pkl')
    if partial.exists():
        with partial.open('rb') as stream:
            saved = cloudpickle.load(stream)
        if saved['checkpoint'] != case['checkpoint'] or saved.get('schema') != REFERENCE_SCHEMA:
            raise ValueError('通し継続の基準チェックポイントが変わりました')
        return saved['runtime'], saved['frame'], saved['rows'], saved['origin']
    with Path(case['checkpoint']).open('rb') as stream:
        state = cloudpickle.load(stream)
    return state['runtimes'][0], state['frame'], [], state['frame']


def save_reference_seed(output: Path, case: dict, runtime: Any, frame: int,
                        rows: list, origin: int) -> None:
    from scripts.image_free_cache import write_chunk
    write_chunk(output/(case['video']+'.reference.resume.pkl'), dict(checkpoint=case['checkpoint'], schema=REFERENCE_SCHEMA,
                runtime=runtime, frame=frame, rows=rows, origin=origin))


def counters(runtime: Any) -> dict:
    pipe = runtime.state['pipe']
    return dict(lean_rows=pipe._b3_queues.index, games=pipe._b3_capture.game,
                adopted_games=pipe._b3_queues.shared.game_idx)


def observed(runtime: Any, frame: Any, index: int, fps: float) -> dict:
    """旧凍結版にないfix1監査列も、同じRuntimeの実観測から補う。"""
    value = runtime.step(frame, index, fps)
    pipe = runtime.state['pipe']
    value.setdefault('game', dict(index=pipe._b3_capture.game,
        active=bool(runtime.state['r'].is_match_active), adopted=pipe._b3_queues.current_game))
    if 'death_0' not in value['ui']:
        value['ui'].update({f'death_{i}': runtime.probes['death']._score(frame, None, i)
                            for i in range(2)})
    return value


def frames(video: str, start: int, end: int) -> Any:
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
        raise RuntimeError('復号開始位置が違います')
    try:
        for index in range(start, end):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'復号失敗: {index}')
            yield index, cv2.resize(frame, FRAME_SIZE), fps
    finally:
        cap.release()


def reference(case_group: list[dict], config: dict, output: Path, guard: Any) -> Path:
    from scripts.compare_image_free_all import mode, save_json
    video = next(v for v in config['videos'] if v['id'] == case_group[0]['video'])
    destination = output / (video['id'] + '.reference.json.gz')
    if destination.exists():
        with gzip.open(destination, 'rt') as stream:
            saved = json.load(stream)
        if saved.get('schema') != REFERENCE_SCHEMA:
            raise ValueError('旧形式の全項目基準です。fix1追加列の検証が必要です')
        return destination
    runtime, start, rows, origin = reference_seed(case_group[0], output)
    first, end = min(c['start'] for c in case_group), max(c['end'] for c in case_group)
    began = time.monotonic()
    with mode(False):
        for index, frame, fps in frames(video['path'], start, end):
            guard.checkpoint()
            value = observed(runtime, frame, index, fps)
            if index >= first:
                rows.append(dict(frame=index, value=value, counters=counters(runtime)))
            if index % PROGRESS_FRAMES == 0:
                save_reference_seed(output, case_group[0], runtime, index+1, rows, origin)
                save_json(output/'reference_progress.json', dict(video=video['id'], frame=index,
                    start=start, end=end, seconds=time.monotonic()-began))
    with gzip.open(destination.with_suffix('.tmp'), 'wt', encoding='utf-8') as stream:
        json.dump(dict(schema=REFERENCE_SCHEMA, start=origin, end=end, rows=rows), stream, ensure_ascii=False)
    destination.with_suffix('.tmp').replace(destination)
    del runtime
    gc.collect()
    return destination


def verify(case: dict, config: dict, reference_file: Path, output: Path, guard: Any) -> dict:
    from scripts.compare_image_free_all import mode, save_json
    from scripts.image_free_metadata import MetadataOffset, leaf_differences
    from scripts.image_free_runtime import Runtime
    with gzip.open(reference_file, 'rt') as stream:
        baseline = {r['frame']: r for r in json.load(stream)['rows']}
    video = next(v for v in config['videos'] if v['id'] == case['video'])
    directory = output / case['id']
    directory.mkdir(exist_ok=True)
    bad, offset, feature_rows, began = [], None, 0, time.monotonic()
    with mode(False):
        runtime = Runtime(Path(video['path']), directory, config)
        for index, frame, fps in frames(video['path'], case['warm_start'], case['end']):
            guard.checkpoint()
            actual = observed(runtime, frame, index, fps)
            if index < case['start']:
                continue
            expected = baseline[index]
            if offset is None:
                current = counters(runtime)
                offset = MetadataOffset(**{k: v-current[k] for k, v in expected['counters'].items()},
                    previous_adopted=expected['value'].get('game', {}).get('adopted'))
            differences = leaf_differences(expected['value'], offset.apply(actual))
            feature_rows += int(expected['value']['b3_features'] is not None)
            if differences:
                bad.append(dict(frame=index, columns=differences))
            if index % PROGRESS_FRAMES == 0:
                save_json(output/'split_progress.json', dict(case=case['id'], frame=index,
                    end=case['end'], mismatches=len(bad), seconds=time.monotonic()-began))
    result = dict(**case, complete_reference=True, offset=vars(offset),
        frames=case['end']-case['start'], feature_rows=feature_rows, mismatches=bad,
        passed=not bad and feature_rows > 0, seconds=time.monotonic()-began,
        schema=REFERENCE_SCHEMA, reference_sha256=hashlib.sha256(reference_file.read_bytes()).hexdigest())
    save_json(directory/'proof.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.previous/'source'))
    import scripts
    scripts.__path__ = [str(args.previous/'source/scripts'), *scripts.__path__]
    import cv2
    import torch
    from scripts.compare_image_free_all import read_config, save_json
    from scripts.obs_priority_guard import ObsPriorityGuard
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    config = read_config(args.previous/'config.json')
    os.chdir(config['asset_root'])
    os.environ['PUYO_B3_JUDGMENT'] = '1'
    cases = json.loads(args.cases.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    guard, results = ObsPriorityGuard(), []
    for video in dict.fromkeys(c['video'] for c in cases):
        group = [c for c in cases if c['video'] == video]
        saved = reference(group, config, args.output, guard)
        for case in group:
            results.append(verify(case, config, saved, args.output, guard))
            gc.collect()
            save_json(args.output/'full_proof.json', dict(cases=results,
                complete=len(results) == len(cases), passed=len(results) == len(cases)
                and len(results) >= 3 and all(r['passed'] for r in results)))


if __name__ == '__main__':
    main()
