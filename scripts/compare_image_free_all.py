"""全区間・全連続フレームの比較。状態と差分を一体でチェックポイント化する。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import gzip
import json
import os
from pathlib import Path
import time
from typing import Iterator
from unittest.mock import patch

import cv2
from joblib.externals import cloudpickle

from scripts.obs_priority_guard import ObsPriorityGuard

CHECKPOINT_FRAMES = 300
PROGRESS_SECONDS = 20


def local_path(value: str) -> str:
    if os.name == 'nt' and value.startswith('/mnt/') and len(value) > 7:
        return value[5].upper() + ':' + value[6:]
    if os.name != 'nt' and len(value) > 2 and value[1] == ':':
        return '/mnt/' + value[0].lower() + value[2:].replace('\\', '/')
    return value


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    for key, value in list(config.items()):
        if isinstance(value, str):
            config[key] = local_path(value)
    for video in config['videos']:
        video['path'] = local_path(video['path'])
    return config


@contextmanager
def mode(candidate: bool) -> Iterator[None]:
    """候補側では画像読込そのものを禁止し、隠れたフォールバックを検出する。"""
    attempts = []
    def forbidden(*args: object, **kwargs: object) -> None:
        attempts.append(str(args[0]))
        raise RuntimeError(f'候補経路でゲーム画像読込を検出: {args[0]}')
    with patch.dict(os.environ, dict(PUYO_IMAGE_FREE_ALL=str(int(candidate)),
                                    PUYO_IMAGE_FREE_SCORE_OCR='0', PUYO_IMAGE_FREE_UI_MASK='0')):
        if candidate:
            from PIL import Image
            with patch.object(cv2, 'imread', forbidden), patch.object(Image, 'open', forbidden):
                yield
            if attempts:
                raise RuntimeError(f'候補経路が画像読込失敗を握りつぶしました: {attempts}')
        else:
            yield


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def checkpoint(output: Path, state: dict, differences: list) -> None:
    """差分と再開状態を書き切ってからコミット点を更新する。"""
    generation = f"{state['frame']:09d}"
    previous = output / 'checkpoint.json'
    old = json.loads(previous.read_text()) if previous.exists() else None
    if old and old['frame'] == state['frame']:
        return
    data_path = output / f'{generation}.differences.json.gz'
    temporary_data = data_path.with_suffix('.tmp')
    with gzip.open(temporary_data, 'wt', encoding='utf-8') as stream:
        json.dump(dict(rows=differences), stream, ensure_ascii=False)
    temporary_data.replace(data_path)
    target = output / f'{generation}.checkpoint.pkl'
    temporary = target.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        cloudpickle.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    save_json(previous, dict(frame=state['frame'], state=target.name,
        differences=data_path.name, chunks=(old['chunks'] if old else [])+[data_path.name],
        fingerprint=state['fingerprint']))
    if old and old['state'] != target.name:
        (output / old['state']).unlink()


def compare(values: list[dict], state: dict, index: int, fps: float) -> dict | None:
    observed = state.setdefault('observed_feature_rows', [0, 0])
    games = state.setdefault('observed_games', [[], []])
    for side, value in enumerate(values):
        observed[side] += int(value.get('b3_features') is not None)
        game = value.get('game', {})
        if game.get('active') and game['index'] not in games[side]:
            games[side].append(game['index'])
    keys = [k for k in values[0] if values[0][k] != values[1][k]]
    for key in keys:
        state['counts'][key] = state['counts'].get(key, 0) + 1
    changed = 'displayed_probability' in keys
    state['display_scenes'] += int(changed and not state['previous_display_difference'])
    state['previous_display_difference'] = changed
    text_changed = 'display_text' in keys
    state['display_text_scenes'] = state.get('display_text_scenes', 0) + int(
        text_changed and not state.get('previous_text_difference', False))
    state['previous_text_difference'] = text_changed
    if not keys:
        return None
    return dict(frame=index, t_sec=index/fps, fields=keys,
                old={k: values[0][k] for k in keys}, new={k: values[1][k] for k in keys})


def report(output: Path, state: dict, expected: int, complete: bool = False) -> None:
    save_json(output / 'progress.json', dict(frames=state['frame'], expected_frames=expected,
        start_frame=state.get('start_frame', 0), processed_frames=state['frame']-state.get('start_frame', 0),
        differences=state['counts'], display_changed_scenes=state['display_scenes'],
        display_text_changed_scenes=state.get('display_text_scenes', 0),
        observed_feature_rows=state.get('observed_feature_rows', [0, 0]), candidate_image_reads=0,
        old_cache=cache_metrics(state),
        observed_games=state.get('observed_games', [[], []]),
        complete=complete, checkpoint_frame=json.loads((output/'checkpoint.json').read_text())['frame'],
        pid=os.getpid(), updated_at=time.time(), all_consecutive=True, gate_passed=False))


def cache_metrics(state: dict) -> dict:
    runtime = state.get('runtimes', [None])[0]
    return {key: getattr(runtime, key) for key in ('hits', 'misses', 'rejected', 'computed')
            if hasattr(runtime, key)}


def initial(video: dict, output: Path, config: dict, fingerprint: str) -> dict:
    from scripts.image_free_runtime import Runtime
    runtimes = []
    for candidate, name in ((False, 'old'), (True, 'new')):
        directory = output / name
        directory.mkdir(exist_ok=True)
        with mode(candidate):
            runtimes.append(Runtime(Path(video['path']), directory, config))
    return dict(frame=int(video.get('start_frame', 0)), start_frame=int(video.get('start_frame', 0)),
                runtimes=runtimes, counts={}, display_scenes=0,
                previous_display_difference=False, fingerprint=fingerprint)


def restore(video: dict, output: Path, config: dict, fingerprint: str) -> dict:
    manifest = output / 'checkpoint.json'
    if not manifest.exists():
        return initial(video, output, config, fingerprint)
    record = json.loads(manifest.read_text())
    if record['fingerprint'] != fingerprint:
        raise ValueError('動画・コード・モデルが変更されたため別出力先が必要です')
    with (output / record['state']).open('rb') as stream:
        return cloudpickle.load(stream)


def run_video(video: dict, config: dict, guard: ObsPriorityGuard, limit: int | None) -> None:
    output = Path(config['output']) / video['id']
    output.mkdir(exist_ok=True)
    fingerprint = identity(video, config)
    state = restore(video, output, config, fingerprint)
    attach_old_cache(video, config, output, state)
    cap = cv2.VideoCapture(video['path'])
    expected, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if expected <= 0 or fps <= 0:
        raise RuntimeError(f'動画を開けません: {video}')
    expected = min(expected, int(video.get('end_frame', expected)))
    if not 0 <= int(video.get('start_frame', 0)) < expected:
        raise ValueError('比較区間が不正です')
    if state['frame']:
        cap.set(cv2.CAP_PROP_POS_FRAMES, state['frame'])
        if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != state['frame']:
            raise RuntimeError('再開位置が一致しません')
    if not (output / 'checkpoint.json').exists():
        checkpoint(output, state, [])
    report(output, state, expected)
    differences, last = [], time.monotonic()
    while state['frame'] < expected and (limit is None or state['frame'] < limit):
        guard.checkpoint()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"途中でデコード失敗: {state['frame']}/{expected}")
        frame = cv2.resize(frame, (1920, 1080))
        values = []
        for candidate, runtime in enumerate(state['runtimes']):
            with mode(bool(candidate)):
                values.append(runtime.step(frame, state['frame'], fps))
        difference = compare(values, state, state['frame'], fps)
        if difference is not None:
            differences.append(difference)
        state['frame'] += 1
        if state['frame'] % CHECKPOINT_FRAMES == 0:
            checkpoint(output, state, differences)
            differences.clear()
        if time.monotonic() - last >= PROGRESS_SECONDS:
            report(output, state, expected)
            last = time.monotonic()
    checkpoint(output, state, differences)
    report(output, state, expected, state['frame'] == expected)
    cap.release()


def attach_old_cache(video: dict, config: dict, output: Path, state: dict) -> None:
    """明示された新実行だけで旧キャッシュを利用する。旧出力は変更しない。"""
    if not config.get('old_cache'):
        return
    from scripts.image_free_cache import CachedOld, cache_key, identity
    if video.get('start_frame', 0):
        raise ValueError('途中初期化の旧キャッシュは同等性未証明のため使用禁止')
    record = identity(Path(video['path']), Path(__file__).resolve().parents[1], config)
    key = cache_key(record)
    if isinstance(state['runtimes'][0], CachedOld):
        if state['runtimes'][0].key != key:
            raise ValueError('旧キャッシュの内容SHAが変更されたため新出力が必要')
        return
    save_json(output / 'old_cache_identity.json', dict(key=key, **record))
    state['runtimes'][0] = CachedOld(state['runtimes'][0], Path(config['old_cache']) / key, key)


def identity(video: dict, config: dict) -> str:
    digest = hashlib.sha256(json.dumps(video, sort_keys=True).encode())
    root = Path(__file__).resolve().parents[1]
    for base in ('src', 'scripts'):
        for path in sorted((root/base).rglob('*.py')):
            digest.update(path.read_bytes())
    for key in ('digit_student.joblib', 'ui.joblib'):
        digest.update((Path(config['output'])/key).read_bytes())
    for key in ('mask_model', 'b3_model', 'b3_table'):
        digest.update(Path(config[key]).read_bytes())
    stat = Path(video['path']).stat()
    digest.update(str((stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    os.chdir(config['asset_root'])
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)
    output = Path(config['output'])
    os.environ.update(PUYO_IMAGE_FREE_UI_MODEL=str(output/'ui.joblib'),
        PUYO_IMAGE_FREE_DIGIT_MODEL=str(output/'digit_student.joblib'),
        PUYO_IMAGE_FREE_UI_MASK_MODEL=config['mask_model'], PUYO_B3_JUDGMENT='1')
    guard = ObsPriorityGuard()
    guard.checkpoint(force=True)
    for video in config['videos']:
        save_json(output/'job.json', dict(video=video['id'], status='running', pid=os.getpid()))
        run_video(video, config, guard, args.limit)
    reports = {v['id']: json.loads((output/v['id']/'progress.json').read_text()) for v in config['videos']}
    save_json(output/'result.json', dict(videos=reports,
        complete=all(r['complete'] for r in reports.values()),
        display_changed_scenes=sum(r['display_changed_scenes'] for r in reports.values()),
        display_text_changed_scenes=sum(r['display_text_changed_scenes'] for r in reports.values()),
        gate_passed=False, verdict='統括による合格判定待ち'))
    save_json(output/'job.json', dict(status='complete' if args.limit is None else 'smoke_complete'))


if __name__ == '__main__':
    main()
