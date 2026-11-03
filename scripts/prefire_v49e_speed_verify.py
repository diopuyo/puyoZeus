"""新nativeでv4.8を再生成し、時間以外の最初の差で停止する検収専用実行。"""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
from typing import Any
from scripts.prefire_v49e_speed_accounting import sources, rows, sha, VIDEOS

GIB = 1024 ** 3
MIN_AVAILABLE = 7 * GIB
PROGRESS = 100
TIME_KEYS = frozenset(('ready_sec', 'publish_sec', 'busy_until'))


def stable(value: Any, path: tuple = ()) -> Any:
    """入力・要求・証拠・ゲーム経過の時刻と予算判定は保持する。"""
    if isinstance(value, dict):
        return {k: stable(v, (*path, k)) for k, v in value.items()
                if not k.endswith('_ms') and k not in TIME_KEYS and not (path == ('stage',) and k == 'ms')}
    if isinstance(value, (list, tuple)):
        return [stable(v, (*path, i)) for i, v in enumerate(value)]
    return value


def differences(a: Any, b: Any, path: str = '$') -> list[dict]:
    if type(a) is not type(b):
        return [dict(path=path, expected=a, actual=b)]
    if isinstance(a, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result += differences(a.get(key, '<missing>'), b.get(key, '<missing>'), f'{path}.{key}')
        return result
    if isinstance(a, list):
        if len(a) != len(b):
            return [dict(path=path, expected=a, actual=b)]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f'{path}[{i}]')]
    return [] if a == b else [dict(path=path, expected=a, actual=b)]


def memory_guard() -> None:
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].split()[0]) * 1024
    if available < MIN_AVAILABLE:
        raise RuntimeError(f'空きメモリ保護停止: {available / GIB:.3f} GiB')


def save_new(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def forecasts(video: int, game: int, engine: Any, out: Path) -> list[dict]:
    from scripts import prefire_v45_origins as replay
    from scripts import prefire_v46_origins as origins
    from scripts.prefire_v48_inputs import sources as inputs
    source = inputs(video, {game}) if video != 80 else (
        (inp, stream) for inp, stream in replay.sources(video) if inp['game'] == game)
    result = []
    previous = replay.snapshot
    replay.snapshot = origins.snapshot
    try:
        for index, (inp, stream) in enumerate(source):
            memory_guard()
            result.extend(replay.replay(inp, stream, engine.policy, engine.native))
            print('起点再生成', video, game, index + 1, flush=True)
    finally:
        replay.snapshot = previous
    result.sort(key=lambda r: r['input_sec'])
    save_new(out/f'origins_{video}_{game}.json', result)
    return result


def compare_row(expected: dict | None, actual: dict | None, context: dict, out: Path) -> None:
    found = differences(stable(expected), stable(actual))
    if found:
        save_new(out/'first_mismatch.json', dict(**context, differences=found,
            expected=expected, actual=actual, excluded_suffix='_ms', excluded_keys=sorted(TIME_KEYS),
            excluded_paths=['stage.ms'], over_budget_compared=True))
        raise AssertionError(f'出力不一致: {context}; {[d["path"] for d in found]}')


def run_partition(video: int, source: Path, out: Path) -> int:
    from scripts.prefire_v47_data import engine_for
    from scripts.prefire_v44_data import notifications
    engine = engine_for()
    expected = iter(rows(source))
    wanted = {r['game'] for r in rows(source)}
    current, pending, cursor, count = None, [], 0, 0
    target = out/f'predictions_{video}.jsonl'
    try:
        with target.open('x') as stream:
            for stamp, game, raw in notifications(video):
                if game not in wanted:
                    continue
                if game != current:
                    memory_guard()
                    pending = forecasts(video, game, engine, out)
                    current, cursor = game, 0
                delivered = []
                while cursor < len(pending) and pending[cursor]['input_sec'] <= stamp:
                    delivered.append(pending[cursor]); cursor += 1
                actual = engine.observe(stamp, game, raw, delivered)
                if actual is None:
                    continue
                actual['video'] = video
                stream.write(json.dumps(actual, ensure_ascii=False) + '\n')
                stream.flush()
                compare_row(next(expected, None), actual, dict(video=video, source=str(source),
                    row=count + 1, game=game, input_sec=stamp, matched_rows=count), out)
                count += 1
                if count % PROGRESS == 0:
                    memory_guard()
                    print('v4.8一致', video, count, flush=True)
        compare_row(next(expected, None), None, dict(video=video, matched_rows=count), out)
    finally:
        engine.pool.shutdown()
    save_new(out/'receipt.json', dict(rows=count, source=str(source), source_sha256=sha(source),
        output_sha256=sha(target), time_independent_diffs=0))
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import puyo_core
    import src.prefire_post_b3 as post
    post.forecast_features = partial(post.forecast_features, fast_copy=True)
    save_new(args.out/'environment.json', dict(native=str(puyo_core.__file__),
        native_extension=str(puyo_core.puyo_core.__file__), native_sha256=sha(Path(puyo_core.puyo_core.__file__)),
        affinity=sorted(os.sched_getaffinity(0)), fast_copy=True, minimum_wsl_available_gib=7,
        environment={k: v for k, v in os.environ.items() if k.startswith(('PUYO', 'B3', 'OMP', 'OPENBLAS', 'PYTHONPATH'))}))
    total = 0
    for video in VIDEOS:
        for index, source in enumerate(sources(video)):
            destination = args.out/f'v{video}_part{index}'
            destination.mkdir()
            total += run_partition(video, source, destination)
    save_new(args.out/'complete.json', dict(total_rows=total, diffs=0))


if __name__ == '__main__':
    main()
