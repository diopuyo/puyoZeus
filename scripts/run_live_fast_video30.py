"""予告 CNN の試作 (b3_warning_batch + b3_warning_prefetch) を video_30 で流し直す入口 (合言葉の門つき)。

scripts/run_live_x_video30.py の設定・差し替え・受領票をそのまま使い、違いは 3 点だけ:
- ON の腕は X のフラグを入れず、b3_warning_batch / b3_warning_prefetch だけを入れる (OFF は 21c と同じ)。
- PC1 の M4 が走っている間に短い測定を回すため、空きメモリを 2 秒ごとに見て、線を割ったら
  子の process 木を止めて受領票に残す (止めた腕は測定に使わない)。
- 出力先の既定は <work>/live_fast_20261023/replay。

plan: 何も作らず計画を表示する。run: --token と確認語の両方がそろったときだけ走る。
低い優先度で回すときは、この入口自体を BelowNormal で起動する (子は優先度を継承する)。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import live_x_video30_spec as spec  # noqa: E402
from scripts import run_live_x_replay as replay  # noqa: E402
from scripts import run_live_x_video30 as v30  # noqa: E402

DEFAULT_ROOT = Path(_public_dev_text('<work>/live_fast_20261023/replay'))
FAST_FLAGS = ('b3_warning_batch', 'b3_warning_prefetch')
RUN_TOKEN = 'RUN-LIVE-FAST'
CONFIRM_ENV = 'LIVE_FAST_CONFIRM'
CONFIRM_WORD = 'PC1-LOWPRIO-MEMGATE'
STOP_FREE_GIB = 8.0        # 親の指示: 空きが 8GiB を割りそうなら止める
POLL_SEC = 2.0
GIB = 1024**3


def user_config(arm: str, out: Path, first_sec: float, end_sec: float,
                saved: Path = replay.SAVED) -> dict[str, Any]:
    """21c の設定 (X は全部 OFF) に、ON の腕だけ予告 CNN の試作フラグを足す。"""
    values = v30.user_config('OFF', out, first_sec, end_sec, (), saved)
    for key in FAST_FLAGS:
        values[key] = arm == 'ON'
    return values


def plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    v30.check_range(args.first_sec, args.end_sec)
    modules = replay.overlay_modules()
    drift = replay.bundle_drift(modules)
    if drift:
        raise SystemExit(f'配布 app と起点 {replay.BASE_COMMIT} のソースが違う: {drift[:10]}')
    runs = []
    for order, arm in enumerate(args.arms, 1):
        out = args.root/f'{args.condition}{order}_{arm}'
        runs.append(dict(name=out.name, arm=arm, out=str(out),
                         config=user_config(arm, out, args.first_sec, args.end_sec),
                         child=v30.child_plan(args.condition, out, modules),
                         command=[str(replay.RUNTIME/'python.exe'), '-B', '-u', str(replay.CHILD), '--config',
                                  str(out/'puyo_live.json'), '--skip-manifest', '--dev-allow-video']))
    return runs


def free_gib() -> float:
    from src.phase_j.live_b3_time_gate import MemoryStatus
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    return memory.available/GIB


def watched(command: list[str], cwd: Path, env: dict, log: Any) -> tuple[int, dict]:
    """子を走らせ、空きメモリの最小値を記録する。線を割ったら process 木ごと止める。"""
    child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    lowest, stopped, started = free_gib(), False, time.monotonic()
    while child.poll() is None:
        time.sleep(POLL_SEC)
        lowest = min(lowest, free_gib())
        if lowest < STOP_FREE_GIB and not stopped:
            stopped = True
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(child.pid)], capture_output=True, check=False)
    return child.wait(), dict(min_free_gib=round(lowest, 2), stopped_by_memory=stopped,
                              wall_sec=round(time.monotonic()-started, 1))


def run_one(run: dict) -> dict:
    """replay.run_one と同じ受領票に、空きメモリの記録を足す。"""
    out = Path(run['out'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'temp').mkdir()
    (out/'puyo_live.json').write_text(json.dumps(run['config'], ensure_ascii=False, indent=2), encoding='utf-8')
    environment = dict(os.environ, **replay.OBS_ENV, TEMP=str(out/'temp'), TMP=str(out/'temp'),
                       OPENCV_FFMPEG_DLL_DIR=str(replay.DLL), PUYO_CALIBRATION_DIR=str(out/'calibration'),
                       LIVE_X_REPLAY_CHILD=json.dumps(run['child']))
    with (out/'run.log').open('xb') as log:
        code, memory = watched(run['command'], replay.BUNDLE/'app', environment, log)
    patches = [json.loads(line) for line in (out/'patches.jsonl').read_text(encoding='utf-8').splitlines()] \
        if (out/'patches.jsonl').exists() else []
    recognition = out/'output/recognition_config.json'
    reference = json.loads((replay.SAVED/'recognition_config.json').read_text(encoding='utf-8'))
    pipeline = out/'output/pipeline_config.json'
    flags = json.loads(pipeline.read_text(encoding='utf-8')) if pipeline.exists() else {}
    return dict(name=run['name'], arm=run['arm'], exit_code=code, overlay_modules=run['child']['source_modules'],
                patches={name: len({p['pid'] for p in patches if p['patch'] == name})
                         for name in ('video_source', 'recognition_config', 'launcher')},
                recognition_matches_21c=recognition.exists()
                and json.loads(recognition.read_text(encoding='utf-8')) == reference,
                fast_flags={key: flags.get(key, False) for key in FAST_FLAGS}, memory=memory)


def execute(args: argparse.Namespace) -> None:
    if args.token != RUN_TOKEN or os.environ.get(CONFIRM_ENV) != CONFIRM_WORD:
        raise SystemExit(f'起動しない: --token {RUN_TOKEN} と 環境変数 {CONFIRM_ENV}={CONFIRM_WORD} の両方が必要')
    if (reason := replay.blocked()) is not None:
        raise SystemExit(f'起動しない: {reason}')
    if replay.digest(spec.VIDEO) != spec.VIDEO_SHA256:
        raise SystemExit('video_30 の SHA256 が固定値と違う')
    runs = plan(args)
    args.root.mkdir(parents=True, exist_ok=True)
    for run in runs:
        receipt = dict(run_one(run), **v30.output_check(Path(run['out']), args.first_sec, args.end_sec))
        (Path(run['out'])/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                                                    encoding='utf-8')
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        failure = v30.accept(receipt, args.condition)
        if receipt['memory']['stopped_by_memory']:
            failure = '空きメモリの線で停止'
        if receipt['fast_flags'] != {key: run['arm'] == 'ON' for key in FAST_FLAGS}:
            failure = f"試作フラグが届いていない: {receipt['fast_flags']}"
        if failure is not None:
            raise SystemExit(f"{run['name']} が不合格 ({failure}): 原票を残して止める")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--condition', choices=tuple(replay.CONDITIONS), default='L')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--arms', type=lambda s: tuple(s.split(',')), default=('OFF', 'ON'))
    parser.add_argument('--first-sec', type=float, default=spec.FIRST_PROCESSED_SEC)
    parser.add_argument('--end-sec', type=float, default=spec.END_SEC)
    parser.add_argument('--token')
    args = parser.parse_args(argv)
    if any(arm not in ('OFF', 'ON') for arm in args.arms):
        parser.error('--arms は OFF/ON のカンマ区切り')
    return args


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    if args.command == 'plan':
        runs = plan(args)
        print(json.dumps(dict(condition=args.condition, workload=spec.workload(args.first_sec, args.end_sec),
                              runs=[dict(name=r['name'], arm=r['arm'], out=r['out'],
                                         fast={k: r['config'][k] for k in FAST_FLAGS},
                                         overlay=r['child']['source_modules']) for r in runs]),
                         ensure_ascii=False, indent=2))
        return
    execute(args)


if __name__ == '__main__':
    main()
