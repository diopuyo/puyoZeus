"""b版の保存原票同値を再実行し、各段のログを逐次保存する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('D:/puyo_analyzer/packaging/build_20261020b_work')
PACKETS = Path('D:/puyo_analyzer/packaging/build_20261019a/packets')
OLD = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive'))
M3_RECORD = Path(_public_dev_text('<work>/build_20261020a_stage/m3_input'))
M3_REFERENCE = Path(_public_dev_text('<work>/build_20261020a_stage/m3_offline_r3.jsonl'))
SELECTED: set[str] | None = None
START_FREE_GIB = 8.5
MEMORY_POLL_SEC = 1.
HEARTBEAT_SEC = 60.


def guarded_run(command: list[str], environment: dict, log: object) -> int:
    """検収の子だけを停止できるよう、空きメモリとC:空きを継続監視する。"""
    from smoke_live_obs_20261020 import memory_guard
    memory_guard(START_FREE_GIB)
    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    began = time.monotonic()
    heartbeat = began+HEARTBEAT_SEC
    try:
        while True:
            try:
                return child.wait(timeout=MEMORY_POLL_SEC)
            except subprocess.TimeoutExpired:
                memory_guard()
                now = time.monotonic()
                if now >= heartbeat:
                    message = f'[verification heartbeat] pid={child.pid} elapsed={now-began:.0f}s\n'
                    log.write(message)
                    log.flush()
                    print(message, end='', flush=True)
                    heartbeat = now+HEARTBEAT_SEC
    except BaseException:
        if child.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            else:
                child.terminate()
            child.wait(timeout=30)
        raise


def run(name: str, script: str, args: list[str], cadence: bool = True, m3: bool = False) -> None:
    if SELECTED is not None and name not in SELECTED:
        return
    environment = dict(os.environ, PUYO_LIVE_SCORE_CADENCE=str(int(cadence)), PUYO_B3_M3=str(int(m3)),
                       PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    command = [sys.executable, str(ROOT/'packaging'/script), *map(str, args)]
    with (OUT/f'{name}.log').open('x', encoding='utf-8') as log:
        result = guarded_run(command, environment, log)
    print(name, result, flush=True)
    if result:
        raise RuntimeError(name)


def main() -> None:
    run('off', 'verify_live_v49f_off.py', ['--root', ROOT, '--packets', PACKETS/'80_7.jsonl.gz',
        '--out', OUT/'off.jsonl'], cadence=False)
    run('ama_inputs', 'verify_live_ama_inputs.py', [OUT/'ama_inputs'])
    run('m3_live', 'verify_live_m3.py', ['--root', ROOT,
        '--record', M3_RECORD, '--model', ROOT/'models/b3_m3/model.joblib',
        '--table', OLD/'app/models/b3/timing_table.json', '--out', OUT/'m3_live.jsonl',
        '--reference', M3_REFERENCE], m3=True)
    for video, game in ((80, 7), (51, 4)):
        run(f'v49f_{video}_{game}', 'verify_live_v49f_windows.py', ['--app', ROOT, '--assets', OLD/'app/models/b3',
            '--policy', ROOT/'models/v49f/policy.json', '--pre-model', ROOT/'models/v49f/pre_model.json',
            '--packets', PACKETS, '--out', OUT/f'v49f_{video}_{game}', '--video', video, '--game', game], m3=True)
    run('ama_values', 'verify_live_ama_windows.py', ['--binary', ROOT/'models/ama_plain/ama_reference.exe',
        '--out', OUT/'ama_values'])


if __name__ == '__main__':
    main()
