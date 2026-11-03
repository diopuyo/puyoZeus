"""既存55分台を再利用し、低優先度・メモリ下限・所有ツリー終了を追加する。"""
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
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = Path(_public_dev_text('<work>/delta'))
SOAK = Path('C:/puyo1_tools/soak')
sys.path[:0] = [str(REPO), str(SOAK)]
from scripts.bench_live_delta import check_resources, system_times
import soak_dshow as base

ORIGINAL_POPEN = subprocess.Popen
ABORT_REASON: str | None = None


class FeedProcess(ORIGINAL_POPEN):
    """venvリダイレクタの子も回収し、次試行へのカメラ持越しを防ぐ。"""

    def terminate(self) -> None:
        subprocess.run(['taskkill', '/PID', str(self.pid), '/T', '/F'],
                       capture_output=True, check=False)
        self.wait(timeout=30)


def popen(command: Any, *args: Any, **kwargs: Any) -> Any:
    kwargs['creationflags'] = kwargs.get('creationflags', 0) | 0x4000
    process = FeedProcess if any('vcam_feed.py' in str(x) for x in command) else ORIGINAL_POPEN
    return process(command, *args, **kwargs)


def watch(process: Any, url: str, bundle: Path, out: Path, seconds: float, started: float) -> None:
    """既存poll/sample形式を維持し、全体負荷と停止理由を追記する。"""
    global ABORT_REASON
    next_sample, previous = 0.0, system_times()
    with (out / 'polls.jsonl').open('a') as polls, (out / 'samples.jsonl').open('a') as samples:
        while process.poll() is None and time.perf_counter() - started < seconds:
            try:
                check_resources()
            except RuntimeError as error:
                ABORT_REASON = str(error)
                break
            now = time.perf_counter() - started
            snapshot = base.fetch(url)
            polls.write(json.dumps(dict(t=now, **(base.summarize(snapshot) if snapshot else {'http': False}))) + '\n')
            polls.flush()
            if now >= next_sample:
                current = system_times()
                cpu = 100 * (1 - (current[0] - previous[0]) / max(1, current[1] - previous[1]))
                samples.write(json.dumps(dict(t=now, system_cpu_percent=cpu, **base.sample(str(bundle)))) + '\n')
                samples.flush()
                previous, next_sample = current, now + base.SAMPLE_EVERY_SEC
            time.sleep(base.POLL_SEC)


def run(mode: str, seconds: int, tag: str) -> None:
    check_resources()
    run_dir = OUT / ('run_' + mode + tag)
    run_dir.mkdir(exist_ok=True)
    bundle = OUT / ('soak_' + mode) / 'PuyoLive'
    settings = json.loads((OUT / ('run_' + mode) / 'puyo_live.json').read_text())
    settings['output_dir'] = str(run_dir / 'output')
    config = run_dir / 'puyo_live.json'
    config.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    (OUT / 'soak_progress.json').write_text(json.dumps(dict(mode=mode, state='running', tag=tag)))
    sys.argv = ['soak_dshow', str(bundle), str(config), str(run_dir), str(seconds)]
    base.main()
    status = json.loads((run_dir / 'soak_status.json').read_text())
    if ABORT_REASON or status.get('exit_before_stop') is not None or status.get('forced'):
        raise RuntimeError(ABORT_REASON or f'試験異常終了: {status}')
    subprocess.run([sys.executable, str(SOAK / 'analyze_soak.py'), str(run_dir)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=3300)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    subprocess.Popen, base.watch = popen, watch
    # この起動環境ではCTRL_C後に子が残ったため、両条件ともCTRL_BREAKで終了する。
    # 終了時の未完了gzipを品質合格には用いず、計測済みのspoolだけを集計する。
    base.CTRL_C = base.CTRL_C.replace('GenerateConsoleCtrlEvent(0,0)', 'GenerateConsoleCtrlEvent(1,0)')
    for mode in ('off', 'on'):
        run(mode, args.seconds, args.tag)
    (OUT / 'soak_progress.json').write_text(json.dumps(dict(state='finished', tag=args.tag)))


if __name__ == '__main__':
    main()
