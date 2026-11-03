"""同じ240秒区間でama停止・共有CPU・I/O分離・CPU分離を逐次比較する。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('D:/puyo_analyzer/packaging/build_20261020d_work')
BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261020d_stage_r3/PuyoLive')
SOURCE_OVERLAY: Path | None = None
SOURCE_PLAN: Path | None = None
STARTUP_TIMEOUT = 240
DECODER = Path('D:/puyo_analyzer/packaging/dev_ffmpeg')
sys.path.insert(0, str(ROOT/'packaging'))
from analyze_live_obs_20261020d import analyze

CASES = dict(off=['--ama-off'], fast_off=['--ama-off', '--fast-audit', '--exact-seed'], shared=[], audit=['--fast-audit'],
             isolated=['--fast-audit', '--cpu-isolation'], priority=['--fast-audit', '--ama-idle'],
             seed=['--fast-audit', '--exact-seed'],
             seed_isolated=['--fast-audit', '--exact-seed', '--cpu-isolation'],
             seed_priority=['--fast-audit', '--exact-seed', '--ama-idle'])


def run(name: str, flags: list[str], formal: bool, port: int, duration: int = 240) -> None:
    target = WORK/f'{"speed" if formal else "diag"}_{name}'
    command = [sys.executable, str(ROOT/'packaging/smoke_live_obs_20261020.py'),
        '--bundle', str(BUNDLE), '--out', str(target), '--duration', str(duration),
        '--video', 'C:/puyo1_tools/soak/media/video_d04.mp4', '--start-sec', '600',
        '--ffmpeg-dll-dir', str(DECODER), '--port', str(port), *flags]
    command += ['--startup-timeout', str(STARTUP_TIMEOUT)]
    if SOURCE_OVERLAY is not None:
        command += ['--source-overlay', str(SOURCE_OVERLAY), '--source-plan', str(SOURCE_PLAN)]
    if formal:
        command += ['--benchmark', '--wait-started-at', (WORK/'wait_started_at.txt').read_text().strip()]
    print('START', name, formal, flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    receipt = json.loads((target/'receipt.json').read_text())
    if completed.returncode and receipt.get('exit_code') != 0:
        raise RuntimeError(f'{name}: 子プロセス異常終了、測定不採用')
    value = analyze(target)
    (target/'analysis.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {k: v for k, v in value.items() if k != 'outliers'}
    summary['all_outliers_file'] = str(target/'analysis.json')
    summary['smoke_passed'] = completed.returncode == 0
    (target/'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', name, value['total_low_spec_ms'], value['ama'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--duration', type=int, default=240)
    parser.add_argument('--source-overlay', type=Path)
    parser.add_argument('--source-plan', type=Path)
    parser.add_argument('--startup-timeout', type=int, default=STARTUP_TIMEOUT)
    parser.add_argument('--ffmpeg-dll-dir', type=Path, default=DECODER)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=['off', 'audit', 'isolated'])
    args = parser.parse_args()
    BUNDLE, WORK = args.bundle, args.work
    SOURCE_OVERLAY, SOURCE_PLAN, STARTUP_TIMEOUT = args.source_overlay, args.source_plan, args.startup_timeout
    DECODER = args.ffmpeg_dll_dir
    for index, name in enumerate(args.cases):
        run(name, CASES[name], args.formal, 18825+index, args.duration)
