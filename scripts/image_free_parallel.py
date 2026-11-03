"""固定済み比較コードを変更せず、単一監視器で動画別に実行する。"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
POLL_SECONDS = 5
QUERY_TIMEOUT = 15
FAILURE_LIMIT = 3
RESERVE_BYTES = 6 * GIB
START_BYTES = 2 * GIB
RECOVER_BYTES = GIB // 2
MAX_WORKERS = 4
PR_SET_PDEATHSIG = 1
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                  OPENCV_FFMPEG_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
QUERY = ("$ErrorActionPreference='Stop'; $p=@(Get-CimInstance Win32_Process | "
         "Where-Object { $_.Name -match '^(PuyoLive|PuyoZeus)([-_].*)?\\.exe$' -or "
         "($_.Name -match '^python(w)?\\.exe$' -and "
         "$_.CommandLine -match 'src\\.phase_j\\.(launcher|distribution_app)') } | "
         "Select-Object -ExpandProperty ProcessId); "
         "$m=Get-CimInstance Win32_OperatingSystem; "
         "@{pids=$p;free_bytes=([long]$m.FreePhysicalMemory*1024)} | ConvertTo-Json -Compress")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def linux_available() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('WSL空きメモリを取得できません')


@dataclass
class Monitor:
    failures: int = 0
    paused: bool = True
    windows_free: int = 0
    wsl_free: int = 0
    reason: str = '初回監視待ち'

    def poll(self) -> bool:
        try:
            result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', QUERY],
                check=True, capture_output=True, text=True, timeout=QUERY_TIMEOUT)
            value = json.loads(result.stdout)
            self.windows_free = int(value['free_bytes'])
            self.wsl_free = linux_available()
            minimum = min(self.windows_free, self.wsl_free)
            threshold = RESERVE_BYTES + (RECOVER_BYTES if self.paused else 0)
            self.paused = bool(value['pids']) or minimum < threshold
            self.reason = f"PuyoLive={value['pids']}, 空き最小={minimum/GIB:.2f}GiB"
            self.failures = 0
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as error:
            self.failures += 1
            self.paused = self.paused or self.failures >= FAILURE_LIMIT
            self.reason = f'監視再試行 {self.failures}/{FAILURE_LIMIT}: {error}'
        return self.paused

    def can_start(self) -> bool:
        return (not self.paused and self.failures == 0
                and min(self.windows_free, self.wsl_free) >= RESERVE_BYTES + START_BYTES)


def signal_worker(child: subprocess.Popen, sig: int) -> None:
    if child.poll() is None:
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass


def stop_worker(child: subprocess.Popen) -> None:
    signal_worker(child, signal.SIGTERM)
    signal_worker(child, signal.SIGCONT)
    try:
        child.wait(timeout=POLL_SECONDS)
    except subprocess.TimeoutExpired:
        signal_worker(child, signal.SIGKILL)
        child.wait()


def worker_cpus() -> list[int]:
    """SMTの兄弟を避け、物理コアを一巡してから追加する。"""
    available = sorted(os.sched_getaffinity(0))
    first, extra, seen = [], [], set()
    for cpu in available:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((topology/'physical_package_id').read_text(), (topology/'core_id').read_text())
        if key in seen:
            extra.append(cpu)
        else:
            first.append(cpu)
            seen.add(key)
    return first + extra


def launch(args: argparse.Namespace, video_id: str, slot: int) -> subprocess.Popen:
    cpus = worker_cpus()
    if args.config.exists() and (read_json(args.config).get('segmented')
                                or read_json(args.config).get('split_streams')):
        from scripts.image_free_resources import unoccupied_cpus
        cpus = unoccupied_cpus(cpus)
        slot = 0
    command = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config),
               '--source', str(args.source), '--worker', video_id, '--parent', str(os.getpid()),
               '--cpu', str(cpus[slot % len(cpus)])]
    directory = args.output / video_id
    directory.mkdir(exist_ok=True)
    with (directory / 'parallel.log').open('a') as log:
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, **THREAD_ENV))


def disabled_videos(output: Path) -> set[str]:
    control = output / 'parallel_control.json'
    return set(read_json(control).get('disabled_videos', [])) if control.exists() else set()


def collect_results(output: Path, ids: list[str], codes: dict) -> None:
    reports = {key: read_json(output/key/'progress.json') for key in ids
               if (output/key/'progress.json').exists()}
    complete = len(reports) == len(ids) and all(r['complete'] for r in reports.values())
    if any(r.get('aggregate_pending') for r in reports.values()):
        save_json(output/'result.json', dict(routes=reports, routes_complete=complete,
            complete=False, comparison_mode='continuous_routes', aggregate_pending=True,
            gate_passed=False, exit_codes=codes))
        return
    if any(r.get('canonical_aggregation_pending') for r in reports.values()):
        paired = all(r.get('comparison_mode') == 'paired_cold' for r in reports.values())
        save_json(output/'result.json', dict(segments=reports, segments_complete=complete,
            complete=False, canonical_aggregation_pending=True, gate_passed=False, exit_codes=codes,
            comparison_mode='paired_cold' if paired else 'continuous', continuous_equivalence=False,
            disclaimer='通しとの同等性は不成立・対の比較' if paired else None))
        return
    save_json(output/'result.json', dict(videos=reports, complete=complete,
        display_changed_scenes=sum(r['display_changed_scenes'] for r in reports.values()),
        display_text_changed_scenes=sum(r['display_text_changed_scenes'] for r in reports.values()),
        gate_passed=False, verdict='統括による合格判定待ち', exit_codes=codes))


def resource_control(monitor: Monitor, children: dict, paused: bool,
                     reserve_bytes: int = RESERVE_BYTES) -> None:
    for child in children.values():
        signal_worker(child, signal.SIGSTOP if paused else signal.SIGCONT)
    # 停止だけではメモリを返せないため、低メモリ時は一つずつ確定点へ戻す。
    if children and monitor.failures == 0 and min(monitor.windows_free, monitor.wsl_free) < reserve_bytes:
        key = next(reversed(children))
        stop_worker(children.pop(key))


def run_loop(args: argparse.Namespace, ids: list[str], children: dict) -> int:
    from scripts.image_free_resources import ResourceBudget
    config = read_json(args.config)
    reserve = RESERVE_BYTES + (GIB if config.get('segmented')
                               and config.get('comparison_mode') != 'paired_cold' else 0)
    budget = ResourceBudget(reserve_bytes=reserve)
    monitor, codes = Monitor(), {}
    while True:
        paused = monitor.poll()
        paused |= monitor.failures == 0 and min(monitor.windows_free, monitor.wsl_free) < reserve
        budget.observe(children, args.output)
        disabled = disabled_videos(args.output)
        for key, child in list(children.items()):
            if key in disabled:
                stop_worker(child)
                del children[key]
            elif child.poll() is not None:
                codes[key] = child.returncode
                del children[key]
        resource_control(monitor, children, paused, reserve)
        maximum = (min(args.workers, budget.limit(monitor, len(children), len(worker_cpus())))
                   if args.auto_workers else args.workers)
        pending = [key for key in ids if key not in children and key not in codes and key not in disabled]
        can_start = budget.can_start(monitor) if args.auto_workers else monitor.can_start()
        if pending and len(children) < maximum and can_start:
            children[pending[0]] = launch(args, pending[0], ids.index(pending[0]))
        finished = not children and not pending
        status = 'failed' if any(codes.values()) else 'finished'
        save_json(args.output/'supervisor.json', dict(supervisor_pid=os.getpid(),
            worker_pids={k: v.pid for k, v in children.items()}, exit_codes=codes,
            disabled_videos=sorted(disabled), paused=paused, reason=monitor.reason,
            windows_free_bytes=monitor.windows_free, wsl_free_bytes=monitor.wsl_free,
            consecutive_failures=monitor.failures, max_workers=args.workers,
            effective_workers=maximum, measured_peak_rss=budget.peak_rss,
            rss_qualified=budget.qualified,
            status=status if finished else ('paused' if paused else 'running'), updated_at=time.time()))
        if finished:
            collect_results(args.output, ids, codes)
            return int(any(codes.values()))
        time.sleep(POLL_SECONDS)


def supervise(args: argparse.Namespace) -> int:
    import fcntl
    children: dict[str, subprocess.Popen] = {}
    ids = [v['id'] for v in read_json(args.config)['videos']]
    if len(ids) != len(set(ids)):
        raise ValueError('動画IDが重複しています')
    args.workers = min(args.workers, len(os.sched_getaffinity(0)), len(ids))
    if args.workers < 1:
        raise ValueError('並列数は1以上必要です')
    with (args.output/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run_loop(args, ids, children)
        finally:
            for child in children.values():
                stop_worker(child)


class CentralGuard:
    """監視は親だけが担当し、親消失時は作業を中止する。"""

    def __init__(self, parent: int) -> None:
        self.parent = parent

    def checkpoint(self, force: bool = False) -> None:
        if os.getppid() != self.parent:
            raise RuntimeError('監視器が終了したため作業を中止します')


def run_worker(args: argparse.Namespace) -> None:
    import fcntl
    # 親が異常終了しても孤児が実行を継続しない。設定後の競合も確認する。
    if ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError('親終了時シグナルを設定できません')
    guard = CentralGuard(args.parent)
    guard.checkpoint()
    os.sched_setaffinity(0, {args.cpu})
    sys.path.insert(0, str(args.source))
    from scripts import compare_image_free_all as compare
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    config = compare.read_config(args.config)
    output = Path(config['output'])
    video = next(v for v in config['videos'] if v['id'] == args.worker)
    os.chdir(config['asset_root'])
    os.environ.update(PUYO_IMAGE_FREE_UI_MODEL=str(output/'ui.joblib'),
        PUYO_IMAGE_FREE_DIGIT_MODEL=str(output/'digit_student.joblib'),
        PUYO_IMAGE_FREE_UI_MASK_MODEL=config['mask_model'], PUYO_B3_JUDGMENT='1')
    with (output/args.worker/'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if config.get('split_streams'):
            if video['route'] == 'both':
                from scripts.image_free_stream_proof import baseline
                baseline(video, config, guard)
            else:
                from scripts.image_free_stream import run
                run(video, config, guard)
        elif config.get('segmented'):
            from scripts.image_free_segments import run_segment
            run_segment(video, config, guard)
        else:
            compare.run_video(video, config, guard, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--auto-workers', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--parent', type=int)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    args.config, args.source = args.config.resolve(), args.source.resolve()
    sys.path.insert(0, str(args.source))
    raw = read_json(args.config)['output'].replace('\\', '/')
    args.output = Path('/mnt/'+raw[0].lower()+raw[2:] if raw[1:2] == ':' else raw)
    if args.worker:
        run_worker(args)
    else:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        raise SystemExit(supervise(args))


if __name__ == '__main__':
    main()
