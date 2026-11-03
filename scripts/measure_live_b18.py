"""B18: 現行本番構成の実時間遅延計測(逐次更新のみ)と取得待ち増加の分解計装。
起動は scripts/launch_live_b18.sh 経由のみ (--confirm-run 必須)。src/は変更しない。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

from scripts._diag_b18_probe import install

install()  # 全spawn子processでも __mp_main__ として実行される

START = 5880.566
SAMPLE_SEC = 1.0
MODE = 'candidate'


def proc_sample(pid: int) -> dict:
    """/proc/PIDから1process分。CPU ticks・RSS・IO・切替・スレッド数。"""
    base = Path('/proc')/str(pid)
    stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
    status = dict(l.split(':', 1) for l in (base/'status').read_text().splitlines())
    io = dict(l.split(':') for l in (base/'io').read_text().splitlines())
    return dict(pid=pid, ppid=int(stat[1]), utime=int(stat[11]), stime=int(stat[12]),
        threads=int(stat[17]), rss=int(status['VmRSS'].split()[0])*1024,
        vol=int(status['voluntary_ctxt_switches']), invol=int(status['nonvoluntary_ctxt_switches']),
        rchar=int(io['rchar']), wchar=int(io['wchar']),
        read_bytes=int(io['read_bytes']), write_bytes=int(io['write_bytes']))


def files_size(root: Path) -> dict:
    total, journal, spool = 0, 0, 0
    for path in root.rglob('*'):
        try:
            if path.is_file():
                size = path.stat().st_size
                total += size
                if 'journal' in path.name:
                    journal += size
                if 'spool' in path.parts:
                    spool += size
        except OSError:
            continue
    return dict(total=total, journal=journal, spool=spool)


def sampler(group: int, output: Path, stop: threading.Event) -> None:
    from scripts.live_b9_resources import group_members
    with (output/'b18_proc.jsonl').open('w', buffering=1) as stream:
        while not stop.is_set():
            rows = []
            for member in group_members(group):
                try:
                    rows.append(proc_sample(member['pid']))
                except (OSError, ValueError, IndexError):
                    continue
            cpu = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
            stream.write(json.dumps(dict(at=time.perf_counter(), procs=rows, cpu_ticks=cpu,
                sizes=files_size(output), loadavg=os.getloadavg()))+'\n')
            stop.wait(SAMPLE_SEC)


def execute(root: Path, duration: float, start: float, fix_frozen: bool = False,
            flags: tuple[str, ...] = ()) -> None:
    from scripts.measure_live_b12 import command, monitor
    from scripts.measure_live_b9 import owned_process, cleanup_audit
    from scripts.measure_live_b6 import save, wait_idle, MAX_WAIT_SECONDS
    if not wait_idle(time.monotonic()+MAX_WAIT_SECONDS, root, 'B18'):
        raise SystemExit('低負荷待ち上限')
    args = command(MODE, root, duration, 'c')
    config_path = Path(args[args.index('--config')+1])
    config = json.loads(config_path.read_text())
    # B20: 追加の既定OFFフラグ(fast_terminal / async_notice_queue)を設定へ足す
    save(config_path, dict(config, start_sec=start, end_sec=start+duration, **{f: True for f in flags}))
    args[2] = 'scripts.measure_live_b18'
    args[3:3] = ['--worker', MODE]
    path = root/MODE
    probe = root/'probe'
    os.environ['PUYO_B18_PROBE'] = str(probe)
    if fix_frozen:
        os.environ['PUYO_B18_FIX_FROZEN'] = '1'
    if os.environ.get('PUYO_B18_STACKS') == '1':
        pass  # 子processへ環境変数がそのまま継承される
    stop = threading.Event()
    save(root/'status.json', dict(state='running', pid=os.getpid(), duration=duration))
    with (root/(MODE+'.log')).open('w') as log, owned_process(args, log) as child:
        deadline = time.monotonic()+120
        while not path.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        save(path/'launch.json', dict(command=args, pid=child.pid, start=start, end=start+duration,
                                      config=config, host_cores=os.cpu_count()))
        thread = threading.Thread(target=sampler, args=(child.pid, path, stop), daemon=True)
        thread.start()
        try:
            monitor(child, path, duration)
            child.wait()
        finally:
            stop.set()
            thread.join(5)
        if child.returncode:
            raise RuntimeError(f'exit={child.returncode}')
    cleanup = cleanup_audit(child.pid)
    save(root/'status.json', dict(state='finished', pid=os.getpid(), cleanup=cleanup))


def main() -> None:
    if '--worker' in sys.argv:
        index = sys.argv.index('--worker')
        mode = sys.argv[index+1]
        del sys.argv[index:index+2]
        from scripts.measure_live_b17 import worker
        worker(mode)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--start', type=float, default=START)
    parser.add_argument('--fix-frozen', action='store_true', help='検証専用: FrozenInstanceErrorを実行時のみ回避')
    parser.add_argument('--no-terminal', action='store_true', help='検証専用what-if: 死亡検出matchTemplateを空実装')
    parser.add_argument('--flags', default='', help='B20: カンマ区切りで有効化する設定フラグ')
    parser.add_argument('--confirm-run', action='store_true')
    options = parser.parse_args()
    if not options.confirm_run:
        raise SystemExit('--confirm-run が必要です(誤起動防止)')
    options.output.mkdir(parents=True, exist_ok=False)
    if options.no_terminal:
        os.environ['PUYO_B18_NO_TERMINAL'] = '1'
    flags = tuple(f for f in options.flags.split(',') if f)
    execute(options.output, options.duration, options.start, options.fix_frozen, flags)


if __name__ == '__main__':
    main()
