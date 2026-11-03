"""自worktreeの未起動待機だけを停止し、検証済み制御器の二重計上を解消する。"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
OUT = ROOT/'logs/eval_set/falsedeath_fix/v2d'
WAITERS = {'full_zenchi', 'full_s1', 'full_fcXG83vInDY', 'resources_parallel'}
CONTROLLERS = {'run_falsedeath_v2d_resume.sh', 'run_falsedeath_v2d_shard.sh'}
VERIFIED = ('gfe_phase_stage2b_driver_20261002', 'gfe_phase_stage2b_monitor_20261002')


def select() -> list[dict]:
    """起動口の排他中に、評価Pythonが無いことと待機bashの引数を確認する。"""
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc/'cwd').resolve() != ROOT:
                continue
            args = (proc/'cmdline').read_bytes().decode().split('\0')[:-1]
        except (OSError, UnicodeError):
            continue
        if not args:
            continue
        assert not ('python' in args[0] and 'scripts.falsedeath_replay_20261001' in args), args
        if args[0] != 'bash' or len(args) < 2:
            continue
        name = Path(args[1]).name
        wrapper = name == 'run_falsedeath_v2d.sh' and len(args)>2 and args[2] in WAITERS
        if wrapper:
            assert not (OUT/f'{args[2]}.log').exists(), args
        if wrapper or name in CONTROLLERS:
            rows.append(dict(pid=int(proc.name), args=args))
    assert len([r for r in rows if Path(r['args'][1]).name=='run_falsedeath_v2d.sh']) == len(WAITERS), rows
    return rows


def main() -> None:
    """実計算は停止せず、起動前制御器のみを再起動するための票を保存する。"""
    with (OUT/'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = select()
        sources = {name:Path('/mnt/d/puyo_analyzer/wt_phase/scripts')/(name+'.py') for name in VERIFIED}
        hashes = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}
        for row in rows:
            os.kill(row['pid'], signal.SIGTERM)
        value = dict(reason='子プロセス待機driverと読取monitorだけを除外。実計算上限6・空き6GiB・nice10・BLAS1は維持。',
            stopped_waiters=rows, verified_controller_sha256=hashes)
        with (OUT/'WAIT_RESTART.json').open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1)
        print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
