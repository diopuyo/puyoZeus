"""自worktreeの未起動待機処理だけを、起動排他内で停止する。"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
OUT = ROOT/'logs/eval_set/falsedeath_fix/v2c'


def main() -> None:
    """既存計算の停止を拒否し、照合済みのbash待機だけを停止する。"""
    with (OUT/'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not (OUT/'short_review.log').exists()
        assert not (OUT/'final_unit.log').exists()
        selected = []
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit():
                continue
            try:
                args = (directory/'cmdline').read_bytes().decode().split('\0')[:-1]
                cwd = (directory/'cwd').resolve()
            except (OSError, UnicodeError):
                continue
            if cwd != ROOT or not args or args[0] != 'bash':
                continue
            wrapper = (len(args)>2 and args[1]=='scripts/run_falsedeath_v2c.sh'
                       and args[2] in ('short_review','final_unit'))
            ladder = args[1:] == ['scripts/run_falsedeath_v2c_ladder.sh']
            if wrapper or ladder:
                selected.append(dict(pid=int(directory.name), args=args))
        assert len(selected) == 3, selected
        for row in selected:
            os.kill(row['pid'], signal.SIGTERM)
        with (OUT/'WAIT_RESTART.json').open('x') as stream:
            json.dump(dict(reason='検証済み制御プロセスの二重計上を除く。実計算上限は6を維持。',
                           stopped_waiters=selected),stream,ensure_ascii=False,indent=1)
        print(json.dumps(selected))


if __name__ == '__main__':
    main()
