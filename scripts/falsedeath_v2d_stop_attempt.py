"""H1の開始証拠修正前の、自作業の候補再生だけを停止する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
OUT = ROOT/'logs/eval_set/falsedeath_fix/v2d'
NAMES = {'run_falsedeath_v2d_resume.sh','run_falsedeath_v2d_shard_resume.sh',
         'run_falsedeath_v2d_wait_finish.sh'}


def main() -> None:
    """cwd・起動スクリプト・独立セッションを照合してから自分のグループだけ停止する。"""
    selected = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().decode().split('\0')[:-1]
            if (proc/'cwd').resolve() != ROOT or len(args)<2 or args[0]!='bash':
                continue
            pid = int(proc.name)
            if Path(args[1]).name in NAMES:
                assert os.getsid(pid) == os.getpgid(pid) == pid, (pid,args)
                selected.append(dict(pid=pid,args=args))
        except OSError:
            continue
    assert selected
    with (OUT/'STOP_IMPLEMENTATION_CHECK.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(reason='開始直後の全消しと試合開始を得点ゼロで区別する単体不備を修正。全体未採点。',
                       stopped=selected), stream, ensure_ascii=False, indent=1)
    for row in selected:
        os.killpg(row['pid'],signal.SIGTERM)
    print(json.dumps(selected))


if __name__ == '__main__':
    main()
