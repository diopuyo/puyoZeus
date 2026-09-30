"""D4の観測再生をnice 19、一プロセスずつ実行する。"""
from __future__ import annotations
from pathlib import Path
import os
import subprocess
import sys

OUT = Path('logs/d4')
SOURCES = ('q_7gc4TgFig', 'review')


def main() -> None:
    """既存タスクのプロセス・出力へ触れず、完了原票で再開する。"""
    for source in SOURCES:
        for mode in ('off', 'on'):
            dest = OUT/'replay'/mode/source
            if (dest/'EQUIVALENCE.json').exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            with (dest/'runner.log').open('w') as stream:
                subprocess.run([sys.executable, '-B', '-m', 'scripts._d4_replay',
                    '--source', source, '--mode', mode], stdout=stream,
                    stderr=subprocess.STDOUT, check=True)
    (OUT/'REPLAY_DONE.json').write_text('{"completed": true}')


if __name__ == '__main__':
    main()
