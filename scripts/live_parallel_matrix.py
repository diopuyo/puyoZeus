"""並列候補を順方向・逆方向に直列実行し、負荷と結果を保存する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/live_parallel'
CANDIDATES = [('base', []), ('intra2', ['--intra', '2']),
    ('intra3', ['--intra', '3']), ('intra4', ['--intra', '4']),
    ('inter2', ['--inter', '2', '--parallel']),
    ('inter3', ['--inter', '3', '--parallel']),
    ('inter4', ['--inter', '4', '--parallel']),
    ('both2', ['--intra', '2', '--inter', '2', '--parallel']),
    ('cv2', ['--cv', '2']), ('cv3', ['--cv', '3']), ('cv4', ['--cv', '4']),
    ('sides', ['--sides'])]
FOCUSED = [('spin2', ['--intra', '2', '--no-spin']),
    ('spin3', ['--intra', '3', '--no-spin']), ('spin4', ['--intra', '4', '--no-spin']),
    ('cv2', ['--cv', '2']), ('cv4', ['--cv', '4']), ('sides', ['--sides']),
    ('idle', ['--idle']), ('sets4', ['--cpu-sets', '4'])]
PREPARATION = [('serial', ['--prepare', 'serial']),
    ('overlap', ['--prepare', 'overlap']), ('reuse', ['--prepare', 'reuse'])]


def main() -> None:
    sys.path.insert(0, str(ROOT/'scripts'))
    from live_parallel_probe import free_bytes, guard
    guard()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    schedule = [(f'{name}_{turn}', arguments)
                for turn, candidates in enumerate((CANDIDATES, list(reversed(CANDIDATES))))
                for name, arguments in candidates]
    if '--focused' in sys.argv:
        schedule = [(f'pair_{name}_{index}', opts)
                    for name, arguments in FOCUSED
                    for index, opts in enumerate(([], arguments, arguments, []))]
        schedule += [(f'prep_{name}_{turn}', args)
                     for turn, candidates in enumerate((PREPARATION, list(reversed(PREPARATION))))
                     for name, args in candidates]
    if '--final' in sys.argv:
        from prepare_live_parallel_soak import archive_baseline
        baseline = archive_baseline()
        schedule = [(f'video_{index}', ['--video-start', '650', *options])
                    for index, options in enumerate((['--source-root', str(baseline)],
                        ['--product'], ['--product'], ['--source-root', str(baseline)]))]
        schedule += [(f'prep2_{name}_{turn}', options)
                     for turn, candidates in enumerate((PREPARATION, list(reversed(PREPARATION))))
                     for name, options in candidates]
    for tag, arguments in schedule:
        guard()
        command = [sys.executable, str(ROOT/'scripts/live_parallel_probe.py'),
                   '--tag', tag, *arguments]
        started = time.time()
        with (OUT/f'{tag}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=log,
                env=dict(os.environ, PYTHONUTF8='1'), cwd=ROOT, check=False)
        rows.append(dict(tag=tag, returncode=result.returncode, start=started,
                         end=time.time(), free_bytes=free_bytes(), command=command))
        ledger = 'focused.json' if '--focused' in sys.argv else 'matrix.json'
        if '--final' in sys.argv:
            ledger = 'final_matrix.json'
        (OUT/ledger).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(tag, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f'{tag}が失敗。ログを確認して再開すること')


if __name__ == '__main__':
    main()
