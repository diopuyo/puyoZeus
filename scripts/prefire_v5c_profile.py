"""5B通知を変更前に計測し、関数別プロファイルを保存する。"""
from __future__ import annotations

import cProfile
import argparse
import json
import pstats
from pathlib import Path
import subprocess
from types import ModuleType

from scripts import prefire_v5b_verify as verify

OUT = Path('logs/prefire_prediction/v5c/profile')
BASELINE_COMMIT = 'b676bc8354aafcd6348900111cb5f54d539d2ad7'
GIT = '/mnt/c/Program Files/Git/mingw64/bin/git.exe'


def baseline_layer() -> type:
    """指定HEADのPython探索と通知層を別名前空間で復元する。worktreeは変更しない。"""
    modules = []
    for name in ('prefire_v5b_search', 'prefire_best_play_v5'):
        path = f'src/{name}.py'
        source = subprocess.check_output([GIT, 'show', f'{BASELINE_COMMIT}:{path}'])
        module = ModuleType(f'baseline_{name}')
        exec(compile(source, f'{BASELINE_COMMIT}/{path}', 'exec'), vars(module))
        modules.append(module)
    modules[1].exhaustive = modules[0]
    return modules[1].BestPlayV5Layer


def main() -> None:
    """固定100局面の先頭をモデルロード後に測る。"""
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', action='store_true')
    args = parser.parse_args()
    name = 'after' if args.after else 'before'
    if (OUT/f'{name}.json').exists():
        raise FileExistsError('既存プロファイルを上書きしない')
    row = json.loads((verify.reference.OUT/'samples_ledger.json').read_text())[0]
    overlay = verify.reference.model_overlay()
    from src import prefire_best_play_v5 as layer
    original = layer.BestPlayV5Layer
    if not args.after:
        layer.BestPlayV5Layer = baseline_layer()
    profile = cProfile.Profile()
    try:
        result = profile.runcall(verify.benchmark, row, overlay)
    finally:
        layer.BestPlayV5Layer = original
    profile.dump_stats(str(OUT/f'{name}.prof'))
    (OUT/f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (OUT/f'{name}.txt').open('w', encoding='utf-8') as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(80)
    print(result, flush=True)


if __name__ == '__main__':
    main()
