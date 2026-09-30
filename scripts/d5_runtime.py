"""並行E35bの編集を取り込まず、指定E35コミットの証明器を読む。"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

from scripts.run_e3_exchange_eval_20260926 import save_json

COMMIT = '0f28a04'
MODULES = ('post_counter_death_bound', 'exchange_post_counter_bound')
GIT = '/mnt/c/Program Files/Git/mingw64/bin/git.exe'
OUT = Path('logs/d5/runtime')


def install() -> None:
    """元ファイルを書き換えず、D5プロセス内だけ指定版を読み込む。"""
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in MODULES:
        source = subprocess.check_output([GIT, 'show', f'{COMMIT}:src/{name}.py'])
        path = OUT/f'{name}.py'
        if not path.exists():
            path.write_bytes(source)
        assert path.read_bytes() == source
        hashes[name] = hashlib.sha256(source).hexdigest()
        spec = importlib.util.spec_from_file_location(f'src.{name}', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    save_json(OUT/'manifest.json', dict(commit=COMMIT, sha256=hashes))
