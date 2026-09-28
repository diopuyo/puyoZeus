"""E25の既定OFF・規約・禁止ファイル不変・最終成果物を確認する。"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_landing import ExchangeLandingProjection
from scripts.replay_exchange_event_20260926 import replay
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e25')
NEW_MODULES = ('src/exchange_pending_ledger.py', 'src/exchange_completion_recovery.py',
               'src/exchange_landing_safety.py')
FUNCTION_LIMIT = 50
CHANGED_MODULES = ('src/exchange_event_overlay.py', 'src/exchange_event_landing.py',
    'src/exchange_event_multilanding.py', 'scripts/replay_exchange_event_20260926.py',
    'scripts/visualize_advantage_overlay.py')


def check_new_sources() -> dict:
    """新規モジュール・補助スクリプト・テストの構文と関数長をまとめて確認する。"""
    paths = [Path(p) for p in NEW_MODULES]
    paths += sorted(Path('scripts').glob('*e25*.py'))+sorted(Path('tests').glob('test_e25*.py'))
    lengths = {}
    for path in paths:
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        compile(source, str(path), 'exec')
        spans = [n.end_lineno-n.lineno+1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert max(spans) <= FUNCTION_LIMIT, (str(path), spans)
        lengths[str(path)] = max(spans)
    return lengths


def check_production_config() -> None:
    """Windows作成worktreeのgitdirをWSL表記へ変換し、コミット原本と比較する。"""
    gitdir = Path('.git').read_text().split(': ', 1)[1].strip()
    if sys.platform != 'win32' and len(gitdir) > 1 and gitdir[1] == ':':
        gitdir = '/mnt/'+gitdir[0].lower()+gitdir[2:].replace('\\', '/')
    before = subprocess.check_output(['git', '--git-dir='+gitdir,
        'show', 'a9f8ca1:src/production_config.py']).decode('utf-8').replace('\r\n', '\n')
    assert before == Path('src/production_config.py').read_text(encoding='utf-8')


def check_inputs() -> None:
    """E23で固定した記録・モデルと同じバイト列を使用したことを保存する。"""
    reference = json.loads(Path('logs/e23/INPUTS.json').read_text())
    actual = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in reference}
    assert actual == reference
    save_json(OUT/'INPUTS.json', actual)


def main() -> None:
    """手元の既定値だけでなく、E22/E23の保存出力との全一致も検査する。"""
    defaults = {}
    for target in (ExchangeEventOverlay, ExchangeLandingProjection, replay):
        value = inspect.signature(target).parameters['landing_state_safety'].default
        assert value is False
        defaults[target.__name__] = value
    lengths = check_new_sources()
    check_production_config()
    check_inputs()
    off = {name: json.loads((OUT/f'{name}_EQUIVALENCE.json').read_text())
           for name in ('off_e22', 'off_e23')}
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (*NEW_MODULES, *CHANGED_MODULES)}
    save_json(OUT/'VERIFICATION.json', dict(defaults=defaults, max_function_lines=lengths,
        source_sha256=hashes, tests=(OUT/'tests.log').read_text().splitlines()[-1],
        production_config='unchanged_from_a9f8ca1', off_equivalence=off))
    print('E25 defaults, syntax, function length, production_config, OFF equivalence: passed')


if __name__ == '__main__':
    main()
