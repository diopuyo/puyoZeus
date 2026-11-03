"""H1′の規約、既定OFFの既存原票一致、完了した資源記録を検査する。"""
from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT
from scripts.falsedeath_v2d_static_checks import check_python

ROOT = Path(__file__).resolve().parents[1]
DEST = OUT/'v2e'
MODIFIED = ('scripts/falsedeath_replay_20261001.py', 'scripts/falsedeath_score_20261001.py',
    'scripts/falsedeath_v2_ladder_20261002.py', 'scripts/replay_exchange_event_20260926.py',
    'scripts/visualize_advantage_overlay.py', 'src/exchange_event_overlay.py')


def default_off() -> dict:
    """既定OFF再生をD-v2dの同一接頭区間と配列・H1証拠で比較する。"""
    candidate, baseline = DEST/'default_off', OUT/'replay/v2d/c1'
    with np.load(candidate/'display.npz') as current, np.load(baseline/'display.npz') as prior:
        assert set(current.files) == set(prior.files)
        for key in current.files:
            old = prior[key][:len(current[key])] if current[key].ndim else prior[key]
            np.testing.assert_equal(current[key], old, err_msg=key)
        frames = len(current['t_sec'])
    left = json.loads((candidate/'history_integrity_audit.json').read_text())
    right = json.loads((baseline/'history_integrity_audit.json').read_text())
    assert left['rows'] == right['rows'][:len(left['rows'])]
    return dict(display_frames=frames, columns=current.files, identical=True,
                audit_rows=len(left['rows']))


def main() -> None:
    """新規コードと変更コードの構文・全新規関数の型注釈と行数を検査する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='VERIFICATION.json')
    args = parser.parse_args()
    paths = sorted((ROOT/'scripts').glob('*v2e*.py'))
    paths += [ROOT/'src/exchange_history_integrity.py', ROOT/'tests/test_falsedeath_v2e.py']
    checked = [check_python(path) for path in paths]
    for name in MODIFIED:
        ast.parse((ROOT/name).read_text(encoding='utf-8'))
    shells = sorted((ROOT/'scripts').glob('*v2e*.sh'))
    for path in shells:
        subprocess.run(['bash', '-n', str(path)], check=True)
    result = dict(python=checked, shell=[str(p.relative_to(ROOT)) for p in shells],
                  default_off=default_off())
    with (DEST/args.output_name).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=1)
    print(json.dumps(result['default_off'], ensure_ascii=False))


if __name__ == '__main__':
    main()
