"""監査の網羅性・変更範囲・原票ハッシュを検証して記録する。"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._audit_firepower_1003_inventory import OUT, ROOT, resources
from scripts._audit_firepower_1003_replay import NAMES, display_hashes

BASELINE = '802efbcbc38156e0bd521e9439a7090f2e56c549'
MAX_FUNCTION_LINES = 50


def hashfile(path: Path) -> str:
    """原票をメモリへ丸ごと展開せずハッシュ化する。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def production_diff() -> str:
    """Windows側Gitで改行・実行属性の環境差を除いて検査する。"""
    executable = '/mnt/c/Program Files/Git/mingw64/bin/git.exe'
    return subprocess.check_output([executable, 'diff', BASELINE,
        '--', 'src', 'models', 'tests'], text=True)


def check_scripts() -> list[dict]:
    """型ヒントと関数長を、生成した監査スクリプトだけ検査する。"""
    rows = []
    for path in sorted(Path('scripts').glob('_audit_firepower_1003_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                size = node.end_lineno-node.lineno+1
                assert size <= MAX_FUNCTION_LINES, (path, node.name, size)
                assert node.returns is not None, (path, node.name, 'return type')
                assert all(a.annotation for a in node.args.args), (path, node.name, 'argument type')
        rows.append(dict(path=str(path), sha256=hashfile(path)))
    return rows


def main() -> None:
    """採否には使わず、監査成果物の再現性だけを確認する。"""
    res = resources()
    if '--scripts-only' in sys.argv:
        print(json.dumps(check_scripts(), ensure_ascii=False))
        return
    assert json.loads((OUT / 'AGGREGATE.json').read_text())['complete']
    assert json.loads((OUT / 'CAPTURE_COMPLETE.json').read_text())
    pop = pd.read_csv(OUT / 'population.csv')
    assert pop.games.tolist() == [57, 58], pop
    inputs, equality = [], []
    for name in NAMES:
        folder = OUT / 'replay' / name
        assert (folder / 'EXTRACTED.json').exists()
        frame = pd.read_csv(folder / 'frames.csv.gz')
        assert frame.shown_p1.notna().all(), name
        assert frame.shown_p1.between(0, 1).all(), name
        assert np.isfinite(frame.rate).all(), name
        item = folder / 'INSTRUMENTATION_EQUIVALENCE.json'
        if item.exists():
            assert json.loads(item.read_text())['identical'], name
            equality.append(name)
        original = ROOT / 'collect/records' / f'{name}.jsonl.gz'
        inputs.append(dict(segment=name, record=str(original), sha256=hashfile(original),
                           display=display_hashes(folder)))
    diff = production_diff()
    assert not diff, '監査中に本番実装が変わった'
    result = dict(baseline=BASELINE, verified_utc=datetime.now(timezone.utc).isoformat(), resources=res,
        populations=pop.to_dict('records'), scripts=check_scripts(), inputs=inputs,
        instrumentation_equivalent_segments=equality, production_diff_empty=True,
        models={str(p): hashfile(p) for p in Path('models').glob('exchange_event_v*/manifest.json')})
    (OUT / 'VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('verification complete', flush=True)


if __name__ == '__main__':
    main()
