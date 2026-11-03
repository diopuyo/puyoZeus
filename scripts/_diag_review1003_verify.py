"""診断成果物の時刻・再生一致・スクリプト制約を検査する。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts._diag_review1003_extract import OUT, WINDOWS, resources


def main() -> None:
    """製品の合否と独立した診断成果物の整合検査。"""
    resource = resources()
    functions = []
    for path in sorted(Path('scripts').glob('_diag_review1003_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                length = node.end_lineno - node.lineno + 1
                assert length <= 50, (str(path), node.name, length)
                assert node.returns is not None, (str(path), node.name)
                functions.append(dict(path=str(path), function=node.name, lines=length))
    counts = []
    for game, start, end in WINDOWS:
        rows = json.loads((OUT / f'game{game:02}_joined.json').read_text())
        for row in rows:
            stamp = float(row['t_sec'])
            assert start <= stamp <= end
            assert row['inputs']['t_sec'] <= stamp
            assert all(b['t_sec'] <= stamp for b in row['inputs']['boards'])
            assert float(row['new_p1']) == float(row['p1_selected'])
        counts.append(len(rows))
    assert counts == [421, 210, 300], counts
    value = dict(resources=resource, functions=functions, frames=counts,
                 total=sum(counts), future_input_rows=0)
    (OUT / 'artifact_checks.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print('診断成果物の検査完了:', sum(counts), 'フレーム', len(functions), '関数')


if __name__ == '__main__':
    main()
