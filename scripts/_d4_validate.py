"""D4の母数・加法性・介入母数・観測非干渉を検収する。"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts._d4_loss import OUT, Q, save

DENOMINATOR = 6526


def read(name: str) -> object:
    """指定成果物を読む。"""
    return json.loads((OUT/name).read_text())


def check_scripts() -> list[dict]:
    """診断用スクリプトの構文・型注釈・関数長を確認する。"""
    result = []
    for path in sorted(Path('scripts').glob('_d4_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.returns is not None, (path.name, node.name, 'return_type')
                count = node.end_lineno-node.lineno+1
                assert count<=50, (path.name, node.name, count)
        result.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


def main() -> None:
    """診断に必要な不変条件だけを検査し、本番テスト群は変更しない。"""
    rows, summary = read('q_rows.json'), read('q_summary.json')
    assert len(rows)==DENOMINATOR
    original = json.loads(Path('logs/r1/SUMMARY.json').read_text())
    for mode in ('off','on'):
        assert np.isclose(summary[mode], original['metrics'][mode]['q']['log_loss'], atol=1e-12, rtol=0)
    for name in ('q_intervals.json','q_matches.json','q_chains.json'):
        groups = read(name)
        assert sum(r['n'] for r in groups)==DENOMINATOR
        assert np.isclose(sum(r['delta_sum'] for r in groups), summary['delta_sum'])
    assert len(read('q_top10_seconds.json'))==10
    assert all(r['delta_sum']>0 for r in read('q_top10_seconds.json'))
    assert len(read('residual_cells.json'))==379
    assert sum(read('residual_summary.json')['categories'].values())==379
    for mode in ('off','on'):
        for source in (Q,'review'):
            assert read(f'replay/{mode}/{source}/EQUIVALENCE.json')['display']=='byte_identical'
    for signal in ('next','ojama','formula'):
        assert read(f'completion_ablation/{signal}/RESULT.json')['n']==DENOMINATOR
    assert read('SCOPE.json')['production_code_changed'] is False
    result = dict(passed=True, rows=DENOMINATOR, residual=379, equivalent_replays=4,
        scripts=check_scripts(), base_commit='58ae8a7', production_code_changed=False)
    save('VALIDATION.json', result)
    print(json.dumps(dict(passed=True, rows=DENOMINATOR, residual=379, equivalent_replays=4)))


if __name__ == '__main__':
    main()
