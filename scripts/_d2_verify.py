"""D2の診断結果・再現一致・診断スクリプトの制約を検証する。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts._d2_inventory import OUT, save

MAX_FUNCTION_LINES = 50
ROWS_PER_RUN = 300
EXPECTED_TOTAL_ROWS = 114146
EXPECTED_DIFFERENT_ROWS = 81834


def read(name: str) -> dict:
    """完成した診断原票を読む。"""
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def scripts() -> list[str]:
    """構文・型注釈・関数長を確認し、評価コードのテスト追加を避ける。"""
    checked = []
    for path in sorted(Path('scripts').glob('_d2_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        compile(tree, str(path), 'exec')
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assert node.end_lineno-node.lineno+1 <= MAX_FUNCTION_LINES, (path, node.name)
            assert node.returns is not None, (path, node.name)
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            assert all(a.annotation is not None for a in args), (path, node.name)
        checked.append(str(path))
    return checked


def main() -> None:
    """短区間で再現した事実だけを機械検証する。"""
    comparisons, summary = read('comparisons.json'), read('summary.json')
    audit = read('original_board_audit.json')
    assert sum(v['e31_to_e34b']['rows'] for v in audit.values()) == EXPECTED_TOTAL_ROWS
    assert sum(v['e31_to_e34b']['boards'] for v in audit.values()) == EXPECTED_DIFFERENT_ROWS
    assert all(v['original_to_e31']['boards'] == v['original_to_e31']['states'] == 0
               for v in audit.values())
    for source in ('q', 'fc', 'mia', 'zenchi', 'review'):
        for variant, stage in (('sig', 'e31'), ('new', 'e34b')):
            result = comparisons['runs'][f'{source}_{variant}'][stage]
            assert result['rows'] == ROWS_PER_RUN
            assert all(v == 0 for v in result['per_field'].values()), (source, variant)
    for name in ('q_hsv', 'q_start', 'q_old', 'zenchi_landing_hsv_carry',
                 'review_landing_hsv_carry'):
        assert comparisons['runs'][name]['e31']['different_rows'] == 0
    assert all(v['equal'] for v in summary['repeat_hashes'].values())
    assert all(all(v['equal'][k] for k in ('cnn_board', 'hsv_board', 'state'))
               for v in read('first_frames.json').values())
    resets = read('reset_events.json')
    assert resets['q_reset'][0]['frame_idx'] == 53
    assert all(r['caller'] != '_step_side' for e in resets['q_reset_old'] for r in e['resets'])
    save('verification.json', dict(state='passed', scripts=scripts(), sources=5,
         rows_per_condition=ROWS_PER_RUN, repetitions_equal=True,
         original_rows_unchanged=EXPECTED_TOTAL_ROWS))


if __name__ == '__main__':
    main()
