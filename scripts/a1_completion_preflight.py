"""A1採点の入力・ラベル・新規関数の静的規約を事前に確認する。"""
from __future__ import annotations

import ast
import json
from scripts.a1_completion_replay import OUT, ROOT, INPUT
from scripts.falsedeath_replay_20261001 import records
from scripts.eval_set2_regression_20261001 import fixed_labels
from scripts.a1_completion_score import configure_labels

MAX_FUNCTION_LINES = 50


def main() -> None:
    """未来ラベルは採点器でのみ読み、再生へ渡さない。"""
    assert all((INPUT/f'{name}.jsonl.gz').exists() for name in records())
    functions = []
    paths = list((ROOT/'scripts').glob('a1_completion_*.py'))
    paths += [ROOT/'src/exchange_completion_score.py']
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            length = node.end_lineno-node.lineno+1
            assert length <= MAX_FUNCTION_LINES, (path.name, node.name, length)
            assert node.returns is not None, (path.name, node.name)
            args = [a for a in node.args.args+node.args.kwonlyargs if a.arg not in ('self', 'cls')]
            assert all(a.annotation is not None for a in args), (path.name, node.name)
            functions.append(dict(file=path.name, function=node.name, lines=length))
    overlay = ROOT/'src/exchange_event_overlay.py'
    for node in ast.walk(ast.parse(overlay.read_text(encoding='utf-8'))):
        if isinstance(node, ast.FunctionDef) and node.name in ('__init__', '_initialize_candidate_inputs'):
            length = node.end_lineno-node.lineno+1
            assert length <= MAX_FUNCTION_LINES, (node.name, length)
            functions.append(dict(file=overlay.name, function=node.name, lines=length))
    configure_labels()
    value = dict(records=len(records()), fixed_labels=len(fixed_labels()), functions=functions)
    (OUT/'PREFLIGHT.json').write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
