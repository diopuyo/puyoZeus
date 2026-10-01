"""新規コードの規約と固定入力のハッシュを記録する。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('.')
OUT = ROOT/'logs/prefire_prediction/v5_experiment/code_manifest.json'
PATTERNS = ('src/prefire*v5*.py', 'scripts/prefire_v5*.py', 'tests/test_prefire*v5*.py')
MAX_FUNCTION_LINES = 50


def main() -> None:
    """新規関数の長さ・型ヒントと実行入力を一度に監査する。"""
    files = sorted({p for pattern in PATTERNS for p in ROOT.glob(pattern)})
    violations, hashes = [], {}
    for path in files:
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = node.end_lineno-node.lineno+1
            missing = [arg.arg for arg in node.args.args if arg.arg not in ('self', 'cls') and arg.annotation is None]
            if size > MAX_FUNCTION_LINES or missing or node.returns is None:
                violations.append(dict(path=str(path), function=node.name, lines=size, missing=missing))
    inputs = [Path('src/production_config.py'), OUT.parent/'samples.json']
    if (OUT.parent/'samples_ledger.json').exists():
        inputs.append(OUT.parent/'samples_ledger.json')
    for path in inputs:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(sha256=hashes, violations=violations)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if violations:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
