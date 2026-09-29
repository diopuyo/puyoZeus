"""E23の関数長・本番設定不変・固定入力を監査する。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from scripts.run_e3_exchange_eval_20260926 import save_json


def main() -> None:
    """E22入力のハッシュと照合し、比較中の入力や本番設定の変更を排除する。"""
    paths = [*Path('scripts').glob('*e23*.py'), Path('src/exchange_event_multilanding.py')]
    oversized = [(str(p), n.name, n.end_lineno-n.lineno+1) for p in paths
                 for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, ast.FunctionDef)
                 and n.end_lineno-n.lineno+1 > 50]
    assert not oversized, oversized
    expected = json.loads(Path('logs/e22/INPUTS.json').read_text())
    actual = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in expected}
    assert actual == expected
    save_json(Path('logs/e23/INPUTS.json'), actual)
    save_json(Path('logs/e23/DELIVERY_CHECK.json'), dict(function_length_passed=True,
        fixed_inputs_identical=True, production_config_unchanged=True))


if __name__ == '__main__':
    main()
