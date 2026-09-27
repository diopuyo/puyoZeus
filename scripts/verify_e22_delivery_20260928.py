"""E22の関数長・固定入力ハッシュ・成果物を確認する。"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
from scripts.run_e3_exchange_eval_20260926 import save_json


def main() -> None:
    """本番設定・モデルの変更なしを後から照合できる形で記録する。"""
    scripts = list(Path('scripts').glob('*e22*.py'))
    paths = scripts+[Path('src/exchange_event_death_formula.py'), Path('src/exchange_event_overlay.py')]
    oversized = [(str(p), n.name, n.end_lineno-n.lineno+1) for p in paths
                 for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, ast.FunctionDef)
                 and n.end_lineno-n.lineno+1 > 50]
    print('oversized', oversized)
    inputs = [*Path('logs/e16/records').glob('*.jsonl.gz'), Path('src/production_config.py'),
              *[p for p in Path('models/exchange_event_v3').rglob('*') if p.is_file()]]
    save_json(Path('logs/e22/INPUTS.json'), {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    assert not oversized


if __name__ == '__main__':
    main()
