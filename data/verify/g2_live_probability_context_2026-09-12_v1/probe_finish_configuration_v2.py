"""追加identityモジュールも含む元設定/解除検査。旧結果を上書きしない。"""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    source = ROOT / 'probe_finish_configuration.py'
    tree = ast.parse(source.read_bytes())
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and n.value == 'FINISH_CONFIGURATION_v1.json']
    assert len(hits) == 1
    hits[0].value = 'FINISH_CONFIGURATION_v2.json'
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'),
         dict(__name__='__main__', __file__=str(source)))


if __name__ == '__main__':
    main()
