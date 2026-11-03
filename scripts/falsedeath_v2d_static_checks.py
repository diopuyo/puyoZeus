"""新規Pythonの構文・関数長・型注釈と起動口の構文を点検する。"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/eval_set/falsedeath_fix/v2d'
MAX_FUNCTION_LINES = 50


def check_python(path: Path) -> dict:
    """既存の巨大関数とは分け、新規ファイルに規約を適用する。"""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        length = node.end_lineno-node.lineno+1
        assert length <= MAX_FUNCTION_LINES, (str(path),node.name,length)
        assert node.returns is not None, (str(path),node.name,'return')
        for arg in (*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs):
            assert arg.arg in ('self','cls') or arg.annotation is not None, (str(path),node.name,arg.arg)
        functions.append(dict(name=node.name, lines=length))
    return dict(path=str(path.relative_to(ROOT)), functions=functions)


def main() -> None:
    """シェルは実行せずbash -nで検査し、新規原票へ保存する。"""
    paths = sorted((ROOT/'scripts').glob('*v2d*.py'))
    paths += [ROOT/'src/exchange_history_integrity.py',ROOT/'tests/test_falsedeath_v2d.py']
    python = [check_python(path) for path in paths]
    shell = sorted((ROOT/'scripts').glob('*v2d*.sh'))
    for path in shell:
        subprocess.run(['bash','-n',str(path)],check=True)
    with (OUT/'STATIC_CHECK.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(python=python,shell=[str(p.relative_to(ROOT)) for p in shell]),stream,indent=1)
    print(f'新規Python {len(paths)}本・起動シェル {len(shell)}本の構文と規約を確認')


if __name__ == '__main__':
    main()
