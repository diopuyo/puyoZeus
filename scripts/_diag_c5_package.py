"""診断の入力範囲・構文・許可出力の受領票を保存する。"""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

import numpy as np

from scripts._diag_c5_inspect import BASE, OUT, ROOT, START, END, guard, save


def timeline_metadata() -> dict[str, Any]:
    """settled/displayの実際の範囲と該当行を保存する。"""
    output = {}
    for name in ('settled', 'display'):
        path = BASE / f'collect/capture/c5/{name}.npz'
        with np.load(path, allow_pickle=True) as data:
            indices = np.flatnonzero((data['t_sec'] >= START) & (data['t_sec'] <= END))
            values = {key: data[key][indices].tolist() for key in data.files if data[key].ndim}
            output[name] = dict(first=float(data['t_sec'][0]), last=float(data['t_sec'][-1]),
                                window_rows=len(indices), values=values)
    return output


def main() -> None:
    """新規診断以外に手を加えず、パッケージを検証する。"""
    available = guard()
    source_rows = []
    for path in sorted((ROOT / 'scripts').glob('_diag_c5_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno-node.lineno+1 <= 50, (path.name, node.name)
        source_rows.append(str(path.relative_to(ROOT)))
    save('timeline_metadata.json', timeline_metadata())
    raw_count = len(json.loads((OUT / 'raw_window.json').read_text()))
    assert raw_count == 601, raw_count
    save('RECEIPT.json', dict(scope='診断のみ・合否外', scripts=source_rows,
         raw_updates=raw_count, nice=10, available_bytes=available, syntax_and_function_size='確認済み'))
    files = []
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name != 'FILES.json':
            files.append(dict(path=str(path.relative_to(OUT)), bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save('FILES.json', files)
    print(json.dumps(dict(files=len(files), bytes=sum(r['bytes'] for r in files), updates=raw_count)))


if __name__ == '__main__':
    main()
