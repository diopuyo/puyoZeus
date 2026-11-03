"""JSONの丸め差を省略せず保存する診断。合否の許容幅を設けない。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any


def differences(a: Any, b: Any, path: str = '$') -> list[dict]:
    if type(a) is not type(b):
        return [dict(path=path, expected=a, actual=b)]
    if isinstance(a, dict):
        return [d for key in sorted(a.keys() | b.keys())
                for d in differences(a.get(key), b.get(key), path+'.'+key)]
    if isinstance(a, list):
        if len(a) != len(b):
            return [dict(path=path, expected_length=len(a), actual_length=len(b))]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in differences(x, y, f'{path}[{i}]')]
    return [] if a == b else [dict(path=path, expected=a, actual=b)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('before', 'after', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    values = differences(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    with args.out.open('x') as stream:
        json.dump(values, stream, indent=2)
    print(json.dumps(values, indent=2))
