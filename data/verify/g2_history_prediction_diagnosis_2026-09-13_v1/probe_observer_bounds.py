"""実A29構成で観測範囲の不一致を列挙する。モデル・動画は生成しない。"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_early_origin_runtime_2026-09-13_v29'
sys.path.insert(0, str(RUNTIME))
import owned_adapter as A


def snapshot() -> list[dict[str, Any]]:
    rows = []
    for alias, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not path or not Path(path).resolve().is_relative_to(ROOT.parent):
            continue
        values = {key: value for key in ('WINDOWS', 'OUTER_FIRST', 'OUTER_LAST',
            'FIRST', 'END', 'END_FRAME', 'FIRST_FRAME', 'LAST', 'STRIDE')
            if type(value := getattr(module, key, None)) in (int, tuple)}
        if not values:
            continue
        row = dict(alias=alias, path=str(Path(path).resolve()), values=values)
        if Path(path).name == 'observer.py' and callable(getattr(module, 'expected', None)):
            scopes = module.expected()
            row['expected'] = dict(count=len(scopes), first=scopes[0], last=scopes[-1])
        rows.append(row)
    return rows


def main() -> None:
    started = time.perf_counter()
    with ExitStack() as stack:
        selected = A.configured(stack)
        session = selected.__globals__['S']
        with session.configured() as env:
            rows = snapshot()
            assert env['runtime'] is not None
    value = dict(seconds=time.perf_counter() - started, actual_a29_configured=True,
        model_created=False, video_updates=0, quality_gate_clear=False, modules=rows)
    with (ROOT / 'OBSERVER_BOUNDS_v1.json').open('x') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(dict(seconds=value['seconds'], modules=len(rows),
        observers=[row for row in rows if 'expected' in row])))


if __name__ == '__main__':
    main()
