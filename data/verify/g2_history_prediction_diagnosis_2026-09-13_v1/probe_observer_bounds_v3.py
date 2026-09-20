"""実A30構成で観測範囲の不一致を列挙する。モデル・動画は生成しない。"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_early_origin_runtime_2026-09-13_v30'
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
    captured = []
    original_bind = A.OC.bind
    def bind(stack: Any, latest: Any, replace: Any) -> Any:
        values = original_bind(stack, latest, replace)
        captured.append(values)
        return values
    A.OC.bind = bind
    with ExitStack() as stack:
        selected = A.configured(stack)
        session = selected.__globals__['S']
        with session.configured() as env:
            rows = snapshot()
            assert env['runtime'] is not None
            expected = {row['alias']: row['expected'] for row in rows if 'expected' in row}
            assert expected['_context_outer_observer']['last'] == A.OC.LAST
            assert expected['_hidden_capture_live_observer']['count'] == 7850
            assert expected['_hsv_live_observer']['count'] == 3925
            actual = env['runtime'].M.A.entry.previous.history
            assert actual.HistoryRecorder.begin_frame.__globals__['END_FRAME'] == 36902
        assert captured and all(value['metadata'].LAST == 36900 for value in captured)
    assert all(value['scope'].WINDOWS == ((29052, 36298),)
        and value['context'].OUTER_LAST == value['metadata'].LAST == 36298 for value in captured)
    A.OC.bind = original_bind
    value = dict(seconds=time.perf_counter() - started, actual_a30_configured=True,
        model_created=False, video_updates=0, quality_gate_clear=False, modules=rows)
    with (ROOT / 'OBSERVER_BOUNDS_v3.json').open('x') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(dict(seconds=value['seconds'], modules=len(rows),
        observers=[row for row in rows if 'expected' in row])))


if __name__ == '__main__':
    main()
