"""対象2moduleの装着点だけ観測し、原collector一更新後に停止。全域profileは使わない。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
REPAIR = ROOT.parent / 'g2_normal_hand_basis_count_repair_2026-09-11_v1'
CONTEXT = ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1/observer.py'
LOOP = ROOT.parent / 'g2_collector_continuous_guard_2026-09-11_v1/continuous_guard.py'
STOP = 'limited_observer_targeted_after_one_update'
ROWS: list[dict[str, Any]] = []


def context(module: Any, stack: Any) -> None:
    old_init, old_wrapper = module.Recorder.__init__, module.update_wrapper
    def initialize(rec: Any, *args: Any, **kwargs: Any) -> None:
        old_init(rec, *args, **kwargs)
        ROWS.append(dict(kind='context_constructed', first=rec.expected[0], last=rec.expected[-1],
                         expected_count=len(rec.expected), object_id=id(rec)))
    def wrapper(original: Any, rec: Any) -> Any:
        wrapped = old_wrapper(original, rec)
        def update(pipe: Any, frame: int, clock: float, *args: Any, **kwargs: Any) -> Any:
            ROWS.append(dict(kind='context_called', frame=frame, selected=frame in rec.expected))
            return wrapped(pipe, frame, clock, *args, **kwargs)
        return update
    stack.callback(setattr, module.Recorder, '__init__', old_init)
    stack.callback(setattr, module, 'update_wrapper', old_wrapper)
    module.Recorder.__init__, module.update_wrapper = initialize, wrapper


def collector(module: Any, stack: Any) -> None:
    old = module.build
    def build(*args: Any, **kwargs: Any) -> Any:
        loop = old(*args, **kwargs)
        original = loop.collect_lean
        def collect(*values: Any, **options: Any) -> Any:
            returned = original(*values, **options)
            ROWS.append(dict(kind='original_collector_return', frame=values[2],
                             pipeline_id=id(values[1]), returned_none=returned is None))
            raise RuntimeError(STOP)
        loop.collect_lean = collect
        return loop
    stack.callback(setattr, module, 'build', old)
    module.build = build


class Loader:
    def __init__(self, original: Any, path: Path, stack: Any) -> None:
        self.original, self.path, self.stack = original, path, stack

    def create_module(self, spec: Any) -> Any:
        return self.original.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.original.exec_module(module)
        ROWS.append(dict(kind='target_module_loaded', path=str(self.path), name=module.__name__))
        (context if self.path == CONTEXT else collector)(module, self.stack)


def main() -> None:
    sys.path.insert(0, str(REPAIR))
    original = importlib.util.spec_from_file_location
    with ExitStack() as stack:
        def spec(name: Any, location: Any = None, **kwargs: Any) -> Any:
            result = original(name, location, **kwargs)
            path = Path(location).resolve() if location is not None else None
            if path in (CONTEXT, LOOP):
                result.loader = Loader(result.loader, path, stack)
            return result
        stack.callback(setattr, importlib.util, 'spec_from_file_location', original)
        importlib.util.spec_from_file_location = spec
        entry_spec = spec('_g2_targeted_probe_entry', REPAIR / 'run.py')
        entry = importlib.util.module_from_spec(entry_spec)
        entry_spec.loader.exec_module(entry)
        code = entry.main()
    with (ROOT / 'OBSERVER_PROBE_v4.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(rows=ROWS, original_exit=code, planned_stop=STOP, quality_gate_clear=False), stream, indent=2)
    assert code == 1 and len([r for r in ROWS if r['kind'] == 'original_collector_return']) == 1
    print(json.dumps(dict(rows=ROWS, original_exit=code, quality_gate_clear=False)))


if __name__ == '__main__':
    main()
