"""実configuredのgeneric aliasを保持して次のSession生成関数だけを組み立てる。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import live_adapter as A
import loader as L
import whole_dependencies as D


def main() -> None:
    assert not any(name == 'src' or name.startswith('src.') for name in sys.modules)
    with ExitStack() as stack:
        A.configured(stack)
        sys.path.insert(0, str(L.SNAPSHOT))
        path = L.SNAPSHOT / 'scripts/collect_boards_lean.py'
        spec = importlib.util.spec_from_file_location('_whole_session_import_collector', path)
        collector = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = collector
        spec.loader.exec_module(collector)
        first = L.bootstrap()
        before, paths = dict(sys.modules), list(sys.path)
        with ExitStack() as inner:
            scope = D.Scope(inner)
            selected, anchor, create = scope.prepare(first.load, (35370, 35410))
            runtime = sys.modules['_whole_runtime_session']
            assert create.__code__ is runtime.create.__code__
            assert selected is not collector.EventAccountingRecorder
            assert anchor.Capture.__module__ == '_whole_start_capture'
            assert anchor.Capture.__mro__[1].__module__ == '_whole_start_anchor_v2'
            assert runtime.CLIENT.worker_path() == D.ROOT / 'start_worker.py'
            assert runtime.V6.OLD.FRAMES == (35370, 35410)
            transport = sys.modules['_whole_parent_transport']
            imported = transport.__builtins__['__import__']('parent_validation')
            assert imported is sys.modules['_whole_parent_validation']
        difference = dict(paths_restored=sys.path == paths, added=sorted(set(sys.modules) - set(before)),
                          removed=sorted(set(before) - set(sys.modules)))
        print(json.dumps(difference), flush=True)
        assert sys.path == paths and not difference['removed']
        assert set(difference['added']) <= sys.stdlib_module_names
        assert all(sys.modules[key] is value for key, value in before.items())
        with ExitStack() as second:
            repeated = D.Scope(second)
            new_type, _, _ = repeated.prepare(first.load, (35370, 35410))
            assert new_type is not selected
        assert repeated.closed and sys.path == paths and not any(name in sys.modules for name in D.ALIASES)
        report = dict(actual_configuration=True, actual_frozen_collector=True,
                      repaired_observer_selected=True, anchor_missing_board_guard_loaded=True,
                      original_session_create_code=True, lazy_validation_same_module=True,
                      project_dependencies_restored=True, sequential_import_cycles=2, new_stdlib_modules=difference['added'],
                      sessions_created=0, child_processes=0, updates=0, quality_gate_clear=False)
    assert '_g2_live_probability_outer' not in sys.modules
    with (A.ROOT / 'WHOLE_SESSION_IMPORT_v4.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
