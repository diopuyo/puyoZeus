"""実configuredの原collect code保持と設定参照復元だけを検査。constructorは呼ばない。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import whole_live_adapter as A


def main() -> None:
    with ExitStack() as stack:
        value = A.configured(stack)
        selected = value.__globals__['collect']
        guard = selected.__globals__['constructor_guard']
        captured = inspect.getclosurevars(guard).nonlocals
        assert captured['original_guard'].__name__ == 'constructor_guard'
        assert tuple(captured['common'].FRAMES)[-1] == 36298
        session = value.__globals__['S']
        assert session.prepare.__globals__['K'] is session.K
        assert session.K.guards.__name__ == 'guarded'
        guarded = session.K.guards
        common = captured['common']
        assert value.__globals__['K'] is common
        session.K.whole_test_marker = object()
        assert session.prepare.__globals__['K'].whole_test_marker is session.K.whole_test_marker
        del session.K.whole_test_marker
        alias_after = value.__globals__['K'] is session.K
        pins = session.K.guards()
        assert all(str(A.A.ROOT / name) in pins for name in A.FILES)
        assert all(str(path) in pins for path in A.D.source_paths())
        namespace = value.__globals__
    assert namespace['collect'] is not selected
    assert namespace['collect'].__code__ is selected.__code__
    assert session.K.guards is not guarded
    report = dict(actual_configuration=True, original_collect_code_preserved=True,
                  references_restored=True, constructor_calls=0, updates=0,
                  full_whole_integration=False, prepare_guard_connected=True,
                  original_common_object_preserved=True, common_alias_after=alias_after,
                  prepare_source_guards_executed=len(pins), quality_gate_clear=False)
    root = Path(__file__).resolve().parent
    with (root / 'WHOLE_ADAPTER_v7.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
