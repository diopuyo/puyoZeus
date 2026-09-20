"""実configuredへの外側接続と解除。生成/update/実終了の合格には使わない。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
import json
import sys
import live_adapter as A


def main() -> None:
    before = list(sys.path)
    with ExitStack() as stack:
        selected = A.configured(stack)
        repeated = selected.__globals__['REPEAT']
        top, raw = repeated.verify, repeated.scope_status
        outer = inspect.getclosurevars(top).nonlocals
        middle = outer['checked']
        inner = inspect.getclosurevars(middle).nonlocals['original_verify']
        chosen = inspect.getclosurevars(inner).nonlocals
        assert set(chosen) == {'adapted', 'function'}
        assert chosen['function'].__globals__['scope_status'] is raw
        assert chosen['adapted'].__globals__['scope_status'] is raw
        assert '_g2_live_probability_outer' in sys.modules
    assert sys.path == before and repeated.verify is not top
    assert '_g2_live_probability_outer' not in sys.modules
    assert not any(name in sys.modules for name in A.FINISH_MODULES)
    report = dict(actual_configured=True, outer_two_wrappers_preserved=True, raw_scope_unchanged=True,
                  references_restored=True, factory_calls=0, updates=0, actual_video=False,
                  full_probability_finalizer_verified=False, quality_gate_clear=False)
    with (A.ROOT / 'OUTER_CONFIGURATION_v2.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
