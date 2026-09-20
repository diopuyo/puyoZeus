"""実configuredのconstructor closureから保存producerの親を確認。生成/更新はしない。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import sys
import live_adapter as A


def main() -> None:
    paths = list(sys.path)
    with ExitStack() as stack:
        main = A.configured(stack)
        repeated = main.__globals__['REPEAT']
        modules = inspect.getclosurevars(repeated.load).nonlocals['modules']
        parent = modules.proof.Context
        status = sys.modules['proof_status']
        assert Path(status.__file__).resolve() == A.PRIOR / 'proof_status.py'
        assert parent.close.__globals__['close'] is status.close
        assert Path(parent.close.__code__.co_filename).resolve() == A.PRIOR / 'proof_status.py'
        assert parent.__name__ == 'Context'
    assert sys.path == paths and '_g2_live_probability_outer' not in sys.modules
    report = dict(actual_constructor_closure=True, actual_status_parent=True,
                  factory_calls=0, updates=0, references_restored=True, quality_gate_clear=False)
    with (A.ROOT / 'CONSTRUCTOR_PARENT_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
