"""実configuredの終了wrapperとalias寿命だけ検査。factory/update/evaluateは0回。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import sys
import live_adapter as A


def main() -> None:
    before = list(sys.path)
    with ExitStack() as stack:
        configured = A.configured(stack)
        target = configured.__globals__['Q'].FINAL
        wrapped = target.evaluate
        assert Path(wrapped.__code__.co_filename).resolve() == A.ROOT / 'probability_finish.py'
        previous = inspect.getclosurevars(wrapped).nonlocals['previous']
        original = inspect.getclosurevars(previous).nonlocals['FINISH']
        assert Path(original.__file__).resolve() == A.PRIOR / 'live_finish_v2.py'
        assert all(name in sys.modules for name in A.FINISH_MODULES)
    assert all(name not in sys.modules for name in A.FINISH_MODULES)
    assert sys.path == before and target.evaluate is not wrapped
    torch = sys.modules.get('torch')
    assert torch is None or not torch.cuda.is_initialized()
    result = dict(configuration_verified=True, original_v2_selected=True, references_restored=True,
                  factory_calls=0, updates=0, evaluate_calls=0, actual_video=False, quality_gate_clear=False)
    with (A.ROOT / 'FINISH_CONFIGURATION_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
