"""v68負例が期待する拒否理由を原CODE.verifyで固定する。実P.verifyは別統合。"""
import importlib.util
from pathlib import Path
import sys

import pytest

BASE = Path(__file__).resolve().parent.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1'
sys.path.insert(0, str(BASE))
import completion_code as C


def test_original_code_guard_reason_and_full_restore() -> None:
    spec = importlib.util.spec_from_file_location('_finish_pin_original_completion', C.SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    C.verify(module)
    before = dict(vars(module))
    try:
        module.verify = lambda *args: None
        with pytest.raises(AssertionError, match='^samecall_global_identity$'):
            C.verify(module)
    finally:
        module.verify = before['verify']
    assert set(vars(module)) == set(before)
    assert all(vars(module)[key] is value for key, value in before.items())
    C.verify(module)
