"""監査以外の計算・状態を省略せず、完全同一入力だけを再利用する。"""
import numpy as np
from scripts.image_free_audit_reuse import FrameMemo, cv_wrapper, key_value


def test_only_audit_hits_and_results_do_not_alias() -> None:
    memo = FrameMemo()
    memo.active = True
    calls = []
    def calculate() -> dict:
        calls.append(1)
        return {'x': np.array([2, 3])}
    first = memo.call('pure', ('input',), calculate)
    memo.call('pure', ('input',), calculate)
    first['x'][0] = 99
    memo.audit = True
    reused = memo.call('pure', ('input',), calculate)
    assert len(calls) == 2 and reused['x'].tolist() == [2, 3]
    reused['x'][0] = 88
    assert memo.call('pure', ('input',), calculate)['x'][0] == 2


def test_content_dtype_shape_and_frame_boundaries_are_distinct() -> None:
    a = np.array([1, 2], dtype=np.uint8)
    assert key_value(a) != key_value(a.astype(np.int16))
    assert key_value(a) != key_value(a.reshape(1, 2))
    assert key_value(a) != key_value(np.array([1, 3], dtype=np.uint8))
    memo = FrameMemo(); memo.active = memo.audit = True
    assert memo.call('x', 1, lambda: 1) == 1
    memo.clear()
    assert memo.call('x', 1, lambda: 2) == 2


def test_memory_bound_and_destination_calls_are_never_skipped() -> None:
    memo = FrameMemo(limit=128); memo.active = memo.audit = True
    for i in range(10):memo.call('x', i, lambda: np.arange(16))
    assert memo.bytes <= 128
    calls = []
    def function(*args: object, **kwargs: object) -> int:
        calls.append(1)
        return len(calls)
    wrapped = cv_wrapper(memo, 'matchTemplate', function)
    assert wrapped(np.array([1]), 0, dst=np.array([0])) == 1
    assert wrapped(np.array([1]), 0, dst=np.array([0])) == 2
