"""認識通知の隔離・順序と既存評価境界の接続を確認する。"""
from __future__ import annotations

import inspect
import pickle
from types import SimpleNamespace

from src.phase_j.live_bridge import PipelineView, RecognitionBridge, RecognitionNotice, build_live_generate, recognize
from src.phase_j.live_source import CapturedFrame


def test_notice_detaches_mutable_boards() -> None:
    original = {'board': [[1, 2]]}
    notice = RecognitionNotice(1, 0.1, 0.0, 0.01, pickle.dumps(original),
        PipelineView((1, 2), (None, None), (0, 0), (False, False)), 0)
    original['board'][0][0] = 9
    assert notice.result()['board'][0][0] == 1
    first = notice.result()
    first['board'][0][0] = 8
    assert notice.result()['board'][0][0] == 1
    assert notice.pipeline.tsumo_count('2P') == 2


def test_fifo_preserves_all_notifications() -> None:
    bridge = RecognitionBridge(False, lambda *args: None)
    for index in range(10):
        bridge._put(SimpleNamespace(frame=index))
    assert [notice.frame for notice in bridge._drain()] == list(range(10))
    assert bridge.queue_max == 10


def test_adapter_preserves_generate_signature() -> None:
    import scripts.visualize_advantage_overlay as overlay
    adapted = build_live_generate(overlay, RecognitionBridge(False, lambda *args: None))
    assert inspect.signature(adapted) == inspect.signature(overlay.generate)


def test_worker_failure_is_propagated() -> None:
    import pytest
    bridge = RecognitionBridge(False, lambda *args: None)
    bridge.error = ValueError('broken')
    bridge.done.set()
    bridge._recognize = lambda pipe: None
    with pytest.raises(RuntimeError, match='認識worker'):
        list(bridge.observations(None, None, 30, 0, 1, 1))


def test_recognition_freezes_notifications_before_next_frame() -> None:
    import numpy as np
    accumulator = SimpleNamespace(total_power=120, step_count=1)
    result = SimpleNamespace(board=np.array([[1]]))
    pipe = SimpleNamespace(update=lambda *args: result, tsumo_count=lambda side: 3,
        _formula_accum_1p=accumulator, _formula_last_read_1p=SimpleNamespace(valid=True),
        _score_ocr=None)
    notice = recognize(pipe, CapturedFrame(8, 1.0, 0.0, 0.0, np.zeros((1, 1, 3)), 2))
    accumulator.total_power = 999
    result.board[0, 0] = 9
    assert notice.pipeline.formula_totals == (120, None)
    assert notice.pipeline.formula_visible == (True, False)
    assert notice.result().board[0, 0] == 1
    assert notice.dropped_before == 2
