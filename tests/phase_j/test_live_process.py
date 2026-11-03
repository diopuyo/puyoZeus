"""プロセス通知の差分復元と特徴間引き境界を検査する。"""
import pickle
from types import SimpleNamespace

import numpy as np

from src.board import Board
from src.board_state_machine import BoardState
from src.recognition_pipeline import PipelineResult, SideResult
from src.phase_j.live_bridge import PipelineView, RecognitionNotice
from src.phase_j.live_process import NoticeDeltaCodec, ProcessRecognitionBridge, coalescing_overlay, coalescing_counter


def notice(frame: int, score: int = 0) -> RecognitionNotice:
    board = Board()
    side = SideResult('1P', BoardState.STABLE, board, None, board, None, score, 0, None)
    result = PipelineResult(frame, frame/30, True, side, side)
    return RecognitionNotice(frame, frame/30, 0, 1, pickle.dumps(result),
        PipelineView((0, 0), (None, None), (0, 0), (False, False)), 0)


def test_delta_carries_changes_without_raw_boards() -> None:
    encoder, decoder = NoticeDeltaCodec(), NoticeDeltaCodec()
    first = encoder.encode(notice(0))
    assert all('cnn_board' not in key for key in first[1])
    assert first[0].result_bytes == b''
    restored = decoder.decode(first).result()
    assert np.array_equal(restored.p1.confirmed_board._grid, notice(0).result().p1.confirmed_board._grid)
    second = encoder.encode(notice(1, 40))
    assert set(second[1]) == {'root', 'p1.score', 'p2.score'}
    assert decoder.decode(second).result().p1.score == 40


def test_all_notifications_including_repeated_score_are_preserved() -> None:
    encoder, decoder = NoticeDeltaCodec(), NoticeDeltaCodec()
    output = [decoder.decode(encoder.encode(notice(i, (i//2)*40))) for i in range(10)]
    assert [n.frame for n in output] == list(range(10))
    assert [n.result().p1.score for n in output] == [(i//2)*40 for i in range(10)]


def test_coalesce_does_not_skip_state_machine_notifications() -> None:
    class Base:
        def _refresh_features(self, snapshot: object, t_sec: float) -> None:
            seen.append(('features', t_sec))
        def _static(self, snapshot: object, t_sec: float) -> None:
            seen.append(('static', t_sec))
        def update(self, t_sec: float) -> None:
            seen.append(('notification', t_sec))
            self._refresh_features(None, t_sec)
            self._static(None, t_sec)
    bridge = SimpleNamespace(skip_features=True, feature_skips={})
    seen = []
    overlay = coalescing_overlay(Base, bridge)()
    overlay.tracker = SimpleNamespace(current=None)
    overlay.update(1.0)
    bridge.skip_features = False
    overlay.update(2.0)
    assert seen == [('notification', 1.0), ('notification', 2.0), ('features', 2.0), ('static', 2.0)]
    bridge.skip_features = True
    overlay.tracker.current = object()
    overlay.update(3.0)
    assert seen[-2:] == [('notification', 3.0), ('static', 3.0)]


def test_counter_skips_search_but_applies_zero_budget() -> None:
    class Base:
        _last_result = (0.1, 0.2, 0.3)
        def update(self, *args: object, **kwargs: object) -> tuple:
            self._last_result = (0.0, 0.0, 0.0)
            return self._last_result
    bridge = SimpleNamespace(skip_features=True, feature_skips={})
    counter = coalescing_counter(Base, bridge)()
    assert counter.update(None, None, 1.0) == (0.1, 0.2, 0.3)
    assert counter.update(None, None, 0.0) == (0.0, 0.0, 0.0)
    assert bridge.feature_skips == {'counter': 1}


def test_final_board_is_not_coalesced_by_eof_messages() -> None:
    bridge = object.__new__(ProcessRecognitionBridge)
    bridge.coalesce = True
    bridge.latest_frame = SimpleNamespace(value=20)
    assert bridge.feature_is_stale(19)
    assert not bridge.feature_is_stale(20)
    bridge.coalesce = False
    assert not bridge.feature_is_stale(19)
