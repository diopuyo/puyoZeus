"""通知単位EMA、画像再利用窓、欠落・入力切替を検証する。"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from src.exchange_event_overlay import ExchangeEventOverlay
from src.phase_j.live_notification_eval import NotificationExchangeOverlay, latest_display
from src.phase_j.live_snapshot import LiveSnapshotInputs, WINDOW_FRAMES, HISTORY_FRAMES
from scripts.visualize_advantage_overlay import _ExchangeDisplayEMA, _exchange_display
from tests.test_exchange_event_tracker import Models
from tests.test_exchange_event_overlay import build_static, Signals
from tests.test_live_b16 import inputs, m0, remote


@pytest.mark.parametrize('period', [1, 7, 65])
def test_notification_display_equals_offline_at_any_publish_rate(period: int) -> None:
    options = dict(per_side_settled=True, death_guard=True, confirmed_death_hold=True)
    offline = ExchangeEventOverlay(Models(), build_static, Signals, m0, **options)
    live = NotificationExchangeOverlay(Models(), build_static, Signals, m0, **options)
    smoothing = _ExchangeDisplayEMA()
    for tick in range(180):
        args = inputs(tick/30, game=1 if tick < 120 else 2)
        args[0].confirmed_dead_sides = ('2P',) if 80 <= tick < 120 else ()
        offline.update(*args)
        live.update(*args)
        expected = _exchange_display(offline, 0., .5, smoothing, args[3])
        assert live.tracker.probability == offline.tracker.probability
        assert latest_display(live, 0., .5) == expected
        if tick % period == 0:
            before = vars(live.smoothing).copy()
            live.calculate()
            assert latest_display(live, 0., .5) == expected
            assert vars(live.smoothing) == before


def window() -> LiveSnapshotInputs:
    value = object.__new__(LiveSnapshotInputs)
    value.buffers = (deque(maxlen=HISTORY_FRAMES), deque(maxlen=HISTORY_FRAMES))
    value.latest, value.start = [None, None], 0.
    return value


def row(stamp: float, color: int = 1, quality: str = '') -> dict:
    board = np.zeros((13, 6), dtype=int)
    board[-1, :2] = color
    return dict(t_sec=stamp, cnn=board.tolist(), hsv=board.tolist(), quality=quality, glow='')


def test_window_uses_e31_vote_and_first_quality_reset() -> None:
    value = window()
    value.buffers[0].extend(row(i/30, quality='frame_diff' if i == 0 else '') for i in range(36))
    result = value.snapshot(0, 1.2)
    assert result['frames'] == 36 and result['reason'] is None
    assert result['board'] == row(0)['cnn']
    value.buffers[0].clear()
    assert value.snapshot(0, 1.2) == result


def test_missing_frames_break_landed_run() -> None:
    value = window()
    value.buffers[0].extend(row(i/30) for i in (1, 2, 4, 5, 7, 8))
    assert value.snapshot(0, 1.2) == dict(reason='no_landed_window', board=None)


def test_delayed_fire_keeps_three_frame_window_at_oldest_edge() -> None:
    value = window()
    # 実測3270.55秒の事例：通知が1枚遅れ、窓の最古3枚だけが着地している。
    value.buffers[0].extend(row(i/30, quality='' if i < 3 else 'effect_glow')
                            for i in range(WINDOW_FRAMES+1))
    result = value.snapshot(0, 1.2)
    assert result['reason'] is None
    assert result['frames'] == 3
    assert result['start_sec'] == 0.
    assert result['board'] == row(0)['cnn']


def test_future_frames_never_enter_snapshot() -> None:
    value = window()
    value.buffers[0].extend(row(i/30, 1 if i < 36 else 2) for i in range(33, 40))
    assert value.snapshot(0, 1.2)['board'] == row(0)['cnn']


def test_device_timestamp_jitter_uses_each_observation_once() -> None:
    value = window()
    value.buffers[0].extend(row(i/30+.001) for i in range(WINDOW_FRAMES))
    result = value.snapshot(0, 1.2)
    assert result['reason'] is None and result['frames'] == WINDOW_FRAMES
    assert result['board'] == row(0)['cnn']


def test_jitter_matching_excludes_outside_window_and_future() -> None:
    value = window()
    value.buffers[0].extend(row(stamp) for stamp in (-.001, 1.201))
    assert value.snapshot(0, 1.2) == dict(reason='no_landed_window', board=None)


def test_retain_reuses_raw_board_without_cnn(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.board import Board
    value = window()
    region = SimpleNamespace(x=0, y=0, width=20, height=20)
    board = Board()
    board._grid[-1, :2] = 1
    value.reader = SimpleNamespace(_p1_region=region, _p2_region=region,
        read_board=Mock(side_effect=AssertionError('追加CNN禁止')),
        read_board_hsv_only=Mock(side_effect=AssertionError('未使用HSVの追加読取禁止')))
    value.filters = (SimpleNamespace(observe=Mock(return_value=(False, False))),)*2
    value.cost = dict(hsv_sec=0.)
    value.retain(np.zeros((20, 20, 3), dtype=np.uint8), SimpleNamespace(raw_cnn_board=board), 0, 0.)
    assert value.buffers[0][0]['cnn'] is board._grid
    value.reader.read_board.assert_not_called()
    value.reader.read_board_hsv_only.assert_not_called()
    value.reader._live_hsv_boards = {0: board}
    value.reader.read_board_hsv_only.reset_mock()
    value.retain(np.zeros((20, 20, 3), dtype=np.uint8), SimpleNamespace(raw_cnn_board=board), 0, 1/30)
    value.reader.read_board_hsv_only.assert_not_called()


def test_snapshot_epoch_discards_previous_input(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_snapshot import enrich_recognition
    first, second, calibrated = Mock(), Mock(), Mock()
    constructor = Mock(side_effect=[first, second, calibrated])
    monkeypatch.setattr('src.phase_j.live_snapshot.LiveSnapshotInputs', constructor)
    owner = SimpleNamespace(_reader=object())
    pipe = SimpleNamespace(pipe=owner, epoch=0, publishing_ready=False)
    enrich_recognition(pipe, None, None, 0.)
    enrich_recognition(pipe, None, None, 1.)
    pipe.epoch = 1
    enrich_recognition(pipe, None, None, 2.)
    assert constructor.call_count == 2
    assert first.update.call_count == 2 and second.update.call_count == 1
    pipe.publishing_ready = True
    enrich_recognition(pipe, None, None, 3.)
    assert calibrated.update.call_count == 1


def test_worker_replay_restores_previous_match_ema() -> None:
    from src.phase_j.live_eval_worker import execute, make_overlay
    import pickle
    config = (Models(), Signals, m0, True)
    # この入力のstatic生成は親IPCを使うため、同じ純関数を接続する。
    live = make_overlay(Mock(), config)
    live._build_static = build_static
    for tick in range(120):
        live.update(*inputs(tick/30))
    seed = (live.smoothing.adv, live.smoothing.probability, live.smoothing.last_sec)
    replayed = make_overlay(Mock(), config)
    replayed._build_static = build_static
    replayed.restore_smoothing(seed)
    for tick in range(120, 150):
        args = inputs(tick/30, game=2)
        live.update(*args)
        execute(replayed, dict(op='advance', commands=[pickle.dumps(('update', args))]))
        assert replayed.display == live.display


def test_publish_recovery_refreshes_next_match_smoothing(remote: object) -> None:
    from src.phase_j.live_eval_supervisor import EvaluationError
    direct = NotificationExchangeOverlay(Models(), build_static, Signals, m0, per_side_settled=True)
    for tick in range(90):
        args = inputs(tick/30)
        direct.update(*args)
        if tick < 89:
            remote.update(*args)
    # 試合末尾の通知で子が停止しても、公開時のjournal復旧が次試合の初期値になる。
    remote.process.terminate()
    remote.process.join()
    with pytest.raises(EvaluationError):
        remote.update(*args)
    remote.calculate()
    expected = (direct.smoothing.adv, direct.smoothing.probability, direct.smoothing.last_sec)
    assert remote.smoothing == expected
    assert remote.display == direct.display
    remote.update(*inputs(3., game=2))
    assert remote.boundary_smoothing == expected


def test_new_functions_stay_within_project_limit() -> None:
    import ast
    from pathlib import Path
    paths = ('src/phase_j/live_notification_eval.py', 'src/phase_j/live_snapshot.py',
             'scripts/verify_live_b18b.py', 'scripts/summarize_live_b18b.py')
    long = [(path, node.name) for path in paths
            for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8')))
            if isinstance(node, ast.FunctionDef) and node.end_lineno-node.lineno+1 > 50]
    assert long == []


def test_prediction_observation_precedes_floating_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.board import Board
    from src.image_reader import ImageReader, DEFAULT_P1_REGION
    from src.phase_j.live_snapshot import prepare_recognition
    reader = ImageReader(apply_inference=True, use_ui_mask=False)
    frame = np.full((1080, 1920, 3), (0, 0, 200), dtype=np.uint8)
    prepare_recognition(SimpleNamespace(_reader=reader))
    # 認識経路の最終盤面を変えても、観測はその前の分類値を保持する。
    monkeypatch.setattr('src.board_rules.clear_floating_above_gap', lambda *args, **kwargs: Board())
    result = reader.read_board(frame, DEFAULT_P1_REGION, skip_tier1=True)
    assert not np.any(result._grid)
    assert np.any(reader._live_raw_boards[0]._grid)
    result._grid[-1] = 9
    assert 9 not in reader._live_raw_boards[0]._grid
    prepare_recognition(SimpleNamespace(_reader=reader))
    assert reader._live_raw_boards == {}
