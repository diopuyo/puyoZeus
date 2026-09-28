"""片側MCと起動から次試合までのライブ状態遷移を検査する。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json

import numpy as np
import pytest

from tests.test_live_counter import Executor, board
from tests.phase_j.test_live_publish import ASSETS, row
from src.phase_j.live_side_counter import SideCounter
from src.phase_j.live_bridge import RecognitionBridge
from src.phase_j.live_publish import LivePublisher, result_snapshot
from src.phase_j.live_layers import evaluation_layers
from src.phase_j.validator import validate_snapshot
from scripts.measure_live_b6 import wait_idle, command
from scripts.run_live_pipeline_20260928 import parse_args


def finish(executor: Executor, index: int, probability: float = .8) -> None:
    future, args = executor.jobs[index]
    future.set_result((args['generation'], (0., probability, float('nan')), 2.))


@pytest.mark.parametrize('changed_side', [0, 1])
def test_opponent_update_keeps_own_completed_result(changed_side: int) -> None:
    executor, boards = Executor(), [board(), board()]
    counter = SideCounter(executor)
    counter.invalidate((1, True, *[b.grid_bytes() for b in boards]))
    counter.update(*boards, 3, t_sec=1)
    finish(executor, 0)
    finish(executor, 1)
    boards[changed_side] = board(1)
    counter.invalidate((1, True, *[b.grid_bytes() for b in boards]))
    counter.update(*boards, 2, t_sec=2)
    assert counter.lanes[changed_side].discarded == 1
    assert counter.lanes[1-changed_side].accepted == 1
    assert counter.status()['sides'][('1P', '2P')[1-changed_side]]['probability'] == .8


@pytest.mark.parametrize('rollouts', [1, 15, 30, 60])
def test_rollouts_are_sent_to_each_independent_job(rollouts: int) -> None:
    executor = Executor()
    counter = SideCounter(executor, rollouts)
    counter.update(board(), board(), 3, t_sec=1)
    assert len(executor.jobs) == 2
    assert all(args['rollouts'] == rollouts and args['defender_side'] == '1P'
               for _, args in executor.jobs)


@pytest.mark.parametrize('rollouts', [0, -1, True, 1.5])
def test_invalid_rollouts_rejected(rollouts: object) -> None:
    with pytest.raises(ValueError):
        SideCounter(Executor(), rollouts)


@pytest.mark.parametrize('boundary', ['game', 'inactive', 'aba'])
def test_own_generation_boundaries_discard(boundary: str) -> None:
    executor, counter = Executor(), None
    counter = SideCounter(executor)
    counter.invalidate((1, True, b'a', b'b'))
    counter.update(board(), board(), 3, t_sec=1)
    counter.invalidate((2 if boundary == 'game' else 1, boundary != 'inactive', b'c', b'b'))
    if boundary == 'aba':
        counter.invalidate((1, True, b'a', b'b'))
    finish(executor, 0)
    counter.update(board(), board(), 2, t_sec=2)
    assert counter.lanes[0].discarded == 1


def test_partial_result_is_not_combined_with_missing_opponent() -> None:
    executor = Executor()
    counter = SideCounter(executor)
    counter.update(board(), board(), 3, t_sec=1)
    finish(executor, 0)
    adv, p1, p2 = counter.update(board(), board(), 2, t_sec=1.1)
    assert adv == 0 and p1 == .8 and np.isnan(p2) and counter.pending and not counter.contributes
    finish(executor, 1, .2)
    adv, p1, p2 = counter.update(board(), board(), 2, t_sec=1.2)
    assert adv == pytest.approx(24) and not counter.pending and counter.contributes


@pytest.mark.parametrize('side,index', [('1P', 0), ('2P', 1)])
def test_defender_only_does_not_require_opponent(side: str, index: int) -> None:
    executor = Executor()
    counter = SideCounter(executor)
    counter.update(board(), board(), 3, t_sec=1, defender_side=side)
    assert len(executor.jobs) == 1
    finish(executor, 0)
    result = counter.update(board(), board(), 2, t_sec=2, defender_side=side)
    assert result[index+1] == .8 and not counter.pending


def test_only_own_next_changes_generation() -> None:
    executor = Executor()
    counter = SideCounter(executor)
    counter.update(board(), board(), 3, next1=(1, 2), next2=(3, 4), t_sec=1)
    finish(executor, 0)
    finish(executor, 1)
    counter.update(board(), board(), 2, next1=(1, 2), next2=(4, 5), t_sec=2)
    assert counter.lanes[0].accepted == 1 and counter.lanes[1].discarded == 1


def test_expiry_and_old_value_pending_are_preserved() -> None:
    executor = Executor()
    counter = SideCounter(executor)
    counter.update(board(), board(), 1, t_sec=1)
    finish(executor, 0)
    finish(executor, 1)
    counter.update(board(), board(), 1, t_sec=3)
    assert counter.status()['discarded'] == 2 and counter.pending


def test_offline_rollouts_remain_sixty_after_live_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.visualize_advantage_overlay as legacy
    from src.phase_j.live_counter import search
    seen = []
    class Tracker:
        last_hands = 2
        def update(self, **kwargs: object) -> tuple:
            seen.append(legacy.COUNTER_N_ROLLOUTS)
            return (0., .5, .5)
    monkeypatch.setattr(legacy, 'CounterReachTracker', Tracker)
    search(dict(generation=1, rollouts=30))
    assert seen == [30] and legacy.COUNTER_N_ROLLOUTS == 60


@pytest.mark.parametrize('probability,included', [(None, True), (.6, False)])
def test_mc_source_only_when_display_uses_legacy_value(probability: float | None, included: bool) -> None:
    bridge = RecognitionBridge(False, Mock())
    overlay = SimpleNamespace(tracker=SimpleNamespace(probability=probability, source='G_fe', _static_probability=None))
    counter = SimpleNamespace(_last_result=(10, .8, .2), last_budget_sec=2)
    assert bridge.mc_in_display(overlay, counter, None, False) == included
    layers = evaluation_layers(overlay, .6, included)
    assert layers['includes_prediction'] == included


def test_lifecycle_until_next_match_uses_same_publisher() -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0)
    for revision, phase in enumerate(('verifying', 'no_puyo_screen', 'calibrating'), 1):
        publisher.input_pending(dict(phase=phase, progress=42, at=1))
        snap = result_snapshot(publisher.initial, publisher.latest, 2, revision, 1)
        assert validate_snapshot(snap).is_valid and snap.evaluations['practical']['p1_win_probability'] is None
    publisher.input_pending(dict(phase='ready', progress=100, at=2))
    for game, active in ((1, True), (1, False), (2, True)):
        publisher.offer(dict(row(), game=game, match_active=active))
        snap = result_snapshot(publisher.initial, publisher.latest, 3, game+4, None)
        assert validate_snapshot(snap).is_valid, validate_snapshot(snap)
        assert (snap.evaluations['practical']['availability'] == 'available') == active
        assert snap.identity['match_id'] == str(game)


def parse(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> object:
    monkeypatch.setattr('sys.argv', ['live', *args])
    return parse_args()


def test_config_cli_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path/'config.json'
    path.write_text(json.dumps(dict(source='video', mc_rollouts=15, realtime=True)))
    options = parse(monkeypatch, ['--config', str(path), '--mc-rollouts', '30'])
    assert options.source == 'video' and options.mc_rollouts == 30 and options.lifecycle
    assert parse(monkeypatch, ['--source=video']).lifecycle


def test_dshow_config_is_shared_with_input_watcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path/'config.json'
    path.write_text('{"name":"OBS","index":0}')
    options = parse(monkeypatch, ['--source', 'dshow', '--config', str(path)])
    assert options.input_config == path and options.realtime and options.end_sec == 86400


@pytest.mark.parametrize('args', [['--source', 'dshow'], ['--mc-rollouts', '0'],
    ['--compare', '--realtime'], ['--source', 'dshow', '--worker-mode', 'thread']])
def test_invalid_cli_rejected(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse(monkeypatch, args)


def test_cpu_selected_before_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    parse(monkeypatch, ['--source', 'video', '--cnn-device', 'cpu'])
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''


@pytest.mark.parametrize('loads,deadline,expected', [([0.5]*20, 70, True),
    ([1.0]*20, 70, False), ([0.1]*8+[2]+[.1]*20, 110, True), ([.1]*8+[2]+[.1]*20, 70, False)])
def test_idle_requires_continuous_minute(tmp_path: Path, loads: list, deadline: float, expected: bool) -> None:
    clock = [0.0]
    def sleep(seconds: float) -> None:
        clock[0] += seconds
    values = iter(loads)
    assert wait_idle(deadline, tmp_path, 'gpu', lambda: next(values), lambda: clock[0], sleep) == expected
    if not expected:
        assert json.loads((tmp_path/'status.json').read_text())['state'] == 'skipped'


@pytest.mark.parametrize('condition,device', [('gpu', 'auto'), ('cpu', 'cpu')])
def test_measurement_uses_live_realtime_120_seconds(condition: str, device: str) -> None:
    args = command(condition, Path('logs/test'))
    assert args[args.index('--source')+1] == 'video' and '--realtime' in args
    assert args[args.index('--cnn-device')+1] == device
    assert args[args.index('--start-sec')+1] == '2580' and args[args.index('--end-sec')+1] == '2700'


def test_fixed_video_keeps_recognition_across_game_effects() -> None:
    from src.phase_j.live_video_session import VerifiedVideo
    frames = [SimpleNamespace(media_sec=t, image=np.full((18, 32, 3), t, np.uint8)) for t in range(4)]
    verifier, session = Mock(side_effect=[False, True, False]), SimpleNamespace(hold=Mock())
    source = VerifiedVideo(frames, session, verifier)
    assert list(source) == frames[1:]
    assert verifier.call_count == 2 and source.gated_frames == 1


@pytest.mark.parametrize('remove', ['source', 'prediction_sources', 'display_layers'])
def test_mc_provenance_is_required(remove: str) -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0)
    data = dict(row(), source='legacy_mc', prediction_sources=['mc_counter'],
                display_layers=dict(current_p1=None, predicted_p1=.6, displayed_p1=.6,
                                    includes_prediction=True, prediction_discard_reason=None))
    snapshot = result_snapshot(publisher.initial, data, 3, 1, None).to_mapping()
    assert validate_snapshot(snapshot).is_valid
    if remove == 'source':
        snapshot['evaluations']['practical']['source'] = 'G_fe'
    else:
        del snapshot['evaluations'][remove]
    assert not validate_snapshot(snapshot).is_valid


def test_side_dto_rejects_stale_result_without_pending() -> None:
    executor = Executor()
    counter = SideCounter(executor)
    counter.update(board(), board(), 3, t_sec=1)
    finish(executor, 0)
    counter.update(board(), board(), 2, t_sec=1.1)
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0)
    data = dict(row(), counter_search=counter.status())
    snapshot = result_snapshot(publisher.initial, data, 3, 1, None).to_mapping()
    assert validate_snapshot(snapshot).is_valid
    snapshot['evaluations']['counter_search']['sides']['1P'].update(generation=99, pending=False)
    assert not validate_snapshot(snapshot).is_valid
