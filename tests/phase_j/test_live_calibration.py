"""色ウォームアップの収束・保存・入力境界・勝率抑止を検査する。"""
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock
import json

import pytest

from src.phase_j.live_calibration import ColorWarmup, MIN_SAMPLES
from src.phase_j.live_device import DeviceConfig
from src.phase_j.live_device_session import DeviceSession, ConfiguredSource, InputChanged
from src.phase_j.live_publish import LivePublisher, initial_snapshot, result_snapshot
from src.phase_j.live_process import ProcessRecognitionBridge
from src.phase_j.validator import validate_snapshot
from tests.phase_j.test_live_publish import ASSETS, row


def setup_warmup(tmp_path: Path, verification: bool = True) -> tuple:
    device = SimpleNamespace(name='OBS', index=0, calibration_path=tmp_path / 'device.json')
    classifier = SimpleNamespace(_hsv=Mock())
    return ColorWarmup(device, verification), classifier, device


def samples(warmup: ColorWarmup, count: int, colors: tuple = (1, 2, 3, 4), shift: int = 0) -> None:
    for color in colors:
        for _ in range(count):
            warmup.calibrator._stats[color].update(color*20+shift, 200, 200)


@pytest.mark.parametrize('count', [0, 1, MIN_SAMPLES-1])
def test_insufficient_samples_never_ready(tmp_path: Path, count: int) -> None:
    warmup, classifier, device = setup_warmup(tmp_path)
    samples(warmup, count)
    warmup.advance(classifier)
    assert not warmup.ready and warmup.progress < 100
    assert not device.calibration_path.exists()


def test_requires_four_actual_colors(tmp_path: Path) -> None:
    warmup, classifier, _ = setup_warmup(tmp_path)
    samples(warmup, MIN_SAMPLES, (1, 2, 3, 9))
    warmup.advance(classifier)
    assert not warmup.ready
    samples(warmup, MIN_SAMPLES, (5,))
    warmup.advance(classifier)
    assert warmup.ready


@pytest.mark.parametrize('verification', [False, True])
def test_complete_save_short_recheck(tmp_path: Path, verification: bool) -> None:
    warmup, classifier, device = setup_warmup(tmp_path, verification)
    samples(warmup, MIN_SAMPLES)
    assert warmup.advance(classifier)['phase'] == 'ready'
    assert classifier._hsv.set_color_ranges_from_simple.call_count == int(not verification)
    recheck = ColorWarmup(device, verification)
    assert recheck.cached and not recheck.ready and recheck.target == 50
    samples(recheck, recheck.target)
    assert recheck.advance(classifier)['phase'] == 'ready'


def test_cached_drift_falls_back_to_full_collection(tmp_path: Path) -> None:
    warmup, classifier, device = setup_warmup(tmp_path)
    samples(warmup, MIN_SAMPLES)
    warmup.advance(classifier)
    recheck = ColorWarmup(device, True)
    samples(recheck, recheck.target, shift=10)
    recheck.advance(classifier)
    assert not recheck.cached and recheck.changed_profile and not recheck.ready
    assert recheck.target == MIN_SAMPLES
    samples(recheck, MIN_SAMPLES, shift=10)
    assert recheck.advance(classifier)['phase'] == 'ready'


@pytest.mark.parametrize('corruption', ['identity', 'nan', 'bounds', 'json'])
def test_invalid_profile_is_not_trusted(tmp_path: Path, corruption: str) -> None:
    warmup, classifier, device = setup_warmup(tmp_path)
    samples(warmup, MIN_SAMPLES)
    warmup.advance(classifier)
    data = json.loads(device.calibration_path.read_text())
    if corruption == 'identity':
        data['device']['index'] = 1
    elif corruption == 'nan':
        data['ranges']['1'][0] = float('nan')
    elif corruption == 'bounds':
        data['ranges']['1'][0] = -1
    device.calibration_path.write_text('!' if corruption == 'json' else json.dumps(data))
    assert not ColorWarmup(device).cached


def test_no_screen_cannot_complete_cached_profile(tmp_path: Path) -> None:
    warmup, classifier, device = setup_warmup(tmp_path)
    samples(warmup, MIN_SAMPLES)
    warmup.advance(classifier)
    recheck = ColorWarmup(device)
    status = recheck.observe(None, SimpleNamespace(is_match_active=False), classifier)
    assert status['phase'] == 'no_puyo_screen' and not recheck.ready


@pytest.mark.parametrize('phase,message', [('verifying', '入力確認中'),
    ('no_puyo_screen', 'ぷよ画面なし'), ('calibrating', '色を較正中 42%')])
def test_warmup_removes_all_old_probabilities(phase: str, message: str) -> None:
    data = dict(row(), input_calibration=dict(phase=phase, progress=42),
                display_layers={'displayed_p1': .6}, counter_search={'pending': False})
    snapshot = result_snapshot(initial_snapshot(ASSETS), data, 2, 1, 1)
    assert validate_snapshot(snapshot).is_valid
    assert snapshot.display['message'] == message
    assert snapshot.evaluations['practical']['p1_win_probability'] is None
    assert 'display_layers' not in snapshot.evaluations and 'counter_search' not in snapshot.evaluations


@pytest.mark.parametrize('phase,progress', [('calibrating', 42), ('ready', 99), ('invalid', 0)])
def test_validator_rejects_inconsistent_warmup(phase: str, progress: int) -> None:
    data = result_snapshot(initial_snapshot(ASSETS), row(), 2, 1, None).to_mapping()
    data['display'].update(input_status=phase, calibration_progress=progress)
    assert not validate_snapshot(data).is_valid


def test_late_offer_cannot_resurrect_probability() -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0)
    publisher.offer(row())
    publisher.input_pending(dict(phase='calibrating', progress=20, at=2))
    publisher.offer(row())
    snapshot = result_snapshot(publisher.initial, publisher.latest, 3, 1, 2)
    assert snapshot.evaluations['practical']['p1_win_probability'] is None
    publisher.input_pending(dict(phase='ready', progress=100, at=3))
    publisher.offer(row())
    assert result_snapshot(publisher.initial, publisher.latest, 4, 2, None).display['input_status'] == 'ready'


def test_switch_and_signal_loss_reset_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.phase_j.live_device.CALIBRATION_ROOT', tmp_path)
    hsv = SimpleNamespace(_ranges={'base': 1}, _native_ranges_cache=1, _native_params_cache=1)
    pipe = SimpleNamespace(reset=Mock(), _online_hsv=None,
                           _reader=SimpleNamespace(_classifier=SimpleNamespace(_hsv=hsv)))
    session = DeviceSession(pipe, tmp_path / 'input.json', 1, Queue())
    session.switch(DeviceConfig('A', 0))
    hsv._ranges = {'changed': 2}
    session.switch(DeviceConfig('B', 1))
    assert hsv._ranges == {'base': 1} and session.epoch == 1
    session.warmup.ready = True
    session.hold('no_puyo_screen')
    assert not session.publishing_ready and not session.warmup.ready and session.epoch == 2
    assert pipe.reset.call_count == 3 and pipe._online_hsv is None
    session.warmup.progress = 42
    session.hold('verifying')
    assert session.epoch == 3 and session.warmup.progress == 0
    session.hold('verifying')
    assert session.epoch == 3


def test_config_change_detected_even_during_hold(tmp_path: Path) -> None:
    path = tmp_path / 'input.json'
    path.write_text('{"name":"B","index":1}')
    source = ConfiguredSource(path, 1, Mock(), Mock())
    with pytest.raises(InputChanged):
        source.check(DeviceConfig('A', 0))


def test_input_epoch_consumed_once() -> None:
    bridge = ProcessRecognitionBridge(False, Mock(), Path('unused'), device=Path('input.json'))
    bridge.on_hold = Mock()
    bridge._message(('hold', dict(epoch=0, phase='verifying')))
    assert not bridge.consume_input_boundary()
    bridge._message(('hold', dict(epoch=1, phase='verifying')))
    bridge._message(('hold', dict(epoch=1, phase='calibrating')))
    assert bridge.consume_input_boundary() and not bridge.consume_input_boundary()
    bridge.queue.close()


def test_publisher_rechecks_gate_at_actual_publication() -> None:
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0)
    old = row()
    publisher.input_pending(dict(phase='calibrating', progress=20, at=2))
    publisher._publish(old, 1, None)
    data = publisher.state.subscribe().get()
    assert data['evaluations']['practical']['p1_win_probability'] is None
    assert data['display']['input_status'] == 'calibrating'


def test_stage_two_parameters_are_inherited_without_samples(tmp_path: Path) -> None:
    from src.online_hsv_calibrator import OnlineHsvCalibrator
    _, classifier, device = setup_warmup(tmp_path)
    template = OnlineHsvCalibrator(high_conf=.85, min_samples=50, require_cnn_proba=False)
    template._stats[1].update(10, 200, 200)
    warmup = ColorWarmup(device, True, template)
    assert warmup.target == 50 and not warmup.calibrator._require_cnn_proba
    assert warmup.calibrator.get_sample_counts()[1] == 0
    samples(warmup, 50)
    warmup.advance(classifier)
    assert ColorWarmup(device, True, template).target == 12


def test_device_queue_preserves_hold_notice_order(tmp_path: Path) -> None:
    from tests.phase_j.test_live_process import notice
    from src.phase_j.live_process import NoticeDeltaCodec
    bridge = ProcessRecognitionBridge(False, Mock(), tmp_path, device=tmp_path)
    bridge.queue.close()
    bridge.queue = Queue()
    bridge.on_hold = Mock()
    encoder = NoticeDeltaCodec()
    bridge.queue.put(('notice', encoder.encode(notice(1))))
    bridge.queue.put(('hold', dict(phase='verifying', epoch=1)))
    bridge.queue.put(('notice', encoder.encode(notice(2))))
    assert [n.frame for n in bridge._receive_batch()] == [1]
    bridge.on_hold.assert_not_called()
    assert bridge._receive_batch() == []
    bridge.on_hold.assert_called_once()
    assert [n.frame for n in bridge._receive_batch()] == [2]


@pytest.mark.parametrize('correction', [True, False])
def test_ready_transition_resets_only_corrected_history(tmp_path: Path, correction: bool) -> None:
    hsv = SimpleNamespace(_ranges={}, _native_ranges_cache=None, _native_params_cache=None)
    pipe = SimpleNamespace(reset=Mock(), update=Mock(return_value='result'), _online_hsv=None,
                           _reader=SimpleNamespace(_classifier=SimpleNamespace(_hsv=hsv)))
    session = DeviceSession(pipe, tmp_path, 1, Queue())
    warmup = SimpleNamespace(ready=False, verification_only=not correction,
                            calibrator=SimpleNamespace(is_ready=lambda color: color != 1))
    def observe(*args: object) -> dict:
        warmup.ready = True
        return dict(phase='ready', progress=100)
    warmup.observe = observe
    session.warmup = warmup
    assert session.update(1, 0, None) == 'result'
    assert session.publishing_ready != correction and pipe.reset.call_count == int(correction)
    assert pipe._online_hsv_injected_colors == {2, 3, 4, 5}
