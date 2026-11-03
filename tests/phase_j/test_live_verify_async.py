"""待たない検証の設定・遅延・後始末を実機なしで確認する。"""
import argparse
import json
from pathlib import Path
from threading import Event, enumerate as threads, get_ident
from typing import Any

import pytest

from src.phase_j import live_speed, live_verify_reasons as reasons
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from src.phase_j.live_screen_worker import AsyncScreenState
from tests.phase_j.test_live_screen_worker import IMAGE, SIZE, completed
from tests.phase_j.test_live_fast_first import session

WAIT_SEC = 2


@pytest.mark.parametrize('enabled', [False, True])
def test_source_switch_and_keyboard_interrupt_cleanup(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from tests.phase_j.test_live_device import FakeDevice
    monkeypatch.setenv('PUYO_LIVE_VERIFY_ASYNC', str(int(enabled)))
    monkeypatch.delenv('PUYO_LIVE_ASYNC_SCREEN_VERIFY', raising=False)
    capture = FakeDevice(np.zeros((1080, 1920, 3), dtype=np.uint8))
    called = Event()
    owners: list[int] = []
    def verifier(image: Any) -> bool:
        owners.append(get_ident())
        called.set()
        return True
    def interrupt(seconds: float) -> None:
        assert called.wait(WAIT_SEC)
        raise KeyboardInterrupt
    source = DirectShowSource(DeviceConfig('OBS', 0), WAIT_SEC, lambda: None,
                              lambda *args: capture, verifier, sleep=interrupt)
    with pytest.raises(KeyboardInterrupt):
        list(source)
    assert capture.released
    assert (owners[0] != get_ident()) == enabled
    assert not any(thread.name.startswith('screen-verify') for thread in threads())


@pytest.mark.parametrize('initial', [True, False])
def test_no_wait_even_on_first_or_held_frame(initial: bool) -> None:
    entered, release = Event(), Event()
    images: list[Any] = []
    def slow(image: Any) -> bool:
        entered.set()
        assert release.wait(WAIT_SEC)
        images.append(image.copy())
        return True
    source = DirectShowSource(DeviceConfig('OBS', 0), 1, lambda: None)
    state = AsyncScreenState(slow, source._judge, 1, wait_initial=False)
    state.fail_streak = 0 if initial else 3
    image = IMAGE.copy()
    try:
        assert state.update(image, SIZE, 0, True) == (False, state.fail_streak)
        assert entered.wait(WAIT_SEC)
        image[:] = 255
        assert state.update(image, SIZE, .1, True)[0] is False
        release.set()
        state.worker.future.result(timeout=WAIT_SEC)
        assert state.update(image, SIZE, .2, True) == (True, 0)
        assert state.application_delay_sec == .2
        assert not images[0].any()
    finally:
        release.set()
        state.close()
    assert all(not thread.is_alive() for thread in state.worker.pool._threads)


def test_three_failures_hold_reset_once_and_pass_recovers(session: Any) -> None:
    source = DirectShowSource(DeviceConfig('OBS', 0, verify_fail_streak=3), 1, lambda: None)
    state = AsyncScreenState(lambda _: True, source._judge, 1, wait_initial=False)
    state.last_verify, state.verified = 0, True
    resets = session.pipe.reset.call_count
    try:
        for count in range(1, 4):
            completed(state, False)
            assert state.update(IMAGE, SIZE, count/10, True) == (count < 3, count)
        session.hold('no_puyo_screen')
        session.hold('no_puyo_screen')
        assert session.pipe.reset.call_count == resets+1
        completed(state, True)
        assert state.update(IMAGE, SIZE, .4, True) == (True, 0)
        session.update(4, .4, None)
        assert session.publishing_ready
    finally:
        state.close()


def test_worker_preserves_reason_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(reasons.FLAG_ENV, '1')
    monkeypatch.setenv(reasons.OUTPUT_ENV, str(tmp_path))
    def verifier(image: Any) -> bool:
        reasons.detail('feedback', passed=False, value=True)
        return False
    state = AsyncScreenState(verifier, lambda p, v, s: (p, s+1), 1, wait_initial=False)
    try:
        state.update(IMAGE, SIZE, 0, True)
        state.worker.future.result(timeout=WAIT_SEC)
        assert state.update(IMAGE, SIZE, .1, True) == (False, 1)
        row = json.loads((tmp_path/reasons.LOG_FILE).read_text(encoding='utf-8'))
        assert row['failed_conditions'] == ['feedback']
        assert row['fail_streak'] == 1 and not row['passed']
    finally:
        state.close()


def test_flag_default_cli_launcher_and_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.launcher import UserConfig, build_pipeline_config, parse_user_config
    assert not live_speed.SPEED_DEFAULTS['live_verify_async']
    assert not UserConfig.__dataclass_fields__['live_verify_async'].default
    user = parse_user_config(dict(source='dshow', device_name='OBS', live_verify_async=True), tmp_path)
    assert build_pipeline_config(user)['live_verify_async']
    parser = argparse.ArgumentParser()
    live_speed.add_arguments(parser)
    options = parser.parse_args(['--live-verify-async'])
    options.output, options.source, options.cpu_threads = tmp_path, 'dshow', 1
    monkeypatch.setattr('os.environ', {})
    live_speed.configure(options, parser)
    row = json.loads((tmp_path/live_speed.TRIAL_FLAGS_FILE).read_text())
    keys = list(row['flags'])
    assert keys == list(live_speed.TRIAL_FLAG_KEYS) and keys[keys.index('live_verify_async')-1] == 'live_native_hsv_main'   # 後ろに別の鍵が続いてよい
    assert row['flags']['live_verify_async'] and row['env']['PUYO_LIVE_VERIFY_ASYNC'] == '1'
    assert row['arm'] == 'OFF'
