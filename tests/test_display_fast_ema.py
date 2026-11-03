"""表示だけのEMA係数切替と、試合境界・欠測・死亡時の契約を確認する。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.b3_live_display import B3Display
from src.exchange_display_smoothing import EMA_ALPHA, FAST_EMA_ALPHA, SwitchAwareDisplayEMA
from src.phase_j.launcher import LauncherConfigError, build_pipeline_config, parse_user_config


@pytest.mark.parametrize('fast, expected', [(False, .35), (True, .5)])
def test_only_display_changes(fast: bool, expected: float) -> None:
    display = B3Display(fast_ema=fast)
    display.boundary(1)
    display.prediction(.2, 0, 1.)
    display.update(1.)
    display.prediction(.8, 0, 2.)
    display.update(2.)
    assert display.probability == display.raw == .8
    assert display.shown[1] == pytest.approx(expected)


@pytest.mark.parametrize('fast', [False, True])
def test_reset_and_death_preserve_flag(fast: bool) -> None:
    display = B3Display(fast_ema=fast)
    display.boundary(1)
    display.prediction(.2, 0, 1.)
    display.update(1.)
    assert display.update(2., 'confirmed_death', 1.) == (100., 1.)
    display.boundary(2)
    display.update(3., active=False)
    assert display.smoother.alpha == (FAST_EMA_ALPHA if fast else EMA_ALPHA)
    display.prediction(.8, 0, 4.)
    assert display.update(4.)[1] == .8


def test_explicit_old_alpha_and_default_remain_compatible() -> None:
    from src.production_config import DISPLAY_FAST_EMA_DEFAULT
    assert DISPLAY_FAST_EMA_DEFAULT is True
    assert B3Display().smoother.alpha == FAST_EMA_ALPHA
    assert SwitchAwareDisplayEMA().alpha == EMA_ALPHA == .25
    assert SwitchAwareDisplayEMA(alpha=.75).alpha == .75
    assert SwitchAwareDisplayEMA(fast_ema=True).alpha == .5


def test_live_display_reset_keeps_fast_setting() -> None:
    from src.phase_j.live_b3 import LiveB3
    live = LiveB3.__new__(LiveB3)
    live.fast_ema = True
    assert live._new_display().smoother.alpha == .5


@pytest.mark.parametrize('value', ['true', 1, None, True])
def test_launcher_rejects_invalid_or_non_b3_flag(tmp_path: Path, value: object) -> None:
    if value is True:
        assert parse_user_config(dict(device_name='OBS', display_fast_ema=value),
                                 tmp_path).display_fast_ema is False
        return
    with pytest.raises(LauncherConfigError, match='display_fast_ema'):
        parse_user_config(dict(device_name='OBS', display_fast_ema=value), tmp_path)


def test_flag_is_independent_of_transport(tmp_path: Path) -> None:
    (tmp_path / 'model.joblib').touch()
    (tmp_path / 'timing.json').touch()
    user = parse_user_config(dict(device_name='OBS', b3_judgment=True,
        b3_model='model.joblib', b3_timing_table='timing.json', display_fast_ema=True,
        display_completion_driven=False), tmp_path)
    config = build_pipeline_config(user)
    assert config['display_fast_ema'] is True
    assert config['display_completion_driven'] is False
