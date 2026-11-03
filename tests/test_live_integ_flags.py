"""統合版 (色補正buffer + ONNX非スピン + 差分CNN) の既定OFFと配線が両立することを確かめる。"""
from __future__ import annotations

import dataclasses

import pytest

from src import production_config
from src.phase_j import launcher, live_speed

CANDIDATE_FLAGS = ('live_color_buffer', 'live_onnx_parallel', 'live_delta_cells', 'live_cnn_dedup')
PRESERVED_FLAGS = ('live_public_match_number', 'live_exact_native', 'live_near_future_cache')


def test_candidate_flags_default_off_and_single_source() -> None:
    for name in CANDIDATE_FLAGS:
        assert getattr(production_config, name.upper() + '_DEFAULT') is False
        assert live_speed.SPEED_DEFAULTS[name] is False
        assert live_speed.validate({})[name] is False


def test_user_config_has_candidate_and_preserved_fields() -> None:
    fields = {item.name for item in dataclasses.fields(launcher.UserConfig)}
    assert set(CANDIDATE_FLAGS + PRESERVED_FLAGS) <= fields
    assert getattr(launcher.UserConfig, 'live_delta_cells') is False


@pytest.mark.parametrize('name', CANDIDATE_FLAGS)
def test_each_candidate_flag_turns_on_independently(name: str) -> None:
    values = live_speed.validate({name: True})
    assert values[name] is True
    assert all(values[other] is False for other in CANDIDATE_FLAGS if other != name)


def test_non_bool_candidate_flag_is_rejected() -> None:
    with pytest.raises(ValueError):
        live_speed.validate({'live_delta_cells': 1})
