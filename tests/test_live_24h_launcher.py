"""利用者JSONから子パイプラインまで、起動時の選択を失わずに届ける。"""
from pathlib import Path

import pytest

from src.phase_j.launcher import LauncherConfigError, build_pipeline_config, parse_user_config


def data(tmp_path: Path, **changes: object) -> dict:
    for name in ('base.joblib', 'timing.json', 'ex.json'):
        (tmp_path / name).write_text('fixture', encoding='utf-8')
    value = dict(device_name='OBS Virtual Camera', b3_judgment=True, b3_async=True,
                 b3_m3=True, b3_c2=True, b3_skip_icon_reading=True,
                 b3_model='base.joblib', b3_timing_table='timing.json',
                 live_v49f=True, live_ama_plain=True, live_winrate_switch=True)
    return dict(value, **changes)


def test_json_on_resolves_model_and_disables_legacy_helpers(tmp_path: Path) -> None:
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_model='ex.json',
                                  live_24h_delay_ms=300.), tmp_path)
    assert user.live_24h_ama and user.live_24h_model == tmp_path / 'ex.json'
    assert not user.live_v49f and not user.live_ama_plain and not user.live_winrate_switch
    result = build_pipeline_config(user)
    assert result['live_24h'] and result['live_24h_ama']
    assert result['live_24h_delay_ms'] == 300.
    assert result['live_24h_model'] == str(tmp_path / 'ex.json')


def test_ama_off_needs_only_the_existing_c2_assets(tmp_path: Path) -> None:
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_ama=False), tmp_path)
    result = build_pipeline_config(user)
    assert result['live_24h'] and result['live_24h_ama'] is False
    assert 'live_24h_model' not in result and user.live_24h_model is None


def test_off_writes_exactly_the_same_pipeline_configuration(tmp_path: Path) -> None:
    original = build_pipeline_config(parse_user_config(data(tmp_path), tmp_path))
    off = build_pipeline_config(parse_user_config(data(tmp_path, live_24h=False), tmp_path))
    assert off == original and not any(key.startswith('live_24h') for key in off)
    assert off['live_v49f'] and off['live_ama_plain'] and off['live_winrate_switch']


@pytest.mark.parametrize(('change', 'needle'), [
    (dict(live_24h='yes'), 'live_24h'),
    (dict(live_24h_ama='yes'), 'live_24h_ama'),
    (dict(live_24h_delay_ms=-1), 'live_24h_delay_ms'),
    (dict(live_24h_delay_ms=float('inf')), 'live_24h_delay_ms'),
    (dict(live_24h_model='absent.json'), 'live_24h_model'),
    (dict(b3_c2=False), 'live_24h'),
])
def test_invalid_unified_configuration_is_actionable(tmp_path: Path, change: dict, needle: str) -> None:
    values = dict(live_24h=True, live_24h_model='ex.json')
    values.update(change)
    with pytest.raises(LauncherConfigError, match=needle):
        parse_user_config(data(tmp_path, **values), tmp_path)
