"""利用者の表示設定を起動前に検証する。"""
import pytest
import argparse
import json
from pathlib import Path

from src.phase_j.overlay_config import configured_html, validate_overlay


@pytest.mark.parametrize('value', [None, [], {'unknown': True}, {'scale': True},
    {'scale': 69}, {'scale': 86}, {'scale': float('nan')}, {'font': 'huge'},
    {'corner': 'middle'}, {'numbers': 'false'}, {'layout': 'unknown'}])
def test_reject_invalid(value: object) -> None:
    with pytest.raises(ValueError, match='overlay'):
        validate_overlay(value)


def test_partial_and_independent_options() -> None:
    data = dict(layout='l-left', corner='bottom-right', scale=77.5, numbers=False)
    assert validate_overlay(data) == data
    assert validate_overlay({}) == {}


def test_script_data_is_escaped() -> None:
    result = configured_html('/*OVERLAY_DEFAULTS*/{}', {'x': '</script>'})
    assert b'</script>' not in result


def test_launcher_to_pipeline_keeps_options(tmp_path: Path) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    from src.phase_j.live_config import _file_defaults
    overlay = dict(layout='l-left', scale=75, corner='bottom-right', band=False,
                   numbers=False, recent=True, full=True, status=False, details=True, font='large')
    user = parse_user_config(dict(device_name='OBS Virtual Camera', overlay=overlay), tmp_path)
    pipeline = build_pipeline_config(user)
    assert pipeline['overlay'] == overlay
    path = tmp_path / 'pipeline.json'
    path.write_text(json.dumps(dict(overlay=pipeline['overlay'])), encoding='utf-8')
    assert _file_defaults(argparse.ArgumentParser(), path)['overlay'] == overlay


def test_launcher_invalid_overlay_is_configuration_error(tmp_path: Path) -> None:
    from src.phase_j.launcher import parse_user_config, LauncherConfigError
    with pytest.raises(LauncherConfigError, match='overlay'):
        parse_user_config(dict(device_name='OBS Virtual Camera', overlay={'scale': 86}), tmp_path)
