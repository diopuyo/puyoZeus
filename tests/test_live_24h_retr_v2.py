"""retR v2の独立した旗をCLI・JSON・親子設定の全経路で検査する。"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j import live_retr_v2_config as v2
from src.phase_j.live_ex_async import _compute


@pytest.mark.parametrize('name,env,child_env', v2.FLAGS)
@pytest.mark.parametrize('active', [False, True])
def test_cli_and_child(monkeypatch: pytest.MonkeyPatch, name: str, env: str,
                       child_env: str, active: bool) -> None:
    from src.phase_j import live_ex_runtime
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    key = 'live_24h_' + name
    assert getattr(parser.parse_args([]), key) is False
    flag = ('--' if active else '--no-') + key.replace('_', '-')
    assert getattr(parser.parse_args([flag]), key) is active
    for _, parent, child in v2.FLAGS:
        monkeypatch.setenv(parent, '1')
        monkeypatch.setenv(child, '1')
    options = SimpleNamespace(live_24h=True, live_24h_ama=False, live_24h_retr=True,
                              b3_judgment=True, b3_async=True, b3_m3=True, b3_c2=True,
                              **{key: active})
    config.configure(options, parser)
    assert os.environ[env] == str(int(active))
    monkeypatch.setenv(config.ENV_AMA, '1')
    settings = config.ex_settings()
    assert settings[name] is active
    other = next(row for row in v2.FLAGS if row[0] != name)
    assert settings[other[0]] is False
    seen = []
    connection = SimpleNamespace(recv=lambda: seen.append(os.environ[child_env]), close=lambda: None)
    _compute(connection, settings)
    assert seen == [str(int(active))]
    assert os.environ[other[2]] == '0'


@pytest.mark.parametrize('name,env,child_env', v2.FLAGS)
@pytest.mark.parametrize('active', [False, True])
def test_json_pipeline(tmp_path: Path, name: str, env: str, child_env: str, active: bool) -> None:
    from src.phase_j.launcher import build_pipeline_config, parse_user_config
    from tests.test_live_24h_launcher import data
    key = 'live_24h_' + name
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_retr=True,
                                  live_24h_model='ex.json', **{key: active}), tmp_path)
    assert getattr(user, key) is active
    assert build_pipeline_config(user)[key] is active


@pytest.mark.parametrize('name,env,child_env', v2.FLAGS)
@pytest.mark.parametrize('value,retr', [(1, True), ('true', True), (None, True), (True, False)])
def test_invalid_json_and_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                name: str, env: str, child_env: str, value: object, retr: bool) -> None:
    from src.phase_j.launcher import parse_user_config
    from tests.test_live_24h_launcher import data
    key = 'live_24h_' + name
    with pytest.raises(ValueError, match=key):
        parse_user_config(data(tmp_path, live_24h=True, live_24h_retr=retr,
                               live_24h_model='ex.json', **{key: value}), tmp_path)
    for _, _, child in v2.FLAGS:
        monkeypatch.setenv(child, '0')
    with pytest.raises(ValueError, match=key):
        v2.configure_child(dict(retr=retr, **{name: value}))


def test_disabled_and_both_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.ENV_RETR, '1')
    for _, env, child in v2.FLAGS:
        monkeypatch.setenv(env, '1')
        monkeypatch.setenv(child, '0')
    assert not any(v2.retr_v2_settings(False).values())
    settings = v2.retr_v2_settings(True)
    assert all(settings.values())
    v2.configure_child(dict(retr=True, **settings))
    assert all(os.environ[child] == '1' for _, _, child in v2.FLAGS)
