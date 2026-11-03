"""配信の置きによる連鎖分割と設定の配送を検証する。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j.live_chain_split_config import ENV_CHAIN_SPLIT_PLACE, set_split_env
from src.phase_j.live_ex_capture import add_fire_inputs
from src.phase_j.launcher import build_pipeline_config, parse_user_config
from tests.test_live_24h_launcher import data


@pytest.mark.parametrize('flag', [False, True])
def test_launcher_cli_and_child_settings(flag: bool, tmp_path: Path) -> None:
    raw = data(tmp_path, live_24h=True, live_24h_model='ex.json',
               live_chain_split_place=flag, live_24h_retr=True,
               live_24h_xc=True, live_24h_xc_same_time=True,
               live_24h_ex_fire_first=True, live_24h_stable_key=True)
    user = parse_user_config(raw, tmp_path)
    pipeline = build_pipeline_config(user)
    assert pipeline.get('live_chain_split_place', False) is flag
    if not flag:
        assert 'live_chain_split_place' not in pipeline
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    options = parser.parse_args(['--live-chain-split-place'] if flag else [])
    assert options.live_chain_split_place is flag
    with patch.dict(os.environ, {ENV_CHAIN_SPLIT_PLACE: '1'}):
        config.configure(SimpleNamespace(**pipeline), parser)
        assert os.environ.get(ENV_CHAIN_SPLIT_PLACE) == ('1' if flag else None)
        with patch('src.phase_j.live_ex_runtime.settings_from_env', return_value={}):
            assert config.ex_settings().get('chain_split_place', False) is flag


@pytest.mark.parametrize('bad', [1, 'true', None])
def test_json_rejects_non_bool(bad: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='live_chain_split_place'):
        config.launcher_values(data(tmp_path, live_24h=True, live_24h_ama=False,
                                   live_chain_split_place=bad), tmp_path)


@pytest.mark.parametrize('flag', [False, True])
def test_capture_preserves_side_counts_only_when_on(flag: bool) -> None:
    with patch.dict(os.environ, {'PUYO_LIVE_24H_RETR': '1'}):
        set_split_env(flag)
        row = dict(sides=[dict(tsumo_count=5), dict(tsumo_count=17)])
        raw = [{}, {}]
        for index in range(2):
            add_fire_inputs(raw[index], row, index)
        assert [('tsumo_count' in side) for side in raw] == [flag, flag]
        if flag:
            assert [side['tsumo_count'] for side in raw] == [5, 17]


@pytest.mark.parametrize('module', ['src.fire_board', 'src.phase_j._live_ex_frozen.src.fire_board'])
@pytest.mark.parametrize('flag,place,split', [(False, 1., False), (True, 1., True),
                                           (True, None, False), (True, .1, False)])
def test_forecast_split_and_real_chain(module: str, flag: bool, place: float | None, split: bool) -> None:
    from importlib import import_module
    with patch.dict(os.environ):
        set_split_env(flag)
        forecast = import_module(module).PairForecast(trace=lambda b: ([], b))
        ids = []
        for t, score in [(0., 40), (.1, 40), (1., None), (1.3, 40), (2.7, 320)]:
            count = 5 + int(place is not None and t >= place)
            raw = [dict(tsumo_count=count, formula=dict(valid=score is not None, product=score)), {}]
            forecast.observe(t, 1, raw)
            if t == 1. and split:
                assert forecast.current(0, t) is None
            if score is not None:
                ids.append(forecast.sides[0].fc.chain_id)
        assert (ids[0] != ids[-1]) is split
        assert forecast.sides[0].fc.observed == (40, 320)


@pytest.mark.parametrize('flag', [False, True])
def test_b3_forecast_receives_observed_count(flag: bool) -> None:
    from src.b3_state import SideState
    with patch.dict(os.environ):
        set_split_env(flag)
        state = SideState('1P', enable_fire_board=True)
        ids = []
        for t, score, count in [(0., 40, 5), (.1, 40, 5), (1., None, 6), (1.3, 40, 6)]:
            side = dict(state='CHAIN', tsumo_count=count, next_slide_motion=False,
                        score=None, confirmed_board=None,
                        formula=dict(valid=score is not None, product=score))
            state.update_frame(t, side, True)
            if score is not None:
                ids.append(state.forecast.fc.chain_id)
        assert (ids[0] != ids[-1]) is flag
