"""セカンド8列の旗・保持・モデルへの配送を検査する。"""
import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
from src.phase_j.live_24h import Live24h, LEARNING_COLUMNS
from src.phase_j.live_24h_model import validate_model
from src.phase_j.live_ex_async import _compute
from src.phase_j.live_ex_hold import LiveExHold
from tests.test_live_24h_hold import source, event
from tests.test_live_24h_model import Worker, model, result


@pytest.mark.parametrize('active', [False, True])
def test_compute_overrides_inherited_second(monkeypatch: pytest.MonkeyPatch, active: bool) -> None:
    monkeypatch.setenv(retr.SECOND_ENV, str(int(not active)))
    seen = []
    connection = SimpleNamespace(recv=lambda: seen.append(os.environ[retr.SECOND_ENV]), close=lambda: None)
    _compute(connection, dict(retr=True, retr_second=active))
    assert seen == [str(int(active))]
    assert set(retr.blank()) == set(retr.COLUMNS + (retr.SECOND_COLUMNS if active else ()))


@pytest.mark.parametrize('value,enabled', [(1, True), ('true', True), (None, True), (True, False)])
def test_invalid_second(value: object, enabled: bool) -> None:
    with pytest.raises(ValueError):
        config.validate_retr_second(value, enabled)


@pytest.mark.parametrize('active', [False, True])
def test_cli_environment_and_settings(monkeypatch: pytest.MonkeyPatch, active: bool) -> None:
    from src.phase_j import live_ex_runtime
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_retr_second is False
    flag = '--live-24h-retr-second' if active else '--no-live-24h-retr-second'
    assert parser.parse_args([flag]).live_24h_retr_second is active
    for name in (config.ENV_ENABLED, config.ENV_AMA, config.ENV_RETR, config.ENV_RETR_SECOND):
        monkeypatch.setenv(name, '1')
    options = SimpleNamespace(live_24h=True, live_24h_ama=False, live_24h_retr=True,
                              live_24h_retr_second=active, b3_judgment=True,
                              b3_async=True, b3_m3=True, b3_c2=True)
    config.configure(options, parser)
    assert os.environ[config.ENV_RETR_SECOND] == str(int(active))
    monkeypatch.setenv(config.ENV_AMA, '1')
    assert config.ex_settings()['retr_second'] is active


@pytest.mark.parametrize('active', [False, True])
def test_json_and_pipeline(tmp_path: Path, active: bool) -> None:
    from src.phase_j.launcher import build_pipeline_config, parse_user_config
    from tests.test_live_24h_launcher import data
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_retr=True,
                                  live_24h_retr_second=active, live_24h_model='ex.json'), tmp_path)
    assert user.live_24h_retr_second is active
    assert build_pipeline_config(user)['live_24h_retr_second'] is active


def test_hold_reset_expiry_and_swap() -> None:
    hold, src = LiveExHold(), source()
    values = {c: float(i) for i, c in enumerate(retr.COLUMNS + retr.SECOND_COLUMNS)}
    src['extra'].update(values)
    adopted = hold.adopt(event(.2), src, 'ok')
    assert all(adopted[c] == v for c, v in values.items())
    held = hold.adopt(event(.3), None, 'not_yet')
    assert all(held[c] == v for c, v in values.items())
    swapped = retr.swap(values)
    for stem in retr.STEMS + retr.SECOND_STEMS:
        assert swapped[stem + '_p1'] == values[stem + '_p2']
        assert swapped[stem + '_p2'] == values[stem + '_p1']
    reset = hold.adopt(event(.4, reset={'p1': True, 'p2': False}), None, 'stale')
    for c in values:
        assert math.isnan(reset[c]) if c.endswith('_p1') else reset[c] == values[c]
    expired = hold.adopt(event(.71), None, 'stale')
    assert all(math.isnan(expired[c]) for c in values)


@pytest.mark.parametrize('active', [False, True])
def test_only_model_columns_reach_b3_input(tmp_path: Path, active: bool) -> None:
    calls = []
    candidate = model(calls)
    columns = LEARNING_COLUMNS + retr.COLUMNS + (retr.SECOND_COLUMNS if active else ())
    candidate.static = candidate.exchange = SimpleNamespace(columns=columns)
    values = dict.fromkeys(LEARNING_COLUMNS + retr.COLUMNS + retr.SECOND_COLUMNS, .7)
    worker = Worker(dict(epoch=1, serial=2, extra=values))
    owner = Live24h(tmp_path, candidate, {}, lambda _: worker, use_ama=True,
                   temperature_enabled=False, counter_fire_shift_enabled=False, midgame_board_layer_enabled=False)
    try:
        owner.predict(result())
        assert set(calls[0]) & set(retr.SECOND_COLUMNS) == (set(retr.SECOND_COLUMNS) if active else set())
    finally:
        owner.close()


def test_validate_second_model() -> None:
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    columns = (*NET_LEDGER_FEATURE_NAMES, *LEARNING_COLUMNS, *retr.COLUMNS, *retr.SECOND_COLUMNS)
    evaluator = SimpleNamespace(columns=columns)
    validate_model(SimpleNamespace(members=(1, 2, 3), static=evaluator, exchange=evaluator))


@pytest.mark.parametrize('value,retr_flag', [(1, True), (True, False)])
def test_json_rejects_invalid_second(tmp_path: Path, value: object, retr_flag: bool) -> None:
    from src.phase_j.launcher import parse_user_config
    from tests.test_live_24h_launcher import data
    with pytest.raises(ValueError, match='live_24h_retr_second'):
        parse_user_config(data(tmp_path, live_24h=True, live_24h_retr=retr_flag,
                               live_24h_retr_second=value, live_24h_model='ex.json'), tmp_path)


def test_calculate_extra_contains_second(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_ex_engine as live
    values = dict.fromkeys(retr.COLUMNS + retr.SECOND_COLUMNS, 7.)
    wrapper = object.__new__(live.LiveExEngine)
    wrapper.monitor = SimpleNamespace(take=lambda: None)
    wrapper.engine = SimpleNamespace(probe=lambda: SimpleNamespace(fight=True, key=(1, 2, (None, None))))
    monkeypatch.setattr(live, 'compute', lambda _: dict(failure='test', proof=None, retr=values))
    output = wrapper.calculate()
    assert all(output['extra'][c] == v for c, v in values.items())
