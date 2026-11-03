"""retR の既定OFF、子への配送、列の保持とモデル互換を検査する。"""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j.live_ex_async import _compute
from src.phase_j.live_ex_hold import LiveExHold
from src.phase_j.live_24h_model import RETR_COLUMNS, validate_model
from tests.test_live_24h_hold import source, event
from tests.test_live_24h_model import Worker, model, result
from src.phase_j.live_24h import Live24h, LEARNING_COLUMNS


@pytest.mark.parametrize('active', [False, True])
def test_compute_flag_overrides_inherited_environment(monkeypatch: pytest.MonkeyPatch, active: bool) -> None:
    import os
    seen = []
    monkeypatch.setenv('PUYO_EX_RETR', '1' if not active else '0')
    connection = SimpleNamespace(recv=lambda: seen.append(os.environ['PUYO_EX_RETR']), close=lambda: None)
    _compute(connection, dict(retr=active))
    assert seen == [str(int(active))]


def test_retr_cli_default_and_parent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_retr is False
    monkeypatch.setenv('PUYO_EX_RETR', '0')
    monkeypatch.setenv(config.ENV_RETR, '0')
    monkeypatch.setenv(config.ENV_ENABLED, '0')
    opt = SimpleNamespace(live_24h=True, live_24h_ama=False, live_24h_retr=True,
                          b3_judgment=True, b3_async=True, b3_m3=True, b3_c2=True)
    config.configure(opt, parser)
    import os
    assert os.environ[config.ENV_RETR] == '1' and os.environ['PUYO_EX_RETR'] == '0'


def test_retr_hold_reset_and_expiry() -> None:
    hold, src = LiveExHold(), source()
    src['extra'].update(dict.fromkeys(RETR_COLUMNS, 7.))
    assert hold.adopt(event(.2), src, 'ok')['retR_c_p1'] == 7.
    held = hold.adopt(event(.3), None, 'not_yet')
    assert all(held[c] == 7. for c in RETR_COLUMNS)
    reset = hold.adopt(event(.4, reset={'p1': True, 'p2': False}), None, 'stale')
    assert math.isnan(reset['retR_c_p1']) and reset['retR_c_p2'] == 7.
    expired = hold.adopt(event(.71), None, 'stale')
    assert all(math.isnan(expired[c]) for c in RETR_COLUMNS)


@pytest.mark.parametrize('retr', [False, True])
def test_only_declared_retr_columns_reach_model(tmp_path: Path, retr: bool) -> None:
    calls, candidate = [], None
    candidate = model(calls)
    columns = LEARNING_COLUMNS + (RETR_COLUMNS if retr else ())
    candidate.static = candidate.exchange = SimpleNamespace(columns=columns)
    values = dict.fromkeys(LEARNING_COLUMNS + RETR_COLUMNS, .7)
    worker = Worker(dict(epoch=1, serial=2, extra=values))
    owner = Live24h(tmp_path, candidate, {}, lambda _: worker, use_ama=True)
    try:
        owner.predict(result())
        assert set(calls[0]) & set(RETR_COLUMNS) == (set(RETR_COLUMNS) if retr else set())
    finally:
        owner.close()


def test_validator_accepts_retr_model() -> None:
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    columns = (*NET_LEDGER_FEATURE_NAMES, *LEARNING_COLUMNS, *RETR_COLUMNS)
    evaluator = SimpleNamespace(columns=columns)
    validate_model(SimpleNamespace(members=(1, 2, 3), static=evaluator, exchange=evaluator))


@pytest.mark.parametrize('retr', [False, True])
def test_launcher_delivers_retr_and_custom_model(tmp_path: Path, retr: bool) -> None:
    from src.phase_j.launcher import build_pipeline_config, parse_user_config
    from tests.test_live_24h_launcher import data
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_retr=retr,
                                  live_24h_model='ex.json'), tmp_path)
    output = build_pipeline_config(user)
    assert output['live_24h_retr'] is retr
    assert output['live_24h_model'] == str(tmp_path/'ex.json')
