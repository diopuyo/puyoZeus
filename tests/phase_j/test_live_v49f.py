"""OBS試験フラグ・公開状態の回帰検査。数値同値は保存入力の全行照合で検査する。"""
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import json

import pytest

from src.phase_j.launcher import parse_user_config, LauncherConfigError
from src.phase_j.live_v49f import LiveV49F
from src.phase_j.live_publish import initial_snapshot, result_snapshot
from src.phase_j.validator import validate_snapshot
from tests.phase_j.test_live_publish import ASSETS, row


def view(value: dict | None = None) -> LiveV49F:
    owner = LiveV49F.__new__(LiveV49F)
    owner.outputs, owner.inputs = Queue(), Queue(1)
    owner.generation, owner.game = 1, 1
    owner.latest = None if value is None else dict(value=value, completed_at=12.)
    owner.reason, owner.failure, owner.c = '更新済み', None, .46
    return owner


def test_trial_default_and_explicit_configuration(tmp_path: Path) -> None:
    config = dict(device_name='OBS Virtual Camera')
    for name in ('model.joblib', 'timing_table.json'):
        (tmp_path/name).write_text('{}')
    config.update(b3_model=str(tmp_path/'model.joblib'), b3_timing_table=str(tmp_path/'timing_table.json'))
    assert parse_user_config(config, tmp_path).live_v49f is False
    assert parse_user_config(dict(config, live_v49f=True, b3_judgment=True, b3_async=True), tmp_path).live_v49f
    for invalid in ('true', 1, None):
        with pytest.raises(LauncherConfigError):
            parse_user_config(dict(config, live_v49f=invalid), tmp_path)
    with pytest.raises(LauncherConfigError):
        parse_user_config(dict(config, live_v49f=True), tmp_path)


@pytest.mark.parametrize('side,probability', [(0, 0.), (1, 1.)])
def test_death_is_exact_and_b3_is_untouched(side: int, probability: float) -> None:
    owner = view(dict(input_sec=10., active=True, death_sides=[side], v_post=.51))
    payload = dict(t_sec=11.25, raw_probability=.63, probability=.61, match_active=True)
    owner.decorate(payload)
    assert payload['probability'] == .61 and payload['raw_probability'] == .63
    assert payload['v49f']['p1'] == probability
    assert payload['v49f']['age_sec'] == 1.25
    assert payload['v49f']['reason'] == '入力変化待ち'


def test_boundary_discards_late_death_and_old_probability() -> None:
    owner = view(dict(input_sec=10., active=True, death_sides=[0], v_post=.51))
    owner.outputs.put(dict(generation=1, game=1, value=dict(death_sides=[0])))
    owner.update(dict(game_idx=2), 2)
    assert owner.latest is None
    payload = dict(t_sec=20., raw_probability=.7, match_active=True)
    owner.decorate(payload)
    assert payload['v49f']['p1'] is None


def test_between_matches_hides_previous_death() -> None:
    owner = view(dict(input_sec=10., active=True, death_sides=[0], v_post=.51))
    payload = dict(t_sec=11., raw_probability=.7, match_active=False)
    owner.decorate(payload)
    assert payload['v49f']['p1'] is None and payload['v49f']['death_sides'] == []


def test_extra_dto_is_optional_and_schema_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.phase_j.live_publish.utc_now', lambda: '2026-10-19T00:00:00+00:00')
    initial = initial_snapshot(ASSETS)
    data = row()
    before = result_snapshot(initial, data, 4., 1, None)
    assert 'v49f' not in before.evaluations
    data['v49f'] = dict(label='撃ち合い勝率 v4.9f', p1=1., p2=0., age_sec=.5,
                        reason='入力変化待ち', death_sides=[1], input_sec=2., completed_at=3.)
    after = result_snapshot(initial, data, 4., 1, None)
    assert validate_snapshot(after).is_valid
    left, right = before.to_mapping(), after.to_mapping()
    right['evaluations'].pop('v49f')
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
