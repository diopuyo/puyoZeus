"""本人用OBS試験の既定OFF・最新1件・別枠保持・入力契約を検査する。"""
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import pytest

from src.phase_j.live_ama_async import replace_latest
from src.phase_j.live_ama_inputs import queue_only
from src.phase_j.live_ama_engine import request_from_saved
from src.phase_j.live_obs_trial import baseline_notice
from src.ama_plain_v71_display import Display
from src.phase_j.launcher import parse_user_config, build_pipeline_config, LauncherConfigError


def test_trial_default_off() -> None:
    config = parse_user_config(dict(device_name='OBS Virtual Camera'), Path('D:/fixed'))
    assert not config.live_ama_plain and not config.b3_m3 and not config.b3_time_gate and not config.live_score_cadence
    assert not config.live_static_exact
    effective = build_pipeline_config(config)
    assert not {'live_ama_plain', 'b3_m3', 'b3_time_gate', 'live_score_cadence', 'live_static_exact'} & effective.keys()


@pytest.mark.parametrize('key', ['live_ama_plain', 'b3_m3', 'b3_time_gate', 'live_score_cadence', 'live_static_exact'])
def test_trial_requires_async(key: str) -> None:
    with pytest.raises(LauncherConfigError):
        parse_user_config(dict(device_name='OBS Virtual Camera', **{key: True}), Path('D:/fixed'))


def test_capture_preparation_does_not_observe_and_closes_on_early_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import b3_observation
    from src.phase_j import live_b3
    calls = []
    class Capture:
        def __init__(self, **options: object) -> None:
            calls.append('load')
        def close(self) -> None:
            calls.append('close')
    monkeypatch.setattr(b3_observation, 'ObservationCapture', Capture)
    monkeypatch.setenv('PUYO_B3_JUDGMENT', '1')
    owner = SimpleNamespace()
    monkeypatch.setenv('PUYO_LIVE_SCORE_CADENCE', '0')
    live_b3.prepare_capture(owner)
    assert not calls and not vars(owner)
    monkeypatch.setenv('PUYO_LIVE_SCORE_CADENCE', '1')
    live_b3.prepare_capture(owner)
    live_b3.prepare_capture(owner)
    assert calls == ['load']
    live_b3.close_capture(owner)
    live_b3.close_capture(owner)
    assert calls == ['load', 'close']


def test_pending_work_is_latest_only() -> None:
    queue = Queue(1)
    assert replace_latest(queue, {'id': 1}) == 0
    assert replace_latest(queue, {'id': 2}) == 1
    assert replace_latest(queue, {'id': 3}) == 1
    assert queue.qsize() == 1 and queue.get_nowait() == {'id': 3}


def test_uninitialized_score_ledger_reports_reason_without_fabricating_job() -> None:
    from src.phase_j.live_ama_async import input_status
    from src.phase_j.live_ama_inputs import AmaInputs
    from src.b3_score_ledger import ScoreLedger
    owner = AmaInputs.__new__(AmaInputs)
    owner.ledger = ScoreLedger(enable_inflight=False, observed_landing_rule=False)
    raw = [dict(state={'state': 'STABLE'}, score=12, confirmed_board=None)]*2
    for stamp in (1., 1.1, 1.2):
        assert owner.book(dict(t=stamp, game=1, raw=raw))['pending'] == [None, None]
        assert input_status(owner, None, stamp) == dict(input_sec=stamp, missing='得点未確定・連続観測待ち')
    assert owner.book(dict(t=1.23, game=1, raw=raw))['pending'] == [0, 0]
    assert input_status(owner, None, 1.23) is None


def test_time_gate_actual_snapshot_is_structured_and_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import json
    import runpy
    from threading import Lock
    from src.phase_j import live_b3_time_gate as gate, live_publish
    owner = gate.TimeGate.__new__(gate.TimeGate)
    owner.log, owner.lock = io.StringIO(), Lock()
    monkeypatch.setattr(gate, 'ACTIVE', owner)
    assets = {k: 'test' for k in ('app_build_id', 'recognition_model_hash', 'recognition_config_hash',
                                 'prediction_model_hash', 'calibration_hash')}
    snapshot = live_publish.initial_snapshot(assets)
    gate.published(dict(frame=1, t_sec=0.), snapshot, 1.)
    row = json.loads(owner.log.getvalue())
    assert isinstance(row['display'], dict) and isinstance(row['practical'], dict)
    script = Path(__file__).resolve().parents[2]/'packaging/summarize_live_obs_time_gate.py'
    aggregate = runpy.run_path(str(script))['time_rates']
    rates = aggregate([dict(row, published_at=1.), dict(row, published_at=11.)])
    assert rates['seconds'] == 10. and rates['null_rate'] == 1.


def test_tape_observation_preserves_slide_evidence() -> None:
    assert queue_only(dict(next_pair={'tuple': [1, 2]}, dnext_pair={'tuple': [3, 4]},
                           next_slide_motion=True)) == ([1, 2], [3, 4], True)


def test_missing_retains_previous_value_without_time_limit() -> None:
    display = Display(True)
    display.observe(dict(input_sec=1., value=dict(probability=.7)))
    display.observe(dict(input_sec=2., missing='不明色または不正盤面'))
    shown = display.at(42., ['STABLE', 'STABLE'])
    assert shown['probability'] == .7 and shown['hold_sec'] == 41.
    assert shown['hold_reason'] == '不明色または不正盤面'
    assert Display(True).at(42., [])['probability'] is None


def test_baseline_frame_does_not_mutate_m3_observation() -> None:
    frame = dict(frame=1, sides=[dict(icon={'total_count': 6}), dict(icon=None)], _ama={'raw': []})
    notice = SimpleNamespace(b3=(frame, (), (), 0), captured_at=10.)
    baseline = baseline_notice(notice)
    assert baseline.b3[0]['sides'][0]['icon'] is None
    assert '_ama' not in baseline.b3[0]
    assert notice.b3[0]['sides'][0]['icon'] == {'total_count': 6}


def test_m3_flags_do_not_mutate_production(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import b3_trial_config, production_config
    before = vars(production_config).copy()
    monkeypatch.setenv('PUYO_B3_M3', '1')
    assert all(getattr(b3_trial_config, flag) for flag in b3_trial_config.M3_FLAGS)
    assert vars(production_config) == before
    monkeypatch.setenv('PUYO_B3_M3', '0')
    assert not b3_trial_config.B3_LEDGER_INFLIGHT_DEFAULT


def test_ama_reports_missing_while_previous_calculation_is_running() -> None:
    from src.phase_j.live_ama import LiveAma
    owner = LiveAma.__new__(LiveAma)
    owner.display = Display(True)
    owner.display.observe(dict(input_sec=1., value=dict(probability=.7, plans=[])))
    owner.worker = SimpleNamespace(poll=lambda: [])
    owner.states, owner.failure, owner.completed = ['STABLE','STABLE'], None, None
    owner.input_status = dict(input_sec=2., missing='発火直前STABLE盤なし・予測不可')
    row = dict(t_sec=12.,match_active=True)
    owner.decorate(row)
    assert row['ama_plain']['p1'] == .7 and row['ama_plain']['age_sec'] == 11.
    assert row['ama_plain']['reason'] == owner.input_status['missing']


def test_v49f_mixture_keeps_v6_when_practical_m3_changes() -> None:
    from tests.phase_j.test_live_v49f import view
    owner = view(dict(input_sec=10.,active=False,death_sides=[],v_post=.51,pre_delta_full=.12))
    previous = dict(t_sec=11.,raw_probability=.63,match_active=True)
    owner.decorate(previous)
    trial = dict(t_sec=11.,raw_probability=.21,match_active=True)
    owner.decorate(trial,baseline=.63,separate_baseline=True)
    assert trial['v49f'] == previous['v49f']
    assert trial['raw_probability'] == .21
    owner.decorate(trial,baseline=None,separate_baseline=True)
    assert trial['v49f']['p1'] is None


def test_ama_publication_retains_value_during_input_verification() -> None:
    from src.phase_j.live_ama_publication import AmaPublication
    owner = AmaPublication()
    owner.observe(dict(game=2,captured_at=10.,ama_plain=dict(p1=.7,age_sec=1.,reason='次の確定盤待ち')),10.)
    held = owner.project(dict(game=2,input_calibration=dict(phase='verifying')),20.)
    assert held['ama_plain']['p1'] == .7 and held['ama_plain']['age_sec'] == 11.
    assert '入力確認中' in held['ama_plain']['reason']
    owner.observe(dict(game=3),21.)
    assert 'ama_plain' not in owner.project(dict(game=3),22.)
