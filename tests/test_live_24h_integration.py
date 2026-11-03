"""起動時固定・OFF互換・特徴だけの子計算・表示選択の停止を検査する。"""
import argparse
from io import StringIO
from pathlib import Path
import pickle
import os
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import b3_async
from src.phase_j import live_24h_config as C
from src.phase_j import live_winrate_switch as S
from src.phase_j.live_b3 import LiveB3


@pytest.fixture(autouse=True)
def restore_environment() -> object:
    with patch.dict(os.environ):
        yield


def options(**changes: object) -> SimpleNamespace:
    values = dict(live_24h=True, live_24h_ama=False, live_24h_delay_ms=200.,
                  b3_judgment=True, b3_async=True, b3_m3=True, b3_c2=True,
                  live_v49f=True, live_ama_plain=True, live_winrate_switch=True)
    return SimpleNamespace(**dict(values, **changes))


def test_off_does_not_change_existing_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_WINRATE_SWITCH', '1')
    opt = options(live_24h=False)
    before = vars(opt).copy()
    C.configure(opt, argparse.ArgumentParser())
    assert vars(opt) == before and S.enabled()


def test_on_disables_old_helpers_and_line_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_24H', '0')
    opt = options()
    C.configure(opt, argparse.ArgumentParser())
    assert not opt.live_v49f and not opt.live_ama_plain and not opt.live_winrate_switch
    assert C.enabled() and not C.ama_enabled() and not S.enabled()


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf'), True, '200'])
def test_invalid_delay_is_rejected(bad: object) -> None:
    with pytest.raises(SystemExit):
        C.configure(options(live_24h_delay_ms=bad), argparse.ArgumentParser())


def test_model_path_and_work_limit_are_fixed_at_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'model.json'
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setenv('PUYO_EX_WORK_LIMIT', '1')
    C.configure(options(live_24h_ama=True, live_24h_model=path), argparse.ArgumentParser())
    import os
    assert os.environ['PUYO_EX_WORK_LIMIT'] == '450'
    assert os.environ['PUYO_EX_WORK_BUDGET'] == '1'
    assert os.environ[C.ENV_MODEL] == str(path.resolve())


def test_features_only_never_calls_c2_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_B3_TIME_GATE', '0')
    monkeypatch.setattr(b3_async, 'contents_from_row', lambda row: {})
    rows = SimpleNamespace(build=lambda: dict(mode_exchange=1, x=2.))
    model = SimpleNamespace(predict_row=lambda row: pytest.fail('C2を余分に推論'))
    request = dict(snapshot=pickle.dumps(('state', 1)), requested_at=0., _features_only=True)
    result = b3_async.calculate(request, rows, model, measure=False)
    assert result['error'] is None and result['raw_probability'] is None
    assert result['row'] == dict(mode_exchange=1, x=2.)


def test_features_only_setting_reaches_state_packet() -> None:
    subject = object.__new__(b3_async.AsyncB3)
    subject.finished, subject.closed, subject.submitted = False, False, 0
    subject.inputs = Queue()
    subject.submit({}, features_only=True)
    metadata = pickle.loads(subject.inputs.get())[-1]
    assert metadata['_features_only'] and metadata['_include_row']


def test_off_submit_metadata_is_unchanged() -> None:
    subject = object.__new__(b3_async.AsyncB3)
    subject.finished, subject.closed, subject.submitted = False, False, 0
    subject.inputs = Queue()
    subject.submit({})
    assert set(pickle.loads(subject.inputs.get())[-1]) == {'submitted_at', 'captured_at'}


def test_legacy_switch_cannot_override_unified_even_with_old_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_24H', '1')
    monkeypatch.setenv('PUYO_LIVE_WINRATE_SWITCH', '1')
    snapshot = object()
    assert S.with_switch(snapshot, {'_v49f_active': True}) is snapshot


def test_model_failure_flows_into_existing_b3_error() -> None:
    live = object.__new__(LiveB3)
    def broken(result: dict) -> float:
        raise ValueError('試験用失敗')
    live.live24h = SimpleNamespace(predict=broken, ama=True)
    value = live._predict_24h(dict(raw_probability=None, error=None))
    assert value['raw_probability'] is None and value['error'] == 'ValueError: 試験用失敗'


def test_unified_html_contains_no_legacy_display_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import overlay_http as H
    for name in ('PUYO_LIVE_24H', 'PUYO_LIVE_V49F', 'PUYO_LIVE_AMA_PLAIN', 'PUYO_LIVE_WINRATE_SWITCH'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setattr(H, 'configured_html', lambda html, options: html)
    sent = []
    owner = SimpleNamespace(overlay_html='<html></html>', overlay_options={},
                            _serve_asset=lambda body, mime: sent.append(body))
    H.OverlayPagesMixin._serve_html(owner)
    assert 'overlay_24h.js' in sent[0]
    assert all(name not in sent[0] for name in ('overlay_v49f.js', 'overlay_ama.js',
                                               'overlay_winrate_switch.js', 'overlay_x_prefire.js'))


def test_capture_off_never_imports_ex(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_b3 import _capture_24h
    monkeypatch.setenv('PUYO_LIVE_24H', '0')
    row = {}
    _capture_24h(None, None, None, row, 0)
    assert row == {}


def test_parent_inference_cost_is_not_counted_as_queue_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_b3 as B
    clock = iter((10., 10.2))
    monkeypatch.setattr(B, 'perf_counter', lambda: next(clock))
    live = object.__new__(LiveB3)
    live.live24h = SimpleNamespace(ama=True, predict=lambda result: .7)
    value = live._predict_24h(dict(raw_probability=None, error=None, completed_at=9.9, timings={}))
    assert value['raw_probability'] == .7 and value['completed_at'] == 10.2
    assert value['timings']['inference_ms'] == pytest.approx(200.)
    assert value['feature_result_wait_ms'] == pytest.approx(100.)


def test_b3_startup_failure_closes_already_started_ex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed = []
    live = object.__new__(LiveB3)
    live.live24h = SimpleNamespace(close=lambda: closed.append('EX'))
    live.async_engine, live.table, live.time_gate = None, object(), None
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError('B3起動の試験用失敗')
    monkeypatch.setattr(b3_async, 'AsyncB3', fail)
    with pytest.raises(RuntimeError, match='B3起動'):
        live._open_pipeline(tmp_path/'model', tmp_path/'table', tmp_path, True)
    assert closed == ['EX']
