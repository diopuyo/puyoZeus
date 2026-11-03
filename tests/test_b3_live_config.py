"""既定OFF・明示ON・配布モデル指定の契約。"""
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from src.phase_j.launcher import parse_user_config, build_pipeline_config, LauncherConfigError
from src.phase_j.live_b3 import configure, LiveB3
from src.b3_live_display import B3Display


def test_shipped_example_enables_b3(tmp_path: Path) -> None:
    """配布直下へコピーされる設定例を読み込み、相対パスとB3の有効化を確認する。"""
    example = Path(__file__).resolve().parents[1] / 'packaging' / 'puyo_live.example.json'
    data = json.loads(example.read_text(encoding='utf-8'))
    expected = {'b3_model': 'app/models/b3/model.joblib',
                'b3_timing_table': 'app/models/b3/timing_table.json'}
    for key, relative in expected.items():
        assert data[key] == relative
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    user = parse_user_config(data, tmp_path)
    config = build_pipeline_config(user)
    assert config['b3_judgment'] is True
    assert user.device_index == 0
    for key, relative in expected.items():
        assert Path(config[key]) == tmp_path / relative


def test_old_native_fails_before_model_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """旧拡張を使ったままOBS入力を開始しない。"""
    from src import puyo_core_bridge as native
    monkeypatch.setattr(native, 'NATIVE_AVAILABLE', True)
    monkeypatch.setattr(native, '_native', SimpleNamespace())
    with pytest.raises(RuntimeError, match='c4ebcfb'):
        LiveB3(tmp_path/'missing.joblib', tmp_path/'missing.json', tmp_path)


def test_launcher_b3_optional(tmp_path: Path) -> None:
    data = dict(device_name='OBS Virtual Camera')
    assert 'b3_judgment' not in build_pipeline_config(parse_user_config(data, tmp_path))
    (tmp_path/'model.joblib').touch()
    (tmp_path/'timing.json').touch()
    data.update(b3_judgment=True, b3_model='model.joblib', b3_timing_table='timing.json')
    config = build_pipeline_config(parse_user_config(data, tmp_path))
    assert config['b3_judgment'] is True
    assert Path(config['b3_model']) == tmp_path/'model.joblib'
    assert 'b3_async' not in config
    data['b3_async'] = True
    assert build_pipeline_config(parse_user_config(data, tmp_path))['b3_async'] is True


@pytest.mark.parametrize('value', ['true', 1, None, True])
def test_async_requires_boolean_and_b3(tmp_path: Path, value: object) -> None:
    with pytest.raises(LauncherConfigError, match='b3_async'):
        parse_user_config(dict(device_name='OBS', b3_async=value), tmp_path)


@pytest.mark.parametrize('value', ['true', 1, None])
def test_launcher_rejects_invalid_flag(tmp_path: Path, value: object) -> None:
    with pytest.raises(LauncherConfigError, match='b3_judgment'):
        parse_user_config(dict(device_name='OBS', b3_judgment=value), tmp_path)


def test_missing_model_fails_before_capture(tmp_path: Path) -> None:
    with pytest.raises(LauncherConfigError, match='b3_model'):
        parse_user_config(dict(device_name='OBS', b3_judgment=True), tmp_path)


def test_b3_display_dto_keys() -> None:
    from src.phase_j.contracts import DisplayLayers
    wrapper = LiveB3.__new__(LiveB3)
    wrapper.display = B3Display()
    wrapper.display.boundary(1)
    wrapper.display.prediction(.8, 0, 1.)
    wrapper.display.update(1.)
    wrapper.display.prediction(.2, 0, 1.1)
    wrapper.display.update(1.1)
    row = {}
    wrapper.decorate(row)
    assert row['raw_probability'] == .2
    assert row['calibration_id'] == 'b3-stage2-display'
    assert row['display_layers']['displayed_p1'] == row['probability']
    assert DisplayLayers(**row['display_layers']).current_p1 == .2


def test_end_audit_can_cross_process() -> None:
    import pickle
    from src.phase_j.live_memory import RecentIntegrityAudit, INTEGRITY_AUDIT_LIMIT
    audit = RecentIntegrityAudit({'n': i} for i in range(INTEGRITY_AUDIT_LIMIT + 1))
    restored = pickle.loads(pickle.dumps(audit))
    assert restored == audit
    assert (restored.append_count, restored.discarded_count) == (audit.append_count, audit.discarded_count)
    restored.append({'n': 'new'})
    assert restored.append_count == audit.append_count + 1


def test_b3_sse_model_identity(tmp_path: Path) -> None:
    from hashlib import sha256
    from src.phase_j.live_b3 import display_assets
    path = tmp_path/'model.joblib'
    path.write_bytes(b'fixed-model')
    assets = dict(prediction_model_hash='reference-R')
    assert display_assets(SimpleNamespace(b3_judgment=False), assets) == assets
    changed = display_assets(SimpleNamespace(b3_judgment=True, b3_model=path), assets)
    assert changed['prediction_model_hash'] == 'sha256:'+sha256(b'fixed-model').hexdigest()
    assert assets['prediction_model_hash'] == 'reference-R'


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('flags', [{}, {'display_completion_driven': False, 'display_fast_ema': False},
                                  {'display_completion_driven': True, 'display_fast_ema': True}])
def test_display_defaults_and_explicit_flags_reach_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool, flags: dict,
) -> None:
    """B3限定の既定ONと明示OFFを、ランチャーから実CLIまで確認する。"""
    import json
    import os
    from scripts.run_live_pipeline_20260928 import parse_args
    from src.phase_j.launcher import build_pipeline_argv
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    for name in ('model.joblib', 'timing.json'):
        (tmp_path / name).touch()
    user = parse_user_config(dict(device_name='OBS', b3_judgment=enabled,
        b3_model='model.joblib', b3_timing_table='timing.json', **flags), tmp_path)
    path = tmp_path / 'pipeline.json'
    path.write_text(json.dumps(build_pipeline_config(user)), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['live', *build_pipeline_argv(user, path)])
    options = parse_args()
    for name in ('display_completion_driven', 'display_fast_ema'):
        assert getattr(user, name) is (enabled and flags.get(name, True))
        assert getattr(options, name) is getattr(user, name)
    monkeypatch.setattr('sys.argv', ['live', *build_pipeline_argv(user, path),
        '--no-display-completion-driven', '--no-display-fast-ema'])
    options = parse_args()
    assert options.display_completion_driven is options.display_fast_ema is False


@pytest.mark.parametrize('enabled', [False, True])
def test_direct_cli_display_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """ランチャーを通さない起動でもB3の有無に追随する。"""
    import os
    from scripts.run_live_pipeline_20260928 import parse_args
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    model, timing = tmp_path / 'model.joblib', tmp_path / 'timing.json'
    model.touch()
    timing.touch()
    monkeypatch.setattr('sys.argv', ['live', '--b3-judgment' if enabled else '--no-b3-judgment',
        '--b3-model', str(model), '--b3-timing-table', str(timing)])
    options = parse_args()
    assert options.display_completion_driven is options.display_fast_ema is enabled
