"""b3_c2 (勝率モデルを C2 に替え、予告アイコン CNN を読まない構成。既定OFF) の配線の試験。

C2 = M3 − アイコン由来 13 列 + 台帳の正味 17 列 (299 列)。本番用モデルの SHA256 はまだ無いので、
試験は仮のモデル (列の名前だけを持つ評価器) と C2_SHA256 の差し替えで行う。

  段 1 (軽い)  : フラグ配線・OFF の同一性・子 process への環境・LiveB3 の列の照合 (A5)・記録と表示名
  段 A1 (重い) : 保存記録 video_51 を、アイコンあり / icon=None で流して C2 の 299 列がビット一致 (子 process で 2 回)
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.phase_j import live_obs_trial as trial
from src.phase_j.launcher import LauncherConfigError, UserConfig, build_pipeline_config, parse_user_config

ROOT = Path(__file__).resolve().parents[1]
FAKE_MODEL_BYTES = b'c2-provisional-model'
FAKE_SHA = sha256(FAKE_MODEL_BYTES).hexdigest()
OTHER_SHA = sha256(b'another-model').hexdigest()
C2_MODEL = Path('/mnt/c/puyo1_work/b3_c2_model_20261025/large_seed1003.joblib' if os.name != 'nt'
                else 'C:/puyo1_work/b3_c2_model_20261025/large_seed1003.joblib')   # PC1 が照合済みの本番用 C2 (LOCK_C2.json)
C2_REAL_SHA = '79d7cb0772936a5f4eda9ba1733c2274ea303f39859af3d6825f138bfa0694d6'
WIRING_DIR = Path('/mnt/c/puyo1_work/b3_c2_wiring_20261025' if os.name != 'nt' else 'C:/puyo1_work/b3_c2_wiring_20261025')
M3_MODEL_NAME = 'PuyoLive_OBS_20261025c/PuyoLive/app/models/b3/model.joblib'   # 24c の配布に入っている M3 (11f7b32a)
BASE = dict(source='dshow', device_name='OBS Virtual Camera', b3_judgment=True, b3_async=True, live_v49f=True,
            b3_m3=True, b3_time_gate=True, b3_ledger_mode=True, b3_ledger_landing=True, live_score_cadence=True,
            live_static_exact=True, live_ama_plain=True)
C2_ENV_KEYS = (trial.C2_ENV, trial.NET_LEDGER_ENV, trial.SKIP_ICON_ENV, 'PUYO_B3_M3')


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure / worker が環境変数を書き換えるので、試験ごとに写しへ差し替える。"""
    env = {k: v for k, v in os.environ.items() if not k.startswith('PUYO_B3_')}
    monkeypatch.setattr(os, 'environ', env)


def make_model(tmp_path: Path) -> Path:
    path = tmp_path/'model.joblib'
    path.write_bytes(FAKE_MODEL_BYTES)
    return path


def options(model: Path, **over: Any) -> SimpleNamespace:
    values = dict(b3_judgment=True, b3_async=True, b3_m3=True, b3_time_gate=True, live_ama_plain=True,
                  live_score_cadence=True, live_static_exact=True, b3_c2=True, b3_skip_icon_reading=True,
                  b3_model=model)
    values.update(over)
    return SimpleNamespace(**values)


def configure_error(opts: SimpleNamespace, capsys: pytest.CaptureFixture) -> str:
    with pytest.raises(SystemExit):
        trial.configure(opts, argparse.ArgumentParser())
    return capsys.readouterr().err


# ------------------------------------------------------------------------------------ configure (起動の設定)

def test_off_sets_env_zero_and_keeps_the_m3_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture) -> None:
    model = make_model(tmp_path)
    monkeypatch.setattr(trial, 'M3_SHA256', FAKE_SHA)
    os.environ[trial.NET_LEDGER_ENV] = '1'                       # 親の環境の持ち込みは OFF が '0' で上書きする
    trial.configure(options(model, b3_c2=False, b3_skip_icon_reading=False), argparse.ArgumentParser())
    assert (os.environ[trial.C2_ENV], os.environ[trial.NET_LEDGER_ENV], os.environ['PUYO_B3_M3']) == ('0', '0', '1')
    trial.configure(SimpleNamespace(b3_judgment=True, b3_async=True), argparse.ArgumentParser())   # b3_c2 属性なしでも従来どおり
    assert os.environ[trial.NET_LEDGER_ENV] == '0' and os.environ['PUYO_B3_M3'] == '0'
    assert 'M3の学習構成ではアイコン読取りが必要です' in configure_error(
        options(model, b3_c2=False, b3_skip_icon_reading=True), capsys)             # M3 の規則は従来のまま
    monkeypatch.setattr(trial, 'M3_SHA256', OTHER_SHA)
    assert 'M3モデルの固定SHA256が不一致です' in configure_error(options(model, b3_c2=False), capsys)


def test_on_requires_registered_sha_and_the_prerequisites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture) -> None:
    model = make_model(tmp_path)
    monkeypatch.setattr(trial, 'M3_SHA256', FAKE_SHA)             # M3 の SHA に合うモデルでも C2 としては通らない
    monkeypatch.setattr(trial, 'C2_SHA256', '')                   # 未登録の状態 (SHA を入れる前) は起動を拒否する
    assert 'C2モデルのSHA256が未登録' in configure_error(options(model), capsys)
    monkeypatch.setattr(trial, 'C2_SHA256', OTHER_SHA)
    assert 'C2モデルの固定SHA256が不一致です' in configure_error(options(model), capsys)
    monkeypatch.setattr(trial, 'C2_SHA256', FAKE_SHA)
    for over, word in ((dict(b3_skip_icon_reading=False), 'b3_skip_icon_reading=true'),
                       (dict(b3_skip_icon_reading=None), 'b3_skip_icon_reading=true'),
                       (dict(b3_m3=False), 'b3_m3'), (dict(b3_async=False), 'b3_async'),
                       (dict(b3_judgment=False), 'b3_judgment'), (dict(b3_c2='true'), 'b3_c2はboolです')):
        assert word in configure_error(options(model, **over), capsys), over


def test_on_sets_env_for_children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model(tmp_path)
    monkeypatch.setattr(trial, 'C2_SHA256', FAKE_SHA)
    trial.configure(options(model), argparse.ArgumentParser())
    assert (os.environ[trial.C2_ENV], os.environ[trial.NET_LEDGER_ENV], os.environ['PUYO_B3_M3']) == ('1', '1', '1')
    assert trial.c2_enabled() and trial.trial_label() == trial.C2_LABEL == '実戦勝率: B3 (C2)'
    os.environ[trial.C2_ENV] = '0'
    assert trial.trial_label() == trial.M3_LABEL == '実戦勝率: B3 (M3)'     # M3 の表示名は従来と同じ文字列


def test_skip_icon_env_name_matches_live_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    """記録 icon_reading が読む環境変数名は、観測器の引数を決める live_speed と同じ (名前がずれると記録が嘘になる)。"""
    from src.phase_j import live_speed
    assert trial.icon_reading_enabled() is True                 # 未設定 = 読んでいる
    os.environ[trial.SKIP_ICON_ENV] = '1'
    assert live_speed.observation_options().get('skip_icon_reading') is True and not trial.icon_reading_enabled()


# ------------------------------------------------------------------------------------ launcher (利用者設定)

def ready_config(tmp_path: Path, **over: Any) -> dict:
    (tmp_path/'model.joblib').write_bytes(FAKE_MODEL_BYTES)
    (tmp_path/'timing.json').write_text('{}', encoding='utf-8')
    return dict(BASE, b3_model='model.joblib', b3_timing_table='timing.json', **over)


def test_launcher_off_has_no_c2_key_and_c2_keeps_its_position(tmp_path: Path) -> None:
    from src.phase_j.launcher import USER_KEYS
    user = parse_user_config(ready_config(tmp_path), tmp_path)
    assert user.b3_c2 is False and 'b3_c2' not in build_pipeline_config(user)      # pipeline_config.json は従来と同じ
    # 後続の項目が増えても、既存の位置引数の位置は変えない。
    c2_position = 81  # UserConfig に b3_c2 を追加した時の位置 (0 始まり)。
    assert 'b3_c2' in USER_KEYS and list(UserConfig.__dataclass_fields__).index('b3_c2') == c2_position
    plain = parse_user_config(dict(device_name='OBS Virtual Camera'), tmp_path)
    assert plain.b3_c2 is False and 'b3_c2' not in build_pipeline_config(plain)


def test_launcher_on_and_its_prerequisites(tmp_path: Path) -> None:
    config = build_pipeline_config(parse_user_config(
        ready_config(tmp_path, b3_c2=True, b3_skip_icon_reading=True), tmp_path))
    assert config['b3_c2'] is True and config['b3_skip_icon_reading'] is True and config['b3_m3'] is True
    for over, word in ((dict(b3_skip_icon_reading=False), 'b3_skip_icon_reading'),
                       (dict(b3_m3=False, b3_time_gate=False), 'b3_m3'), (dict(b3_async=False, live_v49f=False,
                                                                           b3_m3=False, b3_time_gate=False), 'b3_async'),
                       (dict(b3_c2='true'), 'b3_c2'), (dict(b3_c2=1), 'b3_c2')):
        data = ready_config(tmp_path, b3_c2=True, b3_skip_icon_reading=True)
        data.update(over)
        for dependent in ('live_ama_plain', 'live_score_cadence', 'live_static_exact'):
            data[dependent] = data[dependent] and data['b3_async']
        with pytest.raises(LauncherConfigError, match=word):
            parse_user_config(data, tmp_path)


def test_apply_config_registers_b3_c2_and_keeps_the_script_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """配布の run_live_pipeline は基準点に固定なので、--b3-c2 は live_config が登録する。JSON の鍵も通る。"""
    pytest.importorskip('torch')
    from scripts.run_live_pipeline_20260928 import parse_args
    monkeypatch.chdir(ROOT)
    model = make_model(tmp_path)
    monkeypatch.setattr(trial, 'C2_SHA256', FAKE_SHA)
    table = tmp_path/'timing.json'
    table.write_text('{}', encoding='utf-8')
    for on in (False, True):
        data = ready_config(tmp_path, b3_c2=on, b3_skip_icon_reading=on)
        data.update(b3_model=str(model), b3_timing_table=str(table))
        config = tmp_path/'pipeline_config.json'
        config.write_text(json.dumps(build_pipeline_config(parse_user_config(data, tmp_path))), encoding='utf-8')
        monkeypatch.setattr(sys, 'argv', ['live', '--config', str(config), '--output', str(tmp_path/f'run{on}')])
        if not on:
            monkeypatch.setattr(trial, 'M3_SHA256', FAKE_SHA)
        parsed = parse_args()
        assert parsed.b3_c2 is on and os.environ[trial.NET_LEDGER_ENV] == str(int(on))
        assert os.environ[trial.C2_ENV] == str(int(on)) and parsed.b3_skip_icon_reading is on


# ------------------------------------------------------------------------------------ M3=0 の子の環境 (起動時の例外の予防)

def test_net_cols_without_inflight_stop_import_which_is_why_children_must_say_off() -> None:
    """予防の根拠: M3=0 のまま NET=1 だと列仕様の読込 (import) が例外で止まる。M3=0・NET=0 なら通る。"""
    code = 'import src.b3_column_spec as c; print(len(c.COLUMN_SPECS))'
    clean = {k: v for k, v in os.environ.items() if not k.startswith('PUYO_B3_')}
    for env, ok in (({'PUYO_B3_M3': '0', trial.NET_LEDGER_ENV: '1'}, False),
                    ({'PUYO_B3_M3': '0', trial.NET_LEDGER_ENV: '0'}, True)):
        run = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                             env={**clean, 'PYTHONPATH': str(ROOT), **env})
        assert (run.returncode == 0) is ok, run.stderr[-300:]


def test_create_baseline_says_off_to_the_v6_owner_and_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_b3
    seen: dict[str, str | None] = {}

    class Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.update({k: os.environ.get(k) for k in C2_ENV_KEYS})

    monkeypatch.setattr(live_b3, 'LiveB3', Stub)
    os.environ.update({'PUYO_B3_M3': '1', trial.NET_LEDGER_ENV: '1', trial.C2_ENV: '1'})
    trial.create_baseline(tmp_path/'table.json', tmp_path, fast_ema=False)
    assert (seen['PUYO_B3_M3'], seen[trial.NET_LEDGER_ENV], seen[trial.C2_ENV]) == ('0', '0', '0')
    assert (os.environ['PUYO_B3_M3'], os.environ[trial.NET_LEDGER_ENV], os.environ[trial.C2_ENV]) == ('1', '1', '1')


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, value: dict) -> None:
        self.sent.append(value)

    def recv(self) -> None:
        return None            # 入力の終わり (ループへ入らず終了する)

    def close(self) -> None:
        pass


class FakeThread:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass


def test_children_that_run_with_m3_off_also_turn_net_cols_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v4.9f の子と ama の子 (どちらも PUYO_B3_M3=0 で動く) は、親の NET=1 を受け継がず '0' を明示する。"""
    from src.phase_j import live_ama_async, live_v49f_async
    missing = (tmp_path/'none-a', tmp_path/'none-b', tmp_path/'none-c')
    monkeypatch.setattr(live_ama_async, 'Thread', FakeThread)
    monkeypatch.setattr('src.phase_j.live_obs_cpu.configure_ama', lambda *a, **k: None)
    for worker in (live_v49f_async.worker, live_ama_async.worker):
        os.environ.update({'PUYO_B3_M3': '1', trial.NET_LEDGER_ENV: '1'})
        connection = FakeConnection()
        worker(connection, missing, tmp_path)           # 資産が無い / 入力が無いので何もせず終わる。環境は設定済み
        assert connection.sent, worker                   # 子の本体まで到達した (ready か fatal を返した)
        assert (os.environ['PUYO_B3_M3'], os.environ[trial.NET_LEDGER_ENV]) == ('0', '0'), worker


# ------------------------------------------------------------------------------------ LiveB3 の列の照合 (A5)・記録・表示名

@functools.lru_cache(maxsize=None)
def spec_features(inflight: bool, net: bool) -> list[str]:
    """M3 (inflight) / 正味の列つきの列仕様から、学習の特徴列を返す。列仕様は読込時に決まるので環境を付けた子で作る。"""
    code = ('import json; from src.b3_column_spec import COLUMN_NAMES; from src.b3_train_model import feature_columns;'
            'print(json.dumps(feature_columns(COLUMN_NAMES)))')
    env = {k: v for k, v in os.environ.items() if not k.startswith('PUYO_B3_')}
    env.update(PYTHONPATH=str(ROOT), PUYO_B3_M3=str(int(inflight)), PUYO_B3_NET_LEDGER_COLS=str(int(net)))
    run = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-400:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def c2_columns() -> list[str]:
    return [c for c in spec_features(True, True) if c not in trial.C2_REMOVED_ICON_COLUMNS]


def fake_model(columns: list[str]) -> SimpleNamespace:
    evaluator = SimpleNamespace(columns=list(columns))
    return SimpleNamespace(static=evaluator, exchange=evaluator)


def build_live_b3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, columns: list[str],
                  model: Any = None, model_path: Path | None = None) -> Any:
    """仮のモデルで LiveB3 を作る (資産の読込と探索の本体は使わない。列の照合・設定の記録・表示名だけを見る)。"""
    from src import puyo_core_bridge as native
    from src.b3_timing import TimingTable
    from src.phase_j import live_b3
    monkeypatch.setattr(native, 'NATIVE_AVAILABLE', True)
    monkeypatch.setattr(native, '_native', SimpleNamespace(b3_search_py=None))
    if model_path is None:   # 実モデルを使うときは読込を差し替えない (本物の load_model と列の照合を通す)
        monkeypatch.setattr(live_b3, 'load_model', lambda path: model or fake_model(columns))
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path/'timing.json').write_text(TimingTable().to_json(), encoding='utf-8')
    out = tmp_path/'out'
    out.mkdir()
    return live_b3.LiveB3(model_path or make_model(tmp_path), tmp_path/'timing.json', out, trial_helpers=False), out


def set_trial_env(c2: bool) -> None:
    os.environ.update({'PUYO_B3_M3': '1', trial.C2_ENV: str(int(c2)), trial.NET_LEDGER_ENV: str(int(c2)),
                       trial.SKIP_ICON_ENV: str(int(c2))})


def test_c2_column_set_is_299_and_matches_the_removed_columns() -> None:
    columns = c2_columns()
    assert len(columns) == trial.C2_FEATURE_COLUMNS == 299 and len(set(columns)) == 299
    assert len(spec_features(True, False)) == 295 and len(spec_features(True, True)) == 312      # M3 と M3+正味 (C1)
    assert not set(trial.C2_REMOVED_ICON_COLUMNS) & set(columns) and len(trial.C2_REMOVED_ICON_COLUMNS) == 13
    assert set(trial.C2_REMOVED_ICON_COLUMNS) <= set(spec_features(True, False))
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    assert set(NET_LEDGER_FEATURE_NAMES) <= set(columns) and 'land_icon_drop_p1' in columns   # 着地量は外さない


def test_livea5_column_check_passes_for_c2_and_has_teeth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_trial_env(True)
    live, out = build_live_b3(tmp_path/'on', monkeypatch, c2_columns())
    live.close()
    record = json.loads((out/'b3_config.json').read_text(encoding='utf-8'))
    assert record['trial_model'] == 'C2' and record['feature_columns'] == 299
    assert record['net_ledger_cols'] is True and record['icon_reading'] is False
    assert (record['static_columns'], record['exchange_columns']) == (299, 299) and all(record['ledger_flags'].values())
    set_trial_env(True)
    os.environ[trial.NET_LEDGER_ENV] = '0'              # 正味の列を配送し損ねた構成では、列の照合が落ちる (歯がある)
    with pytest.raises(ValueError, match='特徴列が実装と一致しません'):
        build_live_b3(tmp_path/'off', monkeypatch, c2_columns())


def test_c2_model_guard_rejects_icon_columns_and_models_without_net_columns(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    set_trial_env(True)
    with pytest.raises(ValueError, match='アイコン由来列が残っています'):
        build_live_b3(tmp_path/'icon', monkeypatch, c2_columns()+[trial.C2_REMOVED_ICON_COLUMNS[0]])
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    with pytest.raises(ValueError, match='正味の列がありません'):
        build_live_b3(tmp_path/'nonet', monkeypatch, [c for c in c2_columns() if c not in NET_LEDGER_FEATURE_NAMES])


def test_m3_record_and_label_are_unchanged_by_c2_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.b3_live_display import B3Display
    from src.phase_j.live_b3 import LiveB3
    set_trial_env(False)
    os.environ[trial.SKIP_ICON_ENV] = '0'
    live, out = build_live_b3(tmp_path, monkeypatch, spec_features(True, False))     # M3 の列 (295) は検査を通る
    live.close()
    record = json.loads((out/'b3_config.json').read_text(encoding='utf-8'))
    assert list(record)[-3:] == ['trial_model', 'feature_columns', 'ledger_flags']   # 従来のキーと順序。C2 の欄は足さない
    assert (record['trial_model'], record['feature_columns']) == ('M3', 295)
    assert not {'net_ledger_cols', 'icon_reading', 'static_columns'} & set(record)
    for c2, label in ((False, '実戦勝率: B3 (M3)'), (True, '実戦勝率: B3 (C2)')):
        wrapper = LiveB3.__new__(LiveB3)
        wrapper.display = B3Display()
        wrapper.display.boundary(1)
        wrapper.display.prediction(.8, 0, 1.)
        wrapper.display.update(1.)
        os.environ.update({'PUYO_B3_M3': '1', trial.C2_ENV: str(int(c2))})
        row: dict = {}
        wrapper.decorate(row)
        assert row['b3_model_label'] == label
    os.environ['PUYO_B3_M3'] = '0'
    row = {}
    wrapper.decorate(row)
    assert 'b3_model_label' not in row


@pytest.mark.skipif(not Path('/mnt/c/'+M3_MODEL_NAME if os.name != 'nt' else 'C:/'+M3_MODEL_NAME).exists(),
                    reason='24c の配布の M3 モデルが無い')
def test_c2_columns_derived_from_the_real_m3_model_pass_the_check_with_the_expected_counts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """実 M3 モデルの列から PC2 の定義 (M3 − 13 + 正味) で C2 の列を作り、LiveB3 の照合を通す。列数は PREREG の見込み (static 222・exchange 295)。"""
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    from src.b3_train_model import load_model
    path = Path('/mnt/c/'+M3_MODEL_NAME if os.name != 'nt' else 'C:/'+M3_MODEL_NAME)
    m3 = load_model(path)
    removed, net = set(trial.C2_REMOVED_ICON_COLUMNS), list(NET_LEDGER_FEATURE_NAMES)
    static_net = [c for c in net if not c.startswith('ta_')]            # static では ta_*_slnet が全行欠測で落ちる (PREREG の note)
    static = [c for c in m3.static.columns if c not in removed]+static_net
    exchange = [c for c in m3.exchange.columns if c not in removed]+net
    assert (len(static), len(exchange)) == (222, 295)
    set_trial_env(True)
    model = SimpleNamespace(static=SimpleNamespace(columns=static), exchange=SimpleNamespace(columns=exchange))
    live, out = build_live_b3(tmp_path, monkeypatch, exchange, model)
    live.close()
    assert json.loads((out/'b3_config.json').read_text(encoding='utf-8'))['trial_model'] == 'C2'


# ------------------------------------------------------------------------------------ 本番用 C2 モデル (SHA 登録後)

needs_c2_model = pytest.mark.skipif(not C2_MODEL.exists(), reason='PC1 が照合済みの本番用 C2 モデルが無い')


@needs_c2_model
def test_registered_sha_is_the_delivered_c2_model_and_launch_check_accepts_it() -> None:
    """C2_SHA256 (差し替えの 1 か所) が、PC1 が照合したモデルのファイルと一致し、起動の照合が通る。M3 の SHA とは別。"""
    assert trial.C2_SHA256 == C2_REAL_SHA and trial.C2_SHA256 != trial.M3_SHA256
    assert sha256(C2_MODEL.read_bytes()).hexdigest() == C2_REAL_SHA
    trial.configure(options(C2_MODEL), argparse.ArgumentParser())          # monkeypatch なし = 本物の登録値で通る
    assert os.environ[trial.C2_ENV] == os.environ[trial.NET_LEDGER_ENV] == '1'


@needs_c2_model
def test_livea5_real_c2_model_passes_live_b3_column_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A5 (本物): 実 C2 モデルで LiveB3 の列の照合 (評価器の列 ⊂ 実装の列) と C2 の列の保険が通り、記録が正しい。"""
    from src.b3_train_model import load_model
    model = load_model(C2_MODEL)
    static, exchange = list(model.static.columns), list(model.exchange.columns)
    assert (len(static), len(exchange)) == (222, 295)                         # PREREG の見込み・LOCK_C2.json と同じ
    assert set(static) | set(exchange) <= set(c2_columns())                   # 使う列は C2 の 299 列の中だけ
    assert not (set(static) | set(exchange)) & set(trial.C2_REMOVED_ICON_COLUMNS)
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    assert set(NET_LEDGER_FEATURE_NAMES) <= set(exchange)                      # 正味の 17 列は exchange に全部ある
    assert sorted(c for c in set(static) | set(exchange) if 'icon' in c) == ['land_icon_drop_p1', 'land_icon_drop_p2']
    set_trial_env(True)
    live, out = build_live_b3(tmp_path, monkeypatch, [], model_path=C2_MODEL)
    live.close()
    record = json.loads((out/'b3_config.json').read_text(encoding='utf-8'))
    assert (record['trial_model'], record['feature_columns'], record['model_sha256']) == ('C2', 299, C2_REAL_SHA)
    assert (record['static_columns'], record['exchange_columns']) == (222, 295) and record['icon_reading'] is False
    set_trial_env(True)
    os.environ[trial.NET_LEDGER_ENV] = '0'                                      # 正味の列を配送し損ねると、実モデルでも落ちる
    with pytest.raises(ValueError, match='特徴列が実装と一致しません'):
        build_live_b3(tmp_path/'off', monkeypatch, [], model_path=C2_MODEL)


# ------------------------------------------------------------------------------------ 正味の 17 列: 学習行 = 配信の 1 行づくり

C106_ROWS = WIRING_DIR/'m3_rows_video_c106.parquet'    # dl1k の M3 学習行 (469 列・実データ)。PC2 の学習行と同じ形式


def same_column(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=True))


@pytest.mark.skipif(not C106_ROWS.exists(), reason='実データの M3 学習行 (video_c106) が無い')
def test_net_columns_of_real_training_rows_equal_the_live_row_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """実データの学習行で、学習行へ列を足す台本の関数 (derive_frame) と、配信の 1 行づくり (build_features の
    net_ledger_group) が、名前・順序・dtype・値の全てで一致する。入力列は全て f8 / i1 なので parquet でも欠けない。"""
    import pandas as pd
    from src import b3_column_spec
    from src.b3_features import net_ledger_group
    from src.b3_net_ledger import SOURCE_COLUMNS, derive_frame
    df = pd.read_parquet(C106_ROWS)
    assert len(df) > 1000 and set(df['orientation']) == {0, 1}
    assert not set(b3_column_spec.NET_LEDGER_COLUMN_NAMES) & set(df.columns)
    script = derive_frame(df)
    names = list(b3_column_spec.NET_LEDGER_COLUMN_NAMES)
    assert list(script.columns) == names and len(names) == 32                  # 学習行へ足した 32 列の名前と順序
    specs = {c.name: c for c in b3_column_spec.net_ledger_specs()}
    monkeypatch.setattr(b3_column_spec, 'SPEC_BY_NAME', {**b3_column_spec.SPEC_BY_NAME, **specs})   # 案A ON の列仕様
    src = {k: df[k].to_numpy(dtype=np.float64, na_value=np.nan) for k in SOURCE_COLUMNS}
    assert not any(np.isnan(src[k]).any() for k in SOURCE_COLUMNS if k.startswith('mk_'))          # mask に欠測は無い
    live: dict[str, list] = {name: [] for name in names}
    for i in range(len(df)):
        row = {k: (int(src[k][i]) if k.startswith('mk_') else float(src[k][i])) for k in SOURCE_COLUMNS}
        net_ledger_group(row)
        for name in names:
            live[name].append(row[name])
    dtypes = {'f8': np.float64, 'f4': np.float32, 'i1': np.float64}      # i1 は欠測 (<NA>) があるので float で比べる
    bad = [name for name in names if not same_column(np.asarray(live[name], dtype=dtypes[specs[name].dtype]),
                                                     script[name].to_numpy(dtype=dtypes[specs[name].dtype], na_value=np.nan))]
    assert not bad, bad
    finite = {name: int(np.isfinite(np.asarray(live[name], dtype=np.float64)).sum()) for name in names
              if not name.startswith('mk_')}
    assert finite['net_pending_p1'] > 1000 and finite['ta_m_slnet_med_p1'] > 100, finite   # 母数: 全部欠測の一致ではない


# ------------------------------------------------------------------------------------ A1 (重い): アイコンあり / icon=None でビット一致

A1_VIDEO, A1_FRAMES = 'video_51', 12000
A1_ROOT = Path(os.environ.get('B3_C2_A1_B2_ROOT', WIRING_DIR/'b2_root'))   # video_51 の採用記録の写し (WSL から D: が見えないため C: に置く)
A1_CHILD = '''
import pickle, sys
from pathlib import Path
from src.b3_source import resolve_attempt
from src.b3_rows import generate_rows
from src.b3_timing import TimingTable
mode, video, frames, root, out = sys.argv[1:6]
if mode == 'noicon':
    from src.b3_state import IconCorrector
    IconCorrector.push = lambda self, t, icon, gate: None     # icon=None の frame と同じ (push は None で何もしない)
p = resolve_attempt(video, Path(root))
rows, info = generate_rows(p.record, video, TimingTable(), max_frame=int(frames), require_winner=False)
pickle.dump((rows, info), open(out, 'wb'))
'''
EQUIV_CHILD = '''
import json, pickle, sys
import numpy as np, pandas as pd
from src.b3_rows import finalize_weights, to_frame, with_mirrors
from src.b3_column_spec import NET_LEDGER_COLUMN_NAMES
from src.b3_net_ledger import derive_frame
live, m3 = pickle.load(open(sys.argv[1], 'rb'))[0], pickle.load(open(sys.argv[2], 'rb'))[0]
names = list(NET_LEDGER_COLUMN_NAMES)
def same(a, b):
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return a == b
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True))
existing, extra = {}, set()
assert len(live) == len(m3) and all(a['row_key'] == b['row_key'] for a, b in zip(live, m3))
for a, b in zip(live, m3):
    extra |= set(a) - set(b)
    for k, v in b.items():
        if not same(v, a[k]):
            existing[k] = existing.get(k, 0) + 1
finalize_weights(live)
df = to_frame(with_mirrors(live))
df.drop(columns=names).to_parquet(sys.argv[3], engine='pyarrow', compression='zstd', index=False)   # 学習行と同じ保存
script = derive_frame(pd.read_parquet(sys.argv[3]))
pd.testing.assert_frame_equal(script, df[names], check_exact=True)                    # 名前・順序・dtype・値
print(json.dumps(dict(rows_live=len(live), rows_m3=len(m3), added_keys_ok=extra == set(names), existing_diffs=existing,
                      order_ok=list(df.columns[-len(names):]) == names, frame_rows=len(df),
                      finite_net_pending=int(df['net_pending_p1'].notna().sum()),
                      finite_ta_m=int(df['ta_m_slnet_med_p1'].notna().sum()))))
'''
M3_NET_ENV = dict(PUYO_B3_M3='1', PUYO_B3_NET_LEDGER_COLS='1')      # 配信の AsyncB3 の子と同じ環境
M3_ENV = dict(PUYO_B3_M3='1')                                        # 学習行 (M1_rows) と同じ: 5 フラグ = M3、正味の列は無い


def run_child(code: str, args: list[str], **env: str) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith('PUYO_B3_')}
    run = subprocess.run([sys.executable, '-c', code, *args], cwd=ROOT, env={**clean, 'PYTHONPATH': str(ROOT), **env},
                         capture_output=True, text=True, timeout=3600)
    assert run.returncode == 0, run.stderr[-800:]
    return run


def replay(mode: str, out: Path, **env: str) -> Path:
    run_child(A1_CHILD, [mode, A1_VIDEO, str(A1_FRAMES), str(A1_ROOT), str(out)], **env)
    return out


needs_record = pytest.mark.skipif(not (A1_ROOT/'receipts'/f'{A1_VIDEO}.json').exists(),
                                  reason='video_51 の採用記録が無い (B3_C2_A1_B2_ROOT か b3_c2_wiring_20261025/b2_root)')


def same_value(a: Any, b: Any) -> bool:
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return a == b
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True))


@pytest.fixture(scope='module')
def replays(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """3 回の再生 (重い): 配信と同じ環境でアイコンあり・icon=None、学習行と同じ環境 (M3 のみ) でアイコンあり。"""
    root = tmp_path_factory.mktemp('replays')
    return dict(icon=replay('icon', root/'icon.pkl', **M3_NET_ENV), noicon=replay('noicon', root/'noicon.pkl', **M3_NET_ENV),
                m3=replay('icon', root/'m3.pkl', **M3_ENV), root=root)


@needs_record
def test_livea1_c2_columns_are_bit_identical_with_and_without_icon(replays: dict[str, Path]) -> None:
    """C2 の 299 列 (land_icon_drop_p1/p2 を含む) は、予告アイコンを読まなくても同じコマで 1 ビットも変わらない。

    行の集合は変わる (アイコン変化のきっかけ bit16 が無くなる) ので、両方にある同じコマだけを比べる。
    差が出た列は名前と件数を表示して落とす (学習はアイコンありの行、配信はアイコン無しの行になるずれの検出)。
    """
    with_icon, without = (pickle.loads(replays[k].read_bytes())[0] for k in ('icon', 'noicon'))
    by_frame = {(r['game_idx'], r['frame_idx']): r for r in without}
    common = [(r, by_frame[(r['game_idx'], r['frame_idx'])]) for r in with_icon
              if (r['game_idx'], r['frame_idx']) in by_frame]
    assert len(with_icon) > 200 and len(common) > len(with_icon)//2, (len(with_icon), len(without), len(common))
    differing: dict[str, int] = {}
    for a, b in common:
        for name in c2_columns():
            if not same_value(a[name], b[name]):
                differing[name] = differing.get(name, 0)+1
    assert not differing, dict(common_rows=len(common), differing=differing)
    landing = {n: sum(bool(np.isfinite(a[n])) for a, _ in common) for n in ('land_icon_drop_p1', 'land_icon_drop_p2')}
    print('A1', dict(rows_icon=len(with_icon), rows_noicon=len(without), common=len(common), land_icon_drop_finite=landing))


@needs_record
def test_live_built_rows_equal_training_path_rows_on_real_record(replays: dict[str, Path]) -> None:
    """video_51 の実記録で、配信の 1 行づくり (M3+正味の列) の行 = 学習行 (M3 の行 + 台本で足した 32 列、parquet 経由)。

    既存の 469 列は 1 ビットも変わらず (正味の列を足しても不変)、足した 32 列は名前・順序・dtype・値が一致する。
    """
    args = [str(replays['icon']), str(replays['m3']), str(replays['root']/'base.parquet')]
    result = json.loads(run_child(EQUIV_CHILD, args, **M3_NET_ENV).stdout.strip().splitlines()[-1])
    assert result['rows_live'] == result['rows_m3'] > 200 and result['added_keys_ok'] and result['order_ok'], result
    assert not result['existing_diffs'], result
    assert result['finite_net_pending'] > 200 and result['finite_ta_m'] > 100, result            # 母数
