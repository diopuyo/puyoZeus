"""b3_ensemble (複数の TwoStateModel の確率を平均する入れ物。既定OFF) の試験。段 1 (単体) だけ。

  (a) OFF の同一性 (旗・環境・b3_config.json・子 process のモデルの読み方・launcher の出力)
  (b) 同じモデルを並べた出力が 1 本の出力とビット一致 (mean_p / mean_logit、1 本だけの指定も)
  (c) 違うモデルなら平均になる (mean_p は算術平均、mean_logit は logit の平均)
  (d) 列の順序・欠測の扱いが 1 本と同じ (欠測の特徴・mode の欠測と範囲外・評価器の欠け・列の食い違いの拒否)
  マニフェスト・SHA256 の固定・旗の配線 (launcher → CLI → configure → LiveB3 / b3_async の子)
  実データ: 本番用 C2 モデルと dl1k の実行 (ファイルがあるときだけ)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import b3_ensemble as ens
from src.b3_train_model import Evaluator, HgbConfig, TwoStateModel, load_model, save_model
from src.phase_j import live_obs_trial as trial
from src.phase_j.launcher import LauncherConfigError, UserConfig, build_pipeline_config, parse_user_config

N_FEATURES = 6
STATIC_COLUMNS = [f'f{i}' for i in range(N_FEATURES - 1)]
EXCHANGE_COLUMNS = [f'f{i}' for i in range(N_FEATURES)]
N_TRAIN, N_ROWS, TREES = 500, 150, 25
NAN_RATE = 0.1
C2_MODEL = Path('C:/puyo1_work/b3_c2_model_20261025/large_seed1003.joblib' if os.name == 'nt'
                else '/mnt/c/puyo1_work/b3_c2_model_20261025/large_seed1003.joblib')
OTHER_SHA = sha256(b'another-model').hexdigest()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure が環境変数を書き換えるので、試験ごとに写しへ差し替える (PUYO_B3_ の持ち込みも消す)。"""
    monkeypatch.setattr(os, 'environ', {k: v for k, v in os.environ.items() if not k.startswith('PUYO_B3_')})


# ------------------------------------------------------------------------------------ 小さな本物の TwoStateModel

def fit_evaluator(name: str, columns: list[str], seed: int) -> Evaluator:
    """合成データで学習した本物の HGB 評価器 (seed を変えるとデータも木も変わる = 違うモデル)。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N_TRAIN, len(columns))).astype(np.float32)
    x[rng.random(x.shape) < NAN_RATE] = np.nan
    y = (np.nan_to_num(x[:, 0]) + 0.5 * np.nan_to_num(x[:, 1]) + rng.normal(scale=0.7, size=N_TRAIN) > 0).astype(float)
    model = HistGradientBoostingClassifier(max_iter=TREES, random_state=seed).fit(x, y)
    return Evaluator(name, model, list(columns), [], N_TRAIN, HgbConfig('t', TREES, 0.1, 15, 5))


def make_model(seed: int, static: bool = True, exchange: bool = True) -> TwoStateModel:
    return TwoStateModel(fit_evaluator('static', STATIC_COLUMNS, seed) if static else None,
                         fit_evaluator('exchange', EXCHANGE_COLUMNS, seed + 1000) if exchange else None)


def make_rows(seed: int = 7, n: int = N_ROWS, modes: tuple[float, ...] = (0.0, 1.0)) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row = {c: float(rng.normal()) for c in EXCHANGE_COLUMNS}
        for column in EXCHANGE_COLUMNS:
            if rng.random() < NAN_RATE:
                row[column] = float('nan')                  # 欠測の特徴 (HGB が NaN をそのまま分割に使う)
        row[ens_mode_column()] = float(rng.choice(modes))
        rows.append(row)
    return rows


def ens_mode_column() -> str:
    from src.b3_train_model import MODE_COLUMN
    return MODE_COLUMN


@pytest.fixture(scope='module')
def models() -> tuple[TwoStateModel, TwoStateModel, TwoStateModel]:
    return make_model(1), make_model(2), make_model(3)


@pytest.fixture(scope='module')
def rows() -> list[dict[str, float]]:
    return make_rows()


def bits(value: float) -> bytes:
    return struct.pack('d', value)


def same_bits_all(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(bits(x) == bits(y) for x, y in zip(a, b))


# ------------------------------------------------------------------------------------ 平均の関数

@pytest.mark.parametrize('aggregate', ens.AGGREGATES)
def test_equal_members_return_the_value_itself_bit_for_bit(aggregate: str) -> None:
    """素朴な平均は同じ値 3 つでも最後の桁がずれる。全メンバーが同じ値の行はその値をそのまま返す。"""
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    for count in (1, 2, 3, 5):
        out = ens.aggregate_probabilities(np.stack([p] * count), aggregate)
        assert out.dtype == np.float64 and np.array_equal(out, p)
    naive = np.stack([p] * 3).mean(axis=0)
    assert (naive != p).any()                 # 歯: 素朴な平均ではこの母集団でずれる (規則が要る理由)


def test_mean_p_is_the_arithmetic_mean_and_mean_logit_is_the_logit_mean() -> None:
    stack = np.array([[0.2, 0.9, 0.5], [0.4, 0.7, 0.5], [0.9, 0.1, 0.5]])
    np.testing.assert_allclose(ens.aggregate_probabilities(stack, ens.AGG_MEAN_P), stack.mean(axis=0), rtol=1e-15)
    z = np.log(stack / (1 - stack)).mean(axis=0)
    expected = 1 / (1 + np.exp(-z))
    np.testing.assert_allclose(ens.aggregate_probabilities(stack, ens.AGG_MEAN_LOGIT), expected, rtol=1e-12)
    assert ens.aggregate_probabilities(stack, ens.AGG_MEAN_P)[0] != ens.aggregate_probabilities(stack, ens.AGG_MEAN_LOGIT)[0]


def pc2_logit_mean(x: np.ndarray) -> np.ndarray:
    """PC2 の make_ens_oof.py (D:/puyo_analyzer/share/tasks/done/c2_seed_ensemble_eval_20261025) の p_logit と同じ式。"""
    eps = 1e-15
    c = np.clip(x, eps, 1 - eps)
    return 1 / (1 + np.exp(-np.log(c / (1 - c)).mean(axis=1)))


def test_logit_mean_is_the_default_and_matches_the_pc2_formula_bit_for_bit() -> None:
    """既定は logit の平均。式と切り幅は、保存 OOF で LL を測った PC2 の台本と、3 本が違う行で 1 ビットも違わない。

    差が出るのは 3 本が完全に同じ値の行だけ (配信は同じ値をそのまま返す。PC2 の式は切り幅の分 1e-15 だけ内側へ戻る)。
    """
    assert ens.DEFAULT_AGGREGATE == ens.AGG_MEAN_LOGIT and ens.LOGIT_CLIP == 1e-15
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.random((100_000, 3)), rng.random((1000, 3)) ** 40, 1 - rng.random((1000, 3)) ** 40], axis=0)
    expected, default = pc2_logit_mean(x), ens.aggregate_probabilities(x.T)
    equal_rows = (x == x[:, :1]).all(axis=1)
    assert equal_rows.sum() > 10 and (~equal_rows).sum() > 100_000                                         # 母数: 両方の行がある
    assert np.array_equal(default[~equal_rows], expected[~equal_rows])                                    # 違う値の行は 1 ビットも違わない
    assert np.array_equal(default[equal_rows], x[equal_rows, 0])                                          # 同じ値の行はその値
    assert float(np.abs(default - expected).max()) <= 2e-15
    assert np.array_equal(ens.aggregate_probabilities(x.T, ens.AGG_MEAN_LOGIT), default)
    mean_p = ens.aggregate_probabilities(x.T, ens.AGG_MEAN_P)
    np.testing.assert_allclose(mean_p, x.mean(axis=1), rtol=1e-15)                                        # 切り替えた先は算術平均
    assert float(np.abs(default - mean_p).max()) > 1e-3                                                   # 母数: 2 つの平均は違う値を出す


def test_default_aggregate_is_logit_in_the_class_and_the_manifest_builder(tmp_path: Path, models: Any) -> None:
    assert ens.EnsembleTwoStateModel([models[0]]).aggregate == ens.AGG_MEAN_LOGIT
    path = tmp_path/'m.joblib'
    path.write_bytes(b'x')
    assert ens.build_manifest([path])['aggregate'] == ens.AGG_MEAN_LOGIT


def test_logit_mean_stays_finite_at_zero_and_one_and_nan_propagates() -> None:
    stack = np.array([[0.0, 1.0, np.nan], [1.0, 1.0, 0.3]])
    out = ens.aggregate_probabilities(stack, ens.AGG_MEAN_LOGIT)
    assert np.isfinite(out[:2]).all() and 0.0 <= out[:2].min() and out[:2].max() <= 1.0 and out[1] == 1.0
    assert np.isnan(out[2]) and np.isnan(ens.aggregate_probabilities(stack, ens.AGG_MEAN_P)[2])


def test_unknown_aggregate_is_rejected_by_name() -> None:
    with pytest.raises(ens.EnsembleError, match='mean_p'):
        ens.aggregate_probabilities(np.zeros((2, 3)), 'median')


# ------------------------------------------------------------------------------------ (b) 同じモデル = 1 本とビット一致

@pytest.mark.parametrize('aggregate', ens.AGGREGATES)
@pytest.mark.parametrize('count', (1, 3))
def test_b_same_model_repeated_is_bit_identical_to_single(models: Any, rows: list[dict], aggregate: str, count: int) -> None:
    single = [models[0].predict_row(row) for row in rows]
    ensemble = ens.EnsembleTwoStateModel([models[0]] * count, aggregate)
    assert same_bits_all(single, [ensemble.predict_row(row) for row in rows])
    assert len({round(p, 6) for p in single}) > 20 and all(0 <= p <= 1 for p in single)      # 母数: 値は散らばっている


def test_b_separately_loaded_copies_are_bit_identical_to_single(tmp_path: Path, models: Any, rows: list[dict]) -> None:
    """別々に読み込んだ (別 object の) 同じモデル 3 本でも 1 本と一致する (RowPredictor は各自が持つ)。"""
    path = tmp_path/'m.joblib'
    save_model(models[0], path)
    members = [load_model(path) for _ in range(3)]
    assert len({id(m) for m in members}) == 3
    ensemble = ens.EnsembleTwoStateModel(members)
    assert same_bits_all([members[0].predict_row(r) for r in rows], [ensemble.predict_row(r) for r in rows])


def test_b_bulk_predict_is_bit_identical_to_single_and_to_the_row_path(models: Any, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    single = models[0].predict(frame)
    ensemble = ens.EnsembleTwoStateModel([models[0]] * 3)
    assert same_bits_all(list(single), list(ensemble.predict(frame)))
    assert same_bits_all(list(ensemble.predict(frame)), [ensemble.predict_row(r) for r in rows])


# ------------------------------------------------------------------------------------ (c) 違うモデル = 平均

def test_c_different_models_give_the_arithmetic_mean_of_the_single_outputs(models: Any, rows: list[dict]) -> None:
    ensemble = ens.EnsembleTwoStateModel(list(models), ens.AGG_MEAN_P)
    for row in rows:
        singles = [m.predict_row(row) for m in models]
        assert abs(ensemble.predict_row(row) - float(np.mean(singles))) <= 4e-16
    spread = [max(m.predict_row(r) for m in models) - min(m.predict_row(r) for m in models) for r in rows]
    assert float(np.mean(spread)) > 0.05 and max(spread) > 0.2        # 母数: モデル同士は実際に違う (平均の意味がある)


def test_c_logit_mode_averages_the_logits_and_differs_from_mean_p(models: Any, rows: list[dict]) -> None:
    mean_p, mean_logit = (ens.EnsembleTwoStateModel(list(models), a) for a in ens.AGGREGATES)
    diffs = []
    for row in rows:
        singles = np.array([m.predict_row(row) for m in models])
        expected = 1 / (1 + np.exp(-np.log(singles / (1 - singles)).mean()))
        assert abs(mean_logit.predict_row(row) - expected) <= 1e-12
        diffs.append(abs(mean_logit.predict_row(row) - mean_p.predict_row(row)))
    assert max(diffs) > 1e-4                                          # 母数: 2 つの平均は実際に違う値を出す


@pytest.mark.parametrize('aggregate', ens.AGGREGATES)
def test_c_order_of_members_does_not_matter_beyond_the_last_digits(models: Any, rows: list[dict], aggregate: str) -> None:
    forward = ens.EnsembleTwoStateModel(list(models), aggregate)
    backward = ens.EnsembleTwoStateModel(list(models)[::-1], aggregate)
    assert max(abs(forward.predict_row(r) - backward.predict_row(r)) for r in rows) <= 1e-14


# ------------------------------------------------------------------------------------ (d) 列の順序・欠測の扱い

def test_d_row_key_order_extra_keys_and_missing_features_behave_like_single(models: Any, rows: list[dict]) -> None:
    ensemble = ens.EnsembleTwoStateModel([models[0]] * 3)
    for row in rows[:40]:
        shuffled = dict(reversed(list(row.items())))
        noisy = dict(shuffled, unrelated=123.0)
        assert bits(ensemble.predict_row(row)) == bits(ensemble.predict_row(shuffled)) == bits(ensemble.predict_row(noisy))
        assert bits(ensemble.predict_row(row)) == bits(models[0].predict_row(row))
    assert sum(any(np.isnan(v) for v in row.values()) for row in rows) > 40       # 母数: 欠測の特徴を含む行が多い


def test_d_a_feature_missing_from_the_row_raises_like_single(models: Any, rows: list[dict]) -> None:
    ensemble = ens.EnsembleTwoStateModel([models[0]] * 3)
    broken = {k: v for k, v in rows[0].items() if k != 'f2'}
    with pytest.raises(KeyError) as single_error:
        models[0].predict_row(broken)
    with pytest.raises(KeyError) as ensemble_error:
        ensemble.predict_row(broken)
    assert str(single_error.value) == str(ensemble_error.value)


def test_d_mode_missing_out_of_range_and_absent_evaluator_behave_like_single(models: Any, rows: list[dict]) -> None:
    mode = ens_mode_column()
    ensemble = ens.EnsembleTwoStateModel([models[0]] * 3)
    with pytest.raises(ValueError, match='mode_exchange に欠測') as single_error:
        models[0].predict_row(dict(rows[0], **{mode: float('nan')}))
    with pytest.raises(ValueError, match='mode_exchange に欠測') as ensemble_error:
        ensemble.predict_row(dict(rows[0], **{mode: float('nan')}))
    assert str(single_error.value) == str(ensemble_error.value)
    assert np.isnan(models[0].predict_row(dict(rows[0], **{mode: 0.5}))) and np.isnan(
        ensemble.predict_row(dict(rows[0], **{mode: 0.5})))                   # 範囲外の mode は NaN (従来どおり)
    no_exchange = make_model(5, exchange=False)
    partial = ens.EnsembleTwoStateModel([no_exchange, no_exchange])
    with pytest.raises(ValueError, match='mode=1 の行があるが評価器が無い'):
        partial.predict_row(dict(rows[0], **{mode: 1.0}))
    assert bits(partial.predict_row(dict(rows[0], **{mode: 0.0}))) == bits(no_exchange.predict_row(dict(rows[0], **{mode: 0.0})))


def test_d_members_with_different_columns_or_evaluators_are_rejected(models: Any) -> None:
    reordered = make_model(9)
    reordered.exchange = Evaluator('exchange', reordered.exchange.model, EXCHANGE_COLUMNS[::-1], [], N_TRAIN,
                                   reordered.exchange.cfg)
    with pytest.raises(ens.EnsembleError, match='1 番目の特徴列'):
        ens.EnsembleTwoStateModel([models[0], reordered])
    with pytest.raises(ens.EnsembleError, match='1 番目の特徴列'):
        ens.EnsembleTwoStateModel([models[0], make_model(9, static=False)])
    with pytest.raises(ens.EnsembleError, match='0 本'):
        ens.EnsembleTwoStateModel([])
    with pytest.raises(ens.EnsembleError, match='mean_p'):
        ens.EnsembleTwoStateModel([models[0]], 'median')


def test_ensemble_exposes_the_columns_of_the_members_for_the_live_column_check(models: Any) -> None:
    ensemble = ens.EnsembleTwoStateModel(list(models))
    assert ensemble.static is models[0].static and ensemble.exchange is models[0].exchange
    assert list(ensemble.static.columns) == STATIC_COLUMNS and list(ensemble.exchange.columns) == EXCHANGE_COLUMNS


# ------------------------------------------------------------------------------------ マニフェスト

def write_models(folder: Path, models: Any) -> list[Path]:
    paths = []
    for index, model in enumerate(models):
        paths.append(folder/f'seed{1003+index}.joblib')
        save_model(model, paths[-1])
    return paths


def write_manifest(folder: Path, paths: list[Path], aggregate: str = ens.AGG_MEAN_P) -> Path:
    manifest = folder/'ensemble.json'
    manifest.write_text(json.dumps(ens.build_manifest(paths, aggregate)), encoding='utf-8')
    return manifest


def test_manifest_round_trip_loads_the_ensemble_and_matches_the_in_memory_one(tmp_path: Path, models: Any,
                                                                               rows: list[dict]) -> None:
    manifest = write_manifest(tmp_path, write_models(tmp_path, models), ens.AGG_MEAN_LOGIT)
    loaded = ens.load_ensemble_model(manifest)
    assert loaded.aggregate == ens.AGG_MEAN_LOGIT and len(loaded.members) == 3
    direct = ens.EnsembleTwoStateModel(list(models), ens.AGG_MEAN_LOGIT)
    assert same_bits_all([direct.predict_row(r) for r in rows], [loaded.predict_row(r) for r in rows])
    record = ens.ensemble_record(loaded)['ensemble']
    assert [m['name'] for m in record['members']] == ['seed1003.joblib', 'seed1004.joblib', 'seed1005.joblib']
    assert record['aggregate'] == ens.AGG_MEAN_LOGIT and record['n_members'] == 3
    assert [m['sha256'] for m in record['members']] == [sha256(p.read_bytes()).hexdigest() for p in sorted(tmp_path.glob('seed*'))]


def test_manifest_with_a_single_model_is_the_replace_with_one_model_option(tmp_path: Path, models: Any,
                                                                            rows: list[dict]) -> None:
    manifest = write_manifest(tmp_path, write_models(tmp_path, models[:1]))
    loaded = ens.load_ensemble_model(manifest)
    assert len(loaded.members) == 1
    assert same_bits_all([models[0].predict_row(r) for r in rows], [loaded.predict_row(r) for r in rows])


def test_a_swapped_model_stops_the_launch_and_names_the_file(tmp_path: Path, models: Any) -> None:
    paths = write_models(tmp_path, models)
    manifest = write_manifest(tmp_path, paths)
    swap = tmp_path/'swap.joblib'
    save_model(make_model(77), swap)
    paths[1].write_bytes(swap.read_bytes())                       # seed1004 を別のモデルに差し替える
    with pytest.raises(ens.EnsembleError, match='seed1004.joblib'):
        ens.load_ensemble_model(manifest)
    paths[1].unlink()
    with pytest.raises(ens.EnsembleError, match='seed1004.joblib'):
        ens.verify_member_files(ens.read_manifest(manifest))


@pytest.mark.parametrize('mutate,word', [
    (lambda d: d.update(format='b3_ensemble_v0'), 'format'),
    (lambda d: d.update(aggregate='median'), 'aggregate'),
    (lambda d: d.update(members=[]), 'members'),
    (lambda d: d.update(extra=1), '鍵は'),
    (lambda d: d['members'][0].update(name='../x.joblib'), 'ファイル名'),
    (lambda d: d['members'][0].update(name='sub/x.joblib'), 'ファイル名'),
    (lambda d: d['members'][0].update(sha256='ABC'), '64 桁'),
    (lambda d: d['members'][1].update(name=d['members'][0]['name']), '重複'),
    (lambda d: d['members'][0].update(extra=1), 'name'),
])
def test_malformed_manifest_is_rejected_with_a_named_reason(tmp_path: Path, mutate: Any, word: str) -> None:
    paths = []
    for index in range(2):
        paths.append(tmp_path/f'm{index}.joblib')
        paths[-1].write_bytes(f'model-{index}'.encode())
    data = ens.build_manifest(paths)
    mutate(data)
    manifest = tmp_path/'ensemble.json'
    manifest.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ens.EnsembleError, match=word):
        ens.read_manifest(manifest)


def test_manifest_that_is_not_json_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path/'ensemble.json'
    broken.write_bytes(b'\x80 not json')
    with pytest.raises(ens.EnsembleError, match='JSON として読めません'):
        ens.read_manifest(broken)
    with pytest.raises(ens.EnsembleError, match='JSON として読めません'):
        ens.read_manifest(tmp_path/'none.json')


def test_build_manifest_requires_one_folder_and_a_known_aggregate(tmp_path: Path) -> None:
    (tmp_path/'a').mkdir()
    (tmp_path/'b').mkdir()
    paths = [tmp_path/'a'/'x.joblib', tmp_path/'b'/'y.joblib']
    for path in paths:
        path.write_bytes(b'x')
    with pytest.raises(ens.EnsembleError, match='同じフォルダ'):
        ens.build_manifest(paths)
    with pytest.raises(ens.EnsembleError, match='mean_p'):
        ens.build_manifest(paths[:1], 'median')
    with pytest.raises(ens.EnsembleError, match='0 本'):
        ens.build_manifest([])


def test_manifest_script_writes_a_loadable_manifest_and_refuses_to_overwrite(tmp_path: Path, models: Any, capsys: pytest.CaptureFixture) -> None:
    from scripts import make_b3_ensemble_manifest as script
    paths = write_models(tmp_path, models)
    out = tmp_path/'ensemble.json'
    assert script.main(['--out', str(out), '--aggregate', 'mean_p', *map(str, paths)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['manifest_sha256'] == sha256(out.read_bytes()).hexdigest() and printed['members'] == 3 and printed['aggregate'] == 'mean_p'
    assert (printed['static_columns'], printed['exchange_columns']) == (len(STATIC_COLUMNS), len(EXCHANGE_COLUMNS))
    assert ens.load_ensemble_model(out).aggregate == ens.AGG_MEAN_P
    with pytest.raises(FileExistsError):
        script.main(['--out', str(out), *map(str, paths)])
    (tmp_path/'other').mkdir()
    with pytest.raises(SystemExit, match='同じフォルダ'):
        script.main(['--out', str(tmp_path/'other'/'ensemble.json'), *map(str, paths)])


# ------------------------------------------------------------------------------------ (a) OFF の同一性と旗の配線

def options(manifest: Path, **over: Any) -> SimpleNamespace:
    values = dict(b3_judgment=True, b3_async=True, b3_m3=True, b3_time_gate=True, live_ama_plain=True,
                  live_score_cadence=True, live_static_exact=True, b3_c2=True, b3_skip_icon_reading=True,
                  b3_ensemble=True, b3_model=manifest)
    values.update(over)
    return SimpleNamespace(**values)


def configure_error(opts: SimpleNamespace, capsys: pytest.CaptureFixture) -> str:
    with pytest.raises(SystemExit):
        trial.configure(opts, argparse.ArgumentParser())
    return capsys.readouterr().err


def test_env_name_default_and_label_are_single_source() -> None:
    from src import production_config
    assert trial.ENSEMBLE_ENV == ens.ENSEMBLE_ENV == 'PUYO_B3_ENSEMBLE'
    assert production_config.B3_ENSEMBLE_DEFAULT is False and UserConfig.__dataclass_fields__['b3_ensemble'].default is False
    assert trial.C2_ENSEMBLE_SHA256 == '' or len(trial.C2_ENSEMBLE_SHA256) == trial.C2_SHA256_LENGTH
    assert ens.ensemble_enabled() is False
    os.environ[ens.ENSEMBLE_ENV] = '1'
    assert ens.ensemble_enabled() is True
    os.environ[ens.ENSEMBLE_ENV] = 'true'
    assert ens.ensemble_enabled() is False                         # '1' だけが ON


def test_off_configure_sets_env_zero_even_when_the_parent_env_says_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path/'model.joblib'
    model.write_bytes(b'c2')
    monkeypatch.setattr(trial, 'C2_SHA256', sha256(b'c2').hexdigest())
    os.environ[ens.ENSEMBLE_ENV] = '1'                             # 親の環境の持ち込みは OFF が '0' で上書きする
    trial.configure(options(model, b3_ensemble=False), argparse.ArgumentParser())
    assert os.environ[ens.ENSEMBLE_ENV] == '0' and not ens.ensemble_enabled()
    trial.configure(SimpleNamespace(b3_judgment=True, b3_async=True), argparse.ArgumentParser())   # b3_ensemble 属性なしでも従来どおり
    assert os.environ[ens.ENSEMBLE_ENV] == '0'


def test_on_requires_c2_a_bool_and_a_registered_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    paths = []
    for index in range(2):
        paths.append(tmp_path/f'm{index}.joblib')
        paths[-1].write_bytes(f'model-{index}'.encode())
    manifest = write_manifest(tmp_path, paths)
    monkeypatch.setattr(trial, 'C2_ENSEMBLE_SHA256', '')
    assert 'C2アンサンブルのマニフェストのSHA256が未登録' in configure_error(options(manifest), capsys)
    monkeypatch.setattr(trial, 'C2_ENSEMBLE_SHA256', OTHER_SHA)
    assert 'マニフェストの固定SHA256が不一致です' in configure_error(options(manifest), capsys)
    monkeypatch.setattr(trial, 'C2_ENSEMBLE_SHA256', sha256(manifest.read_bytes()).hexdigest())
    for over, word in ((dict(b3_c2=False, b3_skip_icon_reading=False), 'b3_c2=true'), (dict(b3_ensemble='true'), 'b3_ensembleはboolです'),
                       (dict(b3_m3=False), 'b3_m3')):
        assert word in configure_error(options(manifest, **over), capsys), over
    trial.configure(options(manifest), argparse.ArgumentParser())
    assert os.environ[ens.ENSEMBLE_ENV] == os.environ[trial.C2_ENV] == '1' and ens.ensemble_enabled()


def test_on_names_the_swapped_member_at_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture) -> None:
    paths = []
    for index in range(3):
        paths.append(tmp_path/f'seed{1003+index}.joblib')
        paths[-1].write_bytes(f'model-{index}'.encode())
    manifest = write_manifest(tmp_path, paths)
    monkeypatch.setattr(trial, 'C2_ENSEMBLE_SHA256', sha256(manifest.read_bytes()).hexdigest())
    trial.configure(options(manifest), argparse.ArgumentParser())          # 差し替え前は通る
    paths[2].write_bytes(b'tampered')
    assert 'seed1005.joblib' in configure_error(options(manifest), capsys)
    paths[2].unlink()
    assert 'seed1005.joblib' in configure_error(options(manifest), capsys)


def test_launcher_off_has_no_ensemble_key_and_on_needs_c2(tmp_path: Path) -> None:
    base = dict(source='dshow', device_name='OBS Virtual Camera', b3_judgment=True, b3_async=True, live_v49f=True,
                b3_m3=True, b3_time_gate=True, b3_ledger_mode=True, b3_ledger_landing=True, live_score_cadence=True,
                live_static_exact=True, live_ama_plain=True, b3_model='ensemble.json', b3_timing_table='timing.json')
    (tmp_path/'ensemble.json').write_text('{}', encoding='utf-8')
    (tmp_path/'timing.json').write_text('{}', encoding='utf-8')
    from src.phase_j.launcher import USER_KEYS
    off = parse_user_config(dict(base), tmp_path)
    assert off.b3_ensemble is False and 'b3_ensemble' not in build_pipeline_config(off)    # pipeline_config.json は従来と同じ
    # 後続の項目が増えても、既存の位置引数の位置は変えない。
    ensemble_position = 82  # UserConfig に b3_ensemble を追加した時の位置 (0 始まり)。
    assert 'b3_ensemble' in USER_KEYS and list(UserConfig.__dataclass_fields__).index('b3_ensemble') == ensemble_position
    on = parse_user_config(dict(base, b3_c2=True, b3_skip_icon_reading=True, b3_ensemble=True), tmp_path)
    config = build_pipeline_config(on)
    assert config['b3_ensemble'] is True and config['b3_c2'] is True and config['b3_model'].endswith('ensemble.json')
    for over, word in ((dict(b3_ensemble=True), 'b3_c2=true'), (dict(b3_c2=True, b3_skip_icon_reading=True, b3_ensemble='true'), 'b3_ensemble'),
                       (dict(b3_c2=True, b3_skip_icon_reading=True, b3_ensemble=1), 'b3_ensemble')):
        with pytest.raises(LauncherConfigError, match=word):
            parse_user_config(dict(base, **over), tmp_path)


def test_cli_flag_is_registered_once_and_defaults_off() -> None:
    parser = argparse.ArgumentParser()
    trial.add_arguments(parser)
    trial.add_arguments(parser)                                    # 二重登録しない
    assert parser.parse_args([]).b3_ensemble is False
    assert parser.parse_args(['--b3-ensemble']).b3_ensemble is True and parser.parse_args(['--no-b3-ensemble']).b3_ensemble is False


def test_baseline_owner_is_told_off_and_the_parent_env_is_restored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v6 のベースラインは 1 本のモデル。親が ON でも、ベースラインの子 process (spawn) へは OFF を明示する。"""
    from src.phase_j import live_b3
    seen: dict[str, str | None] = {}

    class Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen['env'] = os.environ.get(ens.ENSEMBLE_ENV)

    monkeypatch.setattr(live_b3, 'LiveB3', Stub)
    os.environ[ens.ENSEMBLE_ENV] = '1'
    trial.create_baseline(tmp_path/'table.json', tmp_path, fast_ema=False)
    assert seen['env'] == '0' and os.environ[ens.ENSEMBLE_ENV] == '1'


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, value: dict) -> None:
        self.sent.append(value)

    def recv(self) -> None:
        return None            # 入力の終わり (ループへ入らず終了する)

    def close(self) -> None:
        pass


@pytest.mark.parametrize('on', (False, True))
def test_async_compute_process_loads_one_model_off_and_the_ensemble_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                       on: bool) -> None:
    """子 process の入口 (b3_async._compute): OFF は従来の load_model だけ、ON は環境変数で届いた旗でマニフェストを読む。"""
    from src import b3_async, b3_train_model
    from src.b3_timing import TimingTable
    calls: list[str] = []
    monkeypatch.setattr(b3_train_model, 'load_model', lambda path: calls.append('single') or 'single-model')
    monkeypatch.setattr(ens, 'load_ensemble_model', lambda path: calls.append('ensemble') or 'ensemble-model')
    monkeypatch.setattr(b3_async, 'calculate', lambda *a, **k: {})
    os.environ[ens.ENSEMBLE_ENV] = str(int(on))
    connection = FakeConnection()
    b3_async._compute(connection, tmp_path/'model', TimingTable(), 'live', False)
    assert connection.sent == [{'ready': True}] and calls == ['ensemble' if on else 'single']


def build_live_b3(folder: Path, monkeypatch: pytest.MonkeyPatch, model: Any,
                  manifest: Path | None = None) -> tuple[Any, Path]:
    """仮のモデルで LiveB3 を作り (資産の読込と探索の本体は使わない)、列の照合・設定の記録だけを見る。

    manifest を渡したときは読込を差し替えず、本物の load_ensemble_model でそのマニフェストを読む。"""
    from src import puyo_core_bridge as native
    from src.b3_timing import TimingTable
    from src.phase_j import live_b3
    monkeypatch.setattr(native, 'NATIVE_AVAILABLE', True)
    monkeypatch.setattr(native, '_native', SimpleNamespace(b3_search_py=None))
    if manifest is None:
        monkeypatch.setattr(live_b3, 'load_model', lambda path: model)
        monkeypatch.setattr(ens, 'load_ensemble_model', lambda path: model)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'timing.json').write_text(TimingTable().to_json(), encoding='utf-8')
    (folder/'model.joblib').write_bytes(b'fake-model')
    out = folder/'out'
    out.mkdir()
    return live_b3.LiveB3(manifest or folder/'model.joblib', folder/'timing.json', out, trial_helpers=False), out


def fake_columns_model(columns: list[str]) -> SimpleNamespace:
    evaluator = SimpleNamespace(columns=list(columns))
    return SimpleNamespace(static=evaluator, exchange=evaluator)


def c2_columns() -> list[str]:
    from tests.test_b3_c2_trial import c2_columns as columns
    return columns()


def set_c2_env(ensemble: bool) -> None:
    from tests.test_b3_c2_trial import set_trial_env
    set_trial_env(True)
    os.environ[ens.ENSEMBLE_ENV] = str(int(ensemble))


def test_live_b3_config_gets_the_ensemble_record_only_when_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    columns = c2_columns()
    set_c2_env(ensemble=False)
    live, out = build_live_b3(tmp_path/'off', monkeypatch, fake_columns_model(columns))
    live.close()
    off_record = json.loads((out/'b3_config.json').read_text(encoding='utf-8'))
    assert 'ensemble' not in off_record and list(off_record)[-1] == 'ledger_flags' and off_record['trial_model'] == 'C2'
    manifest_files = []
    for index in range(3):
        manifest_files.append(tmp_path/f'seed{1003+index}.joblib')
        manifest_files[-1].write_bytes(f'm{index}'.encode())
    manifest = ens.read_manifest(write_manifest(tmp_path, manifest_files, ens.AGG_MEAN_LOGIT))
    model = ens.EnsembleTwoStateModel([fake_columns_model(columns)] * 3, ens.AGG_MEAN_LOGIT, manifest)
    set_c2_env(ensemble=True)
    live, out = build_live_b3(tmp_path/'on', monkeypatch, model)
    live.close()
    on_record = json.loads((out/'b3_config.json').read_text(encoding='utf-8'))
    paths_only = ('model', 'timing_table')                                 # フォルダの違いだけで値が変わる欄
    assert list(on_record)[-1] == 'ensemble'
    assert {k: v for k, v in on_record.items() if k not in (*paths_only, 'ensemble')} == {
        k: v for k, v in off_record.items() if k not in paths_only}       # ensemble の欄を足す以外、OFF の記録と同じ
    assert on_record['ensemble']['aggregate'] == ens.AGG_MEAN_LOGIT and on_record['ensemble']['n_members'] == 3
    assert (on_record['static_columns'], on_record['exchange_columns']) == (len(columns), len(columns))


def test_live_b3_column_check_still_bites_on_an_ensemble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """アンサンブルでも、列の照合 (評価器の列 ⊂ 実装の列) と C2 の保険 (アイコン列が残るモデルは不可) が効く。"""
    columns = c2_columns()
    set_c2_env(ensemble=True)
    with pytest.raises(ValueError, match='アイコン由来列が残っています'):
        build_live_b3(tmp_path/'icon', monkeypatch, ens.EnsembleTwoStateModel(
            [fake_columns_model(columns + [trial.C2_REMOVED_ICON_COLUMNS[0]])] * 2))
    with pytest.raises(ValueError, match='特徴列が実装と一致しません'):
        build_live_b3(tmp_path/'unknown', monkeypatch, ens.EnsembleTwoStateModel(
            [fake_columns_model(columns + ['no_such_column'])] * 2))


# ------------------------------------------------------------------------------------ 実データ: 本番用 C2 と dl1k の実行

needs_real = pytest.mark.skipif(not C2_MODEL.exists(), reason='本番用 C2 モデルが無い')


@needs_real
def test_real_c2_model_three_copies_are_bit_identical_to_single_on_real_rows() -> None:
    """本番用 C2 (large・seed1003) を 3 回読み込んで並べた出力は、実データの行 400 本で 1 本と 1 ビットも違わない。"""
    from scripts import b3_ensemble_bench as bench
    if not bench.ROWS_PARQUET.exists():
        pytest.skip('dl1k の学習行が無い')
    members = bench.build_members([C2_MODEL], 3)
    rows = bench.real_rows(sorted(set(members[0].static.columns) | set(members[0].exchange.columns)), 400)
    for aggregate in ens.AGGREGATES:
        ensemble = ens.EnsembleTwoStateModel(members, aggregate)
        single = [members[0].predict_row(r) for r in rows]
        assert same_bits_all(single, [ensemble.predict_row(r) for r in rows]), aggregate
    modes = [r['mode_exchange'] for r in rows]
    assert modes.count(0.0) > 50 and modes.count(1.0) > 50 and len({round(p, 5) for p in single}) > 100     # 母数


# ------------------------------------------------------------------------------------ 実データ: PC2 の最終学習 ref × seed 3 本

REF_DIR = Path('C:/puyo1_work/b3_ensemble_20261025/models' if os.name == 'nt' else '/mnt/c/puyo1_work/b3_ensemble_20261025/models')
REF_SHA256 = {                       # PC2 の FINAL_REF_SUMMARY.md (2026-10-25 15:47〜16:08)。PC1 が写して照合
    'ref_seed1003.joblib': '7441675a0209f4a7db567c066b8fd49d1fceee2591be3bdd1b4d83fa671f3cdd',
    'ref_seed1004.joblib': '8029c6d036646eedddab65e86b55030f17c0b20d9a24e2d0f5d83979374264ce',
    'ref_seed1005.joblib': '98e8d2f973ff8dbd92c75854e5cdf3515e9788178f7959eb5f984c38d27a282d'}
REF_COLUMNS = (222, 295)             # static・exchange の列数 (C2 と同じ。PC2 の記録)
needs_ref = pytest.mark.skipif(not all((REF_DIR/name).exists() for name in REF_SHA256), reason='PC2 の ref 3 本が無い')


def pc2_logit_of_singles(singles: list[float]) -> float:
    return float(pc2_logit_mean(np.array([singles], dtype=np.float64))[0])


@needs_ref
def test_real_ref_models_have_the_delivered_sha_and_the_same_c2_columns() -> None:
    for name, digest in REF_SHA256.items():
        assert sha256((REF_DIR/name).read_bytes()).hexdigest() == digest, name
    members = [load_model(REF_DIR/name) for name in REF_SHA256]
    ensemble = ens.EnsembleTwoStateModel(members)                   # 列の名前・順序が 3 本で同じことの確認を兼ねる
    assert (len(ensemble.static.columns), len(ensemble.exchange.columns)) == REF_COLUMNS
    assert set(ensemble.static.columns) | set(ensemble.exchange.columns) <= set(c2_columns())
    trial.check_c2_model(ensemble)                                  # アイコン由来列が無く、正味の列がある (C2 の保険)


@needs_ref
def test_real_ref_ensemble_b_c_d_on_real_rows_through_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """実データの行 400 本で: (b) 同じ ref を 3 回読んだ出力が 1 本とビット一致、(c) 3 本の出力は PC2 の式の平均と一致し
    1 本とは違う、(d) マニフェスト経由の読込が手で並べた入れ物とビット一致。LiveB3 の列の照合と記録も実モデルで通る。"""
    from scripts import b3_ensemble_bench as bench
    if not bench.ROWS_PARQUET.exists():
        pytest.skip('dl1k の学習行が無い')
    for name in REF_SHA256:
        (tmp_path/name).write_bytes((REF_DIR/name).read_bytes())
    members = [load_model(tmp_path/name) for name in REF_SHA256]
    columns = sorted(set(members[0].static.columns) | set(members[0].exchange.columns))
    rows = bench.real_rows(columns, 400)
    singles = [[m.predict_row(r) for r in rows] for m in members]
    repeated = [load_model(tmp_path/'ref_seed1003.joblib') for _ in range(3)]
    for aggregate in ens.AGGREGATES:                                                                      # (b)
        same = ens.EnsembleTwoStateModel(repeated, aggregate)
        assert same_bits_all(singles[0], [same.predict_row(r) for r in rows]), aggregate
    mixed = ens.EnsembleTwoStateModel(members)                                                            # (c) 既定 = logit の平均
    expected = [pc2_logit_of_singles([singles[k][i] for k in range(3)]) for i in range(len(rows))]
    out = [mixed.predict_row(r) for r in rows]
    assert same_bits_all(out, expected)
    spread = [max(s[i] for s in singles) - min(s[i] for s in singles) for i in range(len(rows))]
    assert float(np.mean(spread)) > 0.005 and max(spread) > 0.05 and any(a != b for a, b in zip(out, singles[0]))   # 母数: 3 本は違う
    manifest = tmp_path/'ensemble.json'
    manifest.write_text(json.dumps(ens.build_manifest([tmp_path/n for n in REF_SHA256], ens.AGG_MEAN_LOGIT)), encoding='utf-8')
    loaded = ens.load_ensemble_model(manifest)                                                            # (d)
    assert same_bits_all(out, [loaded.predict_row(r) for r in rows])
    assert [s.sha256 for s in loaded.manifest.members] == list(REF_SHA256.values())
    set_c2_env(ensemble=True)
    live, written = build_live_b3(tmp_path/'live', monkeypatch, None, manifest=manifest)
    live.close()
    record = json.loads((written/'b3_config.json').read_text(encoding='utf-8'))
    assert record['model_sha256'] == sha256(manifest.read_bytes()).hexdigest() and record['trial_model'] == 'C2'
    assert (record['static_columns'], record['exchange_columns']) == REF_COLUMNS
    assert record['ensemble']['n_members'] == 3 and record['ensemble']['aggregate'] == ens.AGG_MEAN_LOGIT
