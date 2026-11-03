"""user局面の学習・提供一致と明示読込み、既定OFFの不変性を検証する。"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.exchange_event_evaluator import FileExchangeModels, evaluate_exchange_event, file_sha256
from src.exchange_event_features import D_COLUMNS
from src.exchange_event_user_phase import USER_COLUMNS, USER_MODEL_VERSION, static_user_features, user_phase
from scripts.gfe_phase_stage1_20261002 import features
from tests.test_exchange_event_evaluator import inputs, write_models


def candidate(directory: Path, kind: str = "LogisticRegression") -> Path:
    """小規模の実推定器をmanifest付きで保存する。"""
    directory.mkdir()
    x = np.random.default_rng(0).normal(size=(24, len(USER_COLUMNS)))
    model = (LogisticRegression() if kind == "LogisticRegression" else
             HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=2))
    model.fit(x, np.arange(len(x)) % 2)
    path = directory / "G_fe.joblib"
    joblib.dump(model, path)
    entry = dict(file=path.name, columns=USER_COLUMNS, estimator_type=kind, sha256=file_sha256(path))
    (directory / "manifest.json").write_text(json.dumps(dict(
        version=USER_MODEL_VERSION, models=dict(G_fe=entry))), encoding="utf-8")
    return directory


@pytest.mark.parametrize("kind", ["LogisticRegression", "HistGradientBoostingClassifier"])
def test_optional_loading(tmp_path: Path, kind: str) -> None:
    """両推定器を明示指定で読込み、S1/S3の出力バイトを保持する。"""
    write_models(tmp_path)
    directory = candidate(tmp_path / "candidate", kind)
    base = FileExchangeModels.load(tmp_path)
    new = FileExchangeModels.load(tmp_path, gfe_directory=directory)
    assert new.user_phase and not base.user_phase
    assert type(new.estimators["G_fe"]).__name__ == kind
    for event in inputs()[1:]:
        assert np.float64(evaluate_exchange_event(event, base)).tobytes() == np.float64(
            evaluate_exchange_event(event, new)).tobytes()
    assert 0 <= evaluate_exchange_event(inputs()[0], new) <= 1


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("own,opp", [(18, 18), (18, 19), (47, 18), (48, 18), (18, 48),
                                      (18, np.nan), (48, np.nan)])
def test_training_serving_equal(side: int, own: float, opp: float) -> None:
    """欠測・境界・左右とも54列の浮動小数点バイトが一致する。"""
    d = np.linspace(0, 1, len(D_COLUMNS))
    d[D_COLUMNS.index("board_puyo_total")] = own / 72
    d[D_COLUMNS.index("diff_board_puyo_total")] = (own-opp) / 72
    u, missing = user_phase(np.array([own/72]), np.array([(own-opp)/72]))
    expected = features("V3", np.nan_to_num(d)[None, :], np.array([.7]),
                        np.array([1-2*side]), np.column_stack((u, [2])), missing)[0]
    assert static_user_features(d, .7, side).tobytes() == expected.tobytes()


@pytest.mark.parametrize("corruption", ["columns", "version", "kind", "hash", "wrong_type", "path"])
def test_reject_manifest(tmp_path: Path, corruption: str) -> None:
    """未知の列・種類・パス・改変モデルを黙って受け入れない。"""
    directory = candidate(tmp_path / "candidate")
    path = directory / "manifest.json"
    meta = json.loads(path.read_text())
    entry = meta["models"]["G_fe"]
    if corruption == "columns":
        entry["columns"].reverse()
    elif corruption == "version":
        meta["version"] = "unknown"
    elif corruption in ("kind", "wrong_type"):
        entry["estimator_type"] = "unknown" if corruption == "kind" else "HistGradientBoostingClassifier"
    elif corruption == "hash":
        entry["sha256"] = "bad"
    else:
        entry["file"] = "../G_fe.joblib"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        FileExchangeModels((20, 40), {}).with_user_phase(directory)


def test_default_off_byte_equal(tmp_path: Path) -> None:
    """既定と明示Noneは同じ推定器・同じ出力。候補ファイルを参照しない。"""
    write_models(tmp_path)
    base = FileExchangeModels.load(tmp_path)
    off = FileExchangeModels.load(tmp_path, gfe_directory=None)
    for event in inputs():
        left, right = (evaluate_exchange_event(event, m) for m in (base, off))
        assert np.float64(left).tobytes() == np.float64(right).tobytes()


def test_user_phase_ignores_elapsed(tmp_path: Path) -> None:
    """user局面モデルの入力・出力が経過秒に依存しない。"""
    write_models(tmp_path)
    model = FileExchangeModels.load(tmp_path, gfe_directory=candidate(tmp_path / "candidate"))
    event = inputs()[0]
    assert evaluate_exchange_event(event, model) == evaluate_exchange_event(replace(event, elapsed_sec=999), model)
