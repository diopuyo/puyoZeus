"""F9制約の合成データ検証。本番の学習行・モデル・結果は読まない。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from src.b3_column_spec import COLUMN_SPECS, Kind, swap_sides
from src.b3_train_model import (
    F9_MONOTONIC, HgbConfig, fit_two_state, load_model, model_monotonic_constraints,
    monotonic_constraints, predict_hgb, save_model, to_matrix, train_weights,
)

CFG = HgbConfig("synthetic", 35, 0.15, 7, 10)
SEED = 1003
N_ROWS = 800
N_STEPS = 41
TOL = 1e-12
COLS = list(F9_MONOTONIC)


@pytest.fixture
def rows() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(rng.uniform(size=(N_ROWS, len(COLS))), columns=COLS)
    signal = df.to_numpy() @ np.array(list(F9_MONOTONIC.values()))
    df["won_p1"] = (signal + rng.normal(size=N_ROWS) > 1).astype(float)
    df["weight"] = rng.uniform(0.1, 1, N_ROWS)
    df["mode_exchange"] = np.arange(N_ROWS) % 2
    return pd.concat([df, pd.DataFrame([swap_sides(r) for r in df.to_dict("records")])], ignore_index=True)


@pytest.mark.parametrize("column", COLS)
def test_f9_monotonic_in_both_states(rows: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
                                    column: str) -> None:
    monkeypatch.setenv("B3_MONOTONIC", "1")
    model = fit_two_state(rows, COLS, CFG, SEED)
    for ev in (model.static, model.exchange):
        assert ev.monotonic_cst == F9_MONOTONIC
        for record in rows.iloc[[0, 1, N_ROWS, N_ROWS + 1]].to_dict("records"):
            probe = pd.DataFrame([record] * N_STEPS)
            probe[column] = np.linspace(rows[column].min(), rows[column].max(), N_STEPS)
            assert np.all(np.diff(ev.predict(probe)) * F9_MONOTONIC[column] >= -TOL)


@pytest.mark.parametrize("setting", [None, "0"])
def test_off_bitwise_matches_original(rows: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
                                      setting: str | None) -> None:
    monkeypatch.delenv("B3_MONOTONIC", raising=False)
    if setting is not None:
        monkeypatch.setenv("B3_MONOTONIC", setting)
    model = fit_two_state(rows, COLS, CFG, SEED)
    for flag, ev in ((0, model.static), (1, model.exchange)):
        sub = rows[rows.mode_exchange == flag]
        X = to_matrix(sub, COLS)
        with threadpool_limits(limits=1):
            original = HistGradientBoostingClassifier(**CFG.params(SEED)).fit(
                X, sub.won_p1.to_numpy(), sample_weight=train_weights(sub.weight.to_numpy()))
        assert ev.monotonic_cst == {} and ev.model.monotonic_cst is None
        assert ev.predict(rows).tobytes() == predict_hgb(original, to_matrix(rows, COLS)).tobytes()


@pytest.mark.parametrize("column", COLS)
def test_swap_sign_matches_reversed_label(column: str) -> None:
    specs = {s.name: s for s in COLUMN_SPECS}
    base = dict.fromkeys(COLS, 0.25)
    base["won_p1"] = 0.0
    increased = {**base, column: 0.75}
    a, b = swap_sides(base), swap_sides(increased)
    spec = specs[column]
    target = spec.counterpart if spec.kind is Kind.SIDE_ABS else column
    assert F9_MONOTONIC[target] * (b[target] - a[target]) == -F9_MONOTONIC[column] * 0.5
    assert a["won_p1"] == 1 - base["won_p1"]
    assert swap_sides(a) == base
    assert specs["df_cells_color"].kind is Kind.DIFF_RAW
    assert specs["df_ind_board_color_puyo_total"].norm == "0.5+0.5*(p1-p2)"


def test_dropped_columns_extra_and_saved_record(rows: pd.DataFrame, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B3_MONOTONIC", "1")
    dropped = COLS[0]
    rows[dropped] = np.nan
    extra = {"aux": np.arange(len(rows)), "aux_missing": np.full(len(rows), np.nan)}
    model = fit_two_state(rows, COLS[::-1], CFG, SEED, extra)
    expected = {c: s for c, s in F9_MONOTONIC.items() if c != dropped}
    for ev in (model.static, model.exchange):
        assert ev.monotonic_cst == expected
        assert ev.model.monotonic_cst == [expected.get(c, 0) for c in ev.columns]
        assert set(ev.dropped_all_nan) == {dropped, "aux_missing"}
    path = tmp_path / "model.joblib"
    save_model(model, path)
    monkeypatch.setenv("B3_MONOTONIC", "0")
    assert model_monotonic_constraints(load_model(path)) == dict(static=expected, exchange=expected)


def test_invalid_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B3_MONOTONIC", "invalid")
    with pytest.raises(ValueError, match="B3_MONOTONIC"):
        monotonic_constraints(COLS)
    assert monotonic_constraints(COLS, False) == {}


def test_rcv_explicitly_disables_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import b3_train_rcv as module
    from src.b3_eval_cv import TrainContext
    from src.b3_train_synth import make_rows
    monkeypatch.setenv("B3_MONOTONIC", "1")
    df, gmap = make_rows(n_videos=2, games_per_video=2, rows_per_game=24)
    calls = []

    def fit(*args: object, **kwargs: object) -> object:
        calls.append(kwargs["monotonic"])
        model = fit_two_state(*args, **kwargs)
        assert all(not value for value in model_monotonic_constraints(model).values())
        return model

    monkeypatch.setattr(module, "fit_two_state", fit)
    module.rcv_trainer()(df, TrainContext(0, SEED, gmap, {}, COLS))
    assert calls == [False, False]


def test_shared_choice_rejects_other_constraint(rows: pd.DataFrame, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from src import b3_eval_cv as module
    from src.b3_train_model import InnerChoice
    rows["row_key"] = np.arange(len(rows)).astype(str)
    monkeypatch.setattr(module, "select_config", lambda *args: InnerChoice(CFG, [{"ll": 0.5}], False))
    ctx = module.TrainContext(0, SEED, {}, {}, COLS)
    monkeypatch.setenv("B3_MONOTONIC", "0")
    module.candidate_trainer((CFG,), tmp_path)(rows, ctx)
    monkeypatch.setenv("B3_MONOTONIC", "1")
    with pytest.raises(ValueError, match="条件が一致しない"):
        module.candidate_trainer((CFG,), tmp_path)(rows, replace(ctx, seed=SEED + 1))


def test_final_metadata_records_actual_constraints(rows: pd.DataFrame, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import json
    from scripts import b3_train_stage1 as module
    monkeypatch.setenv("B3_MONOTONIC", "1")
    monkeypatch.setattr(module, "DEFAULT_GRID", (CFG,))
    path = tmp_path / "final.joblib"
    args = argparse.Namespace(final_config=CFG.name, score_only=False, folds=None,
                              save_final=path, final_seed=SEED)
    module._save_final(args, rows, COLS, "synthetic-assignment")
    expected = dict(static=F9_MONOTONIC, exchange=F9_MONOTONIC)
    assert json.loads(path.with_suffix(".json").read_text())["monotonic_cst"] == expected
    assert model_monotonic_constraints(load_model(path)) == expected
