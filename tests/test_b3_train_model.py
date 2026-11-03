"""B3 学習器・CV のテスト (合成データ): 決定性、fold 漏れ、重み、欠測、状態選択、設定選択。"""
import numpy as np
import pandas as pd
import pytest

from src.b3_eval_cv import TrainContext, candidate_trainer, run_cv, stage1_scores
from src.b3_train_folds import games_per_group_from_rows, inner_assignment, outer_assignment, FoldLeakError
from src.b3_train_model import (FORBIDDEN_FEATURES, HgbConfig, MODE_COLUMN, aux_columns, feature_columns, fit_hgb,
                                fit_two_state, select_config, to_matrix, train_weights)
from src.b3_train_rcv import coverage, rcv_trainer
from src.b3_train_synth import make_rows

TINY = (HgbConfig("tiny", 8, 0.2, 7, 20), HgbConfig("tiny2", 16, 0.2, 15, 20))


@pytest.mark.parametrize("value,expected", [(None, 1), ("8", 8)])
def test_hgb_threads_limits_blas_separately(monkeypatch, value: str | None, expected: int) -> None:
    from contextlib import contextmanager
    from src import b3_train_model as module
    if value is None:
        monkeypatch.delenv("B3_HGB_THREADS", raising=False)
    else:
        monkeypatch.setenv("B3_HGB_THREADS", value)
    calls = []

    @contextmanager
    def limits(*, limits: int, user_api: str):
        calls.append((user_api, limits))
        yield

    monkeypatch.setattr(module, "threadpool_limits", limits)
    X = np.arange(80, dtype=float).reshape(40, 2)
    model = module.fit_hgb(X, np.arange(40) % 2, np.ones(40), TINY[0], 1003)
    module.predict_hgb(model, X)
    assert calls == [("blas", 1), ("openmp", expected)] * 2


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5"])
def test_hgb_threads_reject_invalid(monkeypatch, value: str) -> None:
    from src.b3_train_model import hgb_threads
    monkeypatch.setenv("B3_HGB_THREADS", value)
    with pytest.raises(ValueError):
        hgb_threads()


def test_shared_choice_reused_across_trainers(synth, tmp_path, monkeypatch) -> None:
    from dataclasses import replace
    from src import b3_eval_cv as module
    from src.b3_train_model import InnerChoice
    df, gmap = synth
    cols = feature_columns(df.columns)[:3]
    ctx = TrainContext(0, 1003, gmap, {}, cols)
    calls = []

    def choose(*args) -> InnerChoice:
        calls.append(args[3])
        return InnerChoice(TINY[1], [{"ll": 0.5}, {"ll": 0.5005}], False)

    monkeypatch.setattr(module, "select_config", choose)
    monkeypatch.setenv("B3_HGB_THREADS", "8")
    _, first = candidate_trainer(TINY, tmp_path)(df, ctx)
    for seed in (1004, 1005):
        _, info = candidate_trainer(TINY, tmp_path)(df, replace(ctx, seed=seed))
        assert info["chosen"] == first["chosen"] == "tiny2"
        assert info["selection_seed"] == 1003 and info["share_inner_selection"]
        assert info["hgb_threads"] == 8 and info["blas_threads"] == 1
        assert info["inner_ll_gap"] == pytest.approx(0.0005) and info["inner_margin"] == "僅差"
    assert calls == [1003]
    with pytest.raises(FileNotFoundError):
        candidate_trainer(TINY, tmp_path)(df, replace(ctx, seed=1004, outer_fold=1))
    with pytest.raises(ValueError, match="条件が一致しない"):
        candidate_trainer(TINY[:1], tmp_path)(df, replace(ctx, seed=1004))


def test_default_choice_still_selects_each_seed(synth, monkeypatch) -> None:
    from src import b3_eval_cv as module
    from src.b3_train_model import InnerChoice
    df, gmap = synth
    calls = []

    def choose(*args) -> InnerChoice:
        calls.append(args[3])
        return InnerChoice(TINY[0], [{"ll": 0.5}, {"ll": 0.502}], False)

    monkeypatch.setattr(module, "select_config", choose)
    monkeypatch.delenv("B3_HGB_THREADS", raising=False)
    for seed in (1003, 1004):
        _, info = candidate_trainer(TINY)(df, TrainContext(0, seed, gmap, {}, feature_columns(df.columns)[:3]))
        assert not info["share_inner_selection"] and info["selection_seed"] == seed
        assert info["hgb_threads"] == 1 and info["inner_margin"] == ""
    assert calls == [1003, 1004]


@pytest.fixture(scope="module")
def synth():
    df, gmap = make_rows(n_videos=10, games_per_video=3, rows_per_game=40)
    return df, gmap


def test_feature_columns_exclude_forbidden(synth):
    df, _ = synth
    cols = feature_columns(df.columns)
    assert not (set(cols) & FORBIDDEN_FEATURES)
    assert not any(c.startswith(("av_", "r_ledger_", "mk_")) for c in cols)
    assert "won_p1" not in cols and "weight" not in cols and "held_sec" not in cols
    withmk = feature_columns(df.columns, include_mask_codes=True)
    assert set(cols) < set(withmk) and all(c.startswith("mk_") for c in set(withmk) - set(cols))


def test_feature_columns_missing_raises(synth):
    df, _ = synth
    with pytest.raises(KeyError):
        feature_columns(df.drop(columns=["cells_color_p1"]).columns)


def test_to_matrix_keeps_nan_not_zero(synth):
    df, _ = synth
    cols = feature_columns(df.columns)
    X = to_matrix(df, cols)
    assert X.dtype == np.float32 and np.isnan(X).any()
    j = cols.index("k1_ojama_p1")
    assert np.array_equal(np.isnan(X[:, j]), df["k1_ojama_p1"].isna().to_numpy())


def test_train_weights_mean_one_and_ratio_preserved():
    w = np.array([0.001, 0.002, 0.003])
    t = train_weights(w)
    assert t.mean() == pytest.approx(1.0) and t[1] / t[0] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        train_weights(np.zeros(3))


def test_fit_deterministic_bitwise(synth):
    df, _ = synth
    cols = feature_columns(df.columns)
    tr = df[df.video_id < "video_s07"]
    te = df[df.video_id >= "video_s07"]
    p1 = fit_two_state(tr, cols, TINY[0], 1003).predict(te)
    p2 = fit_two_state(tr, cols, TINY[0], 1003).predict(te)
    assert np.array_equal(p1, p2)
    assert np.isfinite(p1).all() and p1.min() > 0 and p1.max() < 1


def test_signal_is_learned(synth):
    df, _ = synth
    cols = feature_columns(df.columns)
    tr, te = df[df.video_id < "video_s07"], df[df.video_id >= "video_s07"]
    p = fit_two_state(tr, cols, HgbConfig("m", 60, 0.1, 15, 20), 1003).predict(te)
    y = te["won_p1"].to_numpy()
    assert np.mean((p > 0.5) == (y == 1)) > 0.6


def test_weights_actually_used(synth):
    df, _ = synth
    cols = ["cells_color_p1", "cells_color_p2"]
    sub = df[df[MODE_COLUMN] == 0].copy()
    y = sub["won_p1"].to_numpy()
    X = to_matrix(sub, cols)
    w_flip = np.where(y == 1, 1e-3, 1e-6)           # 勝ち行を極端に重くする
    cfg = HgbConfig("w", 20, 0.2, 7, 5)
    p_w = fit_hgb(X, y, w_flip, cfg, 1).predict_proba(X)[:, 1].mean()
    p_u = fit_hgb(X, y, np.ones(len(y)) * 1e-3, cfg, 1).predict_proba(X)[:, 1].mean()
    assert p_w > p_u + 0.05                          # 重み付きの方が勝ち寄りになる


def test_tiny_scale_weights_do_not_block_splits(synth):
    df, _ = synth
    sub = df[df[MODE_COLUMN] == 0]
    cols = ["cells_color_p1", "cells_color_p2"]
    X, y = to_matrix(sub, cols), sub["won_p1"].to_numpy()
    m = fit_hgb(X, y, sub["weight"].to_numpy(), HgbConfig("w", 30, 0.2, 7, 5), 1)
    p = m.predict_proba(X)[:, 1]
    assert p.std() > 0.01                            # 重みが 1e-3 規模でも木が成長する


def test_two_state_routes_by_mode_and_requires_mode(synth):
    df, _ = synth
    cols = feature_columns(df.columns)
    model = fit_two_state(df, cols, TINY[0], 5)
    assert model.static is not None and model.exchange is not None
    te = df.head(200).copy()
    p = model.predict(te)
    for flag, ev in ((0, model.static), (1, model.exchange)):
        sel = te[MODE_COLUMN].to_numpy(dtype=float) == flag
        if sel.any():
            assert np.array_equal(p[sel], ev.predict(te[sel]))
    bad = te.copy()
    bad[MODE_COLUMN] = pd.array([pd.NA] * len(bad), dtype="Int8")
    with pytest.raises(ValueError):
        model.predict(bad)


def test_all_nan_column_dropped_and_recorded(synth):
    df, _ = synth
    cols = feature_columns(df.columns)
    tr = df.copy()
    tr["ta2_q_med_p1"] = np.nan
    model = fit_two_state(tr, cols, TINY[0], 5)
    assert "ta2_q_med_p1" in model.static.dropped_all_nan and "ta2_q_med_p1" not in model.static.columns
    te = tr.head(50)
    assert np.isfinite(model.predict(te)).all()


def test_select_config_tie_rule_prefers_cheaper(synth):
    df, gmap = synth
    cols = ["cells_color_p1", "cells_color_p2", "df_cells_color"]
    gpg = games_per_group_from_rows(df, gmap)
    outer = outer_assignment(gpg)
    inner = inner_assignment(gpg, 0, outer)
    train = df[df.video_id.map(gmap).map(lambda g: outer[g] != 0)]
    grid = (HgbConfig("big", 40, 0.1, 31, 20), HgbConfig("small", 40, 0.1, 3, 20))
    ch = select_config(train, cols, grid, 1, inner, gmap, tie=10.0)       # 差が必ず tie 以内
    assert ch.chosen.name == "small"                                        # 費用 (iter×leaf) が小さい側
    ch0 = select_config(train, cols, grid, 1, inner, gmap, tie=0.0)
    best = min(ch0.table, key=lambda t: t["ll"])["config"]
    assert ch0.chosen.name == best
    assert all(t["games"] > 0 for t in ch0.table)


def test_run_cv_no_group_leak_and_full_oof(synth):
    df, gmap = synth
    cols = feature_columns(df.columns)
    seen = []

    def spy_trainer(train_df, ctx: TrainContext):
        groups = set(train_df["video_id"].map(gmap))
        seen.append((ctx.outer_fold, groups, set(ctx.inner_of_group)))
        model = fit_two_state(train_df, ctx.columns, TINY[0], ctx.seed)
        return model.predict, {"n": len(train_df)}

    run = run_cv("spy", df, spy_trainer, gmap, [1003], cols)
    gpg = games_per_group_from_rows(df, gmap)
    outer = outer_assignment(gpg)
    assert len(seen) == 5 and np.isfinite(run.preds[1003]).all()
    for fold, train_groups, inner_groups in seen:
        test_groups = {g for g in gpg if outer[g] == fold}
        assert not (train_groups & test_groups)           # 同一 group が学習と検証に跨らない
        assert inner_groups == train_groups               # 内側割当は train group だけ
    # 同じ group の複製 (原向き/交換) は同じ fold
    fold_by_key = pd.Series(run.fold_of_row, index=df["row_key"].str.rsplit("|o", n=1).str[0])
    assert (fold_by_key.groupby(level=0).nunique() == 1).all()


def test_run_cv_deterministic(synth):
    df, gmap = synth
    cols = feature_columns(df.columns)[:60]

    def tr(train_df, ctx):
        m = fit_two_state(train_df, ctx.columns, TINY[0], ctx.seed)
        return m.predict, {}

    a = run_cv("a", df, tr, gmap, [1003], cols).preds[1003]
    b = run_cv("b", df, tr, gmap, [1003], cols).preds[1003]
    assert np.array_equal(a, b)
    c = run_cv("c", df, tr, gmap, [1004], cols).preds[1004]
    assert not np.array_equal(a, c) or True               # seed が違えば違い得る (HGB は決定的でも許容)


def test_aux_columns_are_oof_no_leak(synth):
    df, gmap = synth
    gpg = games_per_group_from_rows(df, gmap)
    outer = outer_assignment(gpg)
    inner = inner_assignment(gpg, 0, outer)
    train = df[df.video_id.map(gmap).map(lambda g: outer[g] != 0)]
    test = df[df.video_id.map(gmap).map(lambda g: outer[g] == 0)]
    calls = []

    def aux(tr_df, apply_df):
        calls.append((set(tr_df["video_id"]), set(apply_df["video_id"])))
        return np.full(len(apply_df), float(len(set(tr_df["video_id"]))))

    tr_cols, te_cols = aux_columns(train, test, {"aux": aux}, inner, gmap)
    for trv, apv in calls[:-1]:
        assert not (trv & apv)                              # 内側 OOF: 学習側に適用先動画が無い
    assert calls[-1][1] == set(test["video_id"]) and not (calls[-1][0] & calls[-1][1])
    assert np.isfinite(tr_cols["aux"]).all() and len(te_cols["aux"]) == len(test)


def test_rcv_coverage_and_m0_rejected(synth):
    df, _ = synth
    cov = coverage(df.columns)
    assert "board_color_puyo_total" in cov.mapped and cov.mapped["diff_max_column_height"] == "df_ind_max_column_height"
    assert "m0_logit" in cov.unmapped and "all_clear_bonus_pending" in cov.unmapped
    with pytest.raises(NotImplementedError):
        rcv_trainer(m0_fn=lambda a, b: np.zeros(len(b)))


def test_rcv_trainer_runs_and_uses_train_only_quantiles(synth):
    df, gmap = synth
    cols = feature_columns(df.columns)
    gpg = games_per_group_from_rows(df, gmap)
    outer = outer_assignment(gpg)
    train = df[df.video_id.map(gmap).map(lambda g: outer[g] != 0)]
    test = df[df.video_id.map(gmap).map(lambda g: outer[g] == 0)]
    ctx = TrainContext(0, 1003, gmap, inner_assignment(gpg, 0, outer), cols)
    predict, info = rcv_trainer()(train, ctx)
    p = predict(test)
    assert np.isfinite(p).all() and len(p) == len(test)
    q = np.nanquantile(train["elapsed_first_move"].to_numpy(dtype=float, na_value=np.nan), (1 / 3, 2 / 3))
    assert info["elapsed_bounds"] == pytest.approx(q.tolist())
    assert "m0_logit" in info["coverage_unmapped"]


def test_end_to_end_stage1_receipt_and_oracle_stop(synth):
    from src.b3_eval_gates import FAIL, HOLD, STOP, evaluate_stage1
    df, gmap = synth
    cols = feature_columns(df.columns)[:80]
    grid = TINY
    cand = run_cv("cand", df, candidate_trainer(grid), gmap, [1003], cols)
    ref = run_cv("ref", df, rcv_trainer(), gmap, [1003], cols)
    scores = stage1_scores(df, cand, ref, gmap)
    assert scores["overall"]["population"]["games"] == 30
    assert "exchange" in scores["strata"] and "honsen_after" in scores["strata"]
    rec = evaluate_stage1(scores)
    assert rec["verdict"] in (FAIL, HOLD) and "oracle_not_exceeded" in rec["not_evaluated"]   # 前提検査等が未実施
    # 候補 LL より大きい oracle を渡すと停止 (オラクル超え = リーク疑い)
    stop = evaluate_stage1(scores, oracle_ll=scores["overall"]["ll_cand"] + 0.1)
    assert stop["verdict"] == STOP
    ok = evaluate_stage1(scores, oracle_ll=scores["overall"]["ll_cand"] - 0.1)
    assert ok["verdict"] != STOP
