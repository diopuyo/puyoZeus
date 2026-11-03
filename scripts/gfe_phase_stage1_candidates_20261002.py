"""V0の再現合格後だけ使用する、段1候補の固定学習手順。"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from scripts.gfe_phase_stage1_20261002 import (
    FOLDS, HGB_PARAMS, LR_PARAMS, OUTPUT, REFERENCE, column_names,
    features, memory_guard, run_path, save_json,
)
from src.exchange_event_features import PHASE_BOUNDS


def fold_inputs(rows: pd.DataFrame, elapsed: np.ndarray, seed: int,
                fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """既存trainerと同じ内側OOF・外側予測・経過秒閾値を読み込む。"""
    root = run_path(REFERENCE, "m0") / f"seed_{seed}/fold_{fold}"
    outer = pd.read_csv(root / "predictions.csv")
    test = outer.row_id.to_numpy()
    train = np.flatnonzero(~rows.video_id.isin(outer.video_id.unique()))
    np.testing.assert_array_equal(rows.iloc[test].video_id, outer.video_id)
    np.testing.assert_array_equal(rows.iloc[test].label, outer.label)
    oof = np.full(len(rows), np.nan)
    for inner in FOLDS:
        with np.load(root / f"inner_{inner}_m0.npz") as saved:
            assert not np.isfinite(oof[saved["indices"]]).any()
            oof[saved["indices"]] = saved["probability"]
    assert np.isfinite(oof[train]).all() and np.isnan(oof[test]).all()
    thresholds = np.quantile(elapsed[train], PHASE_BOUNDS)
    return train, test, oof, outer.A.to_numpy(), np.searchsorted(thresholds, elapsed, side="left")


def train_candidate(variant: str, rows: pd.DataFrame, design: np.ndarray,
                    u: np.ndarray, missing: np.ndarray, elapsed: np.ndarray,
                    seed: int, fold: int) -> pd.DataFrame:
    """構成とハイパーパラメータを固定し、1つのfoldを学習する。"""
    memory_guard()
    started = time.monotonic()
    train, test, oof, outer, time_phase = fold_inputs(rows, elapsed, seed, fold)
    phase = np.column_stack((u, time_phase))
    sign, labels = rows.sign.to_numpy(), rows.label.to_numpy()
    x = features(variant, design[train], oof[train], sign[train], phase[train], missing[train])
    assert x.shape[1] == len(column_names(variant))
    model = (HistGradientBoostingClassifier(**HGB_PARAMS) if variant == "V3"
             else LogisticRegression(**LR_PARAMS))
    model.fit(x, np.where(sign[train] > 0, labels[train], 1-labels[train]))
    del x
    x = features(variant, design[test], outer, sign[test], phase[test], missing[test])
    p = model.predict_proba(x)[:, 1]
    result = rows.iloc[test][["row_id", "video_id", "label", "phase"]].copy()
    result[variant] = np.where(sign[test] > 0, p, 1-p)
    result["seed"], result["fold"] = seed, fold
    destination = OUTPUT / f"{variant}_seed_{seed}_fold_{fold}"
    result.to_csv(destination.with_suffix(".csv"), index=False)
    details: dict[str, Any] = dict(columns=column_names(variant), parameters=model.get_params(),
                                  seconds=time.monotonic()-started)
    if hasattr(model, "coef_"):
        details.update(coef=model.coef_.tolist(), intercept=model.intercept_.tolist(),
                       iterations=model.n_iter_.tolist())
    save_json(destination.with_suffix(".json"), details)
    print(f"{variant} seed={seed} fold={fold} seconds={details['seconds']:.2f}", flush=True)
    return result
