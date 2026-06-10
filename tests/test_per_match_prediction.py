"""scripts.per_match_prediction のスモークテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.old.eda_features import Dataset
from scripts.old.per_match_prediction import (
    EVAL_PHASE_END,
    EVAL_PHASE_MID,
    Strategy,
    hit_rate_by_video,
    load_strategies,
    per_phase_predict,
    plot_correctness_matrix,
    plot_hit_rate_bars,
    predict_strategy,
)


def _make_dataset() -> Dataset:
    """midpoint と end_minus_5 を含む簡単なデータ。"""
    from scripts.old.generate_training_dataset import FEATURE_NAMES
    rng = np.random.default_rng(0)
    n = 60
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    main_idx = FEATURE_NAMES.index("main_chain_maturity")
    y = np.where(X[:, main_idx] > 0, 1, -1)
    phases = [EVAL_PHASE_MID] * 30 + [EVAL_PHASE_END] * 30
    videos = (["01"] * 10 + ["02"] * 10 + ["03"] * 10) * 2
    return Dataset(
        feature_names=FEATURE_NAMES, X=X, y=y,
        video_ids=videos, time_phases=phases,
    )


def test_load_strategies_includes_default_and_global() -> None:
    """DEFAULT と LEARNED_GLOBAL が常に含まれる。V3 は存在時のみ。"""
    strategies = load_strategies()
    names = {s.name for s in strategies}
    assert "DEFAULT" in names
    assert "LEARNED_GLOBAL" in names


def test_predict_strategy_returns_signed_vector() -> None:
    """予測ベクトルは ±1。"""
    ds = _make_dataset()
    strategies = load_strategies()
    pred = predict_strategy(ds, strategies[0])
    assert pred.shape == (len(ds.y),)
    assert set(np.unique(pred).tolist()).issubset({1, -1})


def test_per_phase_predict_filters_phase() -> None:
    """指定 phase のサンプルだけが予測対象になる。"""
    ds = _make_dataset()
    strat = load_strategies()[0]
    pred, true, correct, keys = per_phase_predict(ds, strat, EVAL_PHASE_MID)
    n_mid = sum(1 for p in ds.time_phases if p == EVAL_PHASE_MID)
    assert pred.shape == (n_mid,)
    assert true.shape == (n_mid,)
    assert correct.shape == (n_mid,)
    assert len(keys) == n_mid


def test_hit_rate_by_video_returns_per_video_and_overall() -> None:
    """動画別 + overall キーが返る。"""
    ds = _make_dataset()
    strat = load_strategies()[0]
    rates = hit_rate_by_video(ds, strat, EVAL_PHASE_MID)
    assert "overall" in rates
    for v in ("01", "02", "03"):
        assert v in rates
        assert 0.0 <= rates[v] <= 1.0


def test_plot_outputs_create_pngs(tmp_path: Path) -> None:
    """棒グラフとマトリクス PNG が出力されること。"""
    ds = _make_dataset()
    strategies = load_strategies()
    results: dict[str, dict[str, dict[str, float]]] = {}
    correctness: dict[str, dict[str, np.ndarray]] = {}
    keys: dict[str, list[tuple[str, int]]] = {}
    for phase in (EVAL_PHASE_MID, EVAL_PHASE_END):
        results[phase] = {}
        correctness[phase] = {}
        for s in strategies:
            results[phase][s.name] = hit_rate_by_video(ds, s, phase)
            _, _, c, k = per_phase_predict(ds, s, phase)
            correctness[phase][s.name] = c
            keys[phase] = k

    bar_path = tmp_path / "bar.png"
    matrix_path = tmp_path / "matrix.png"
    plot_hit_rate_bars(results, bar_path)
    plot_correctness_matrix(correctness, keys, matrix_path)
    assert bar_path.exists() and bar_path.stat().st_size > 0
    assert matrix_path.exists() and matrix_path.stat().st_size > 0
