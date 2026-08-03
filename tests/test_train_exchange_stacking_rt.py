"""scripts/train_exchange_stacking_rt.py (併用スタッキング版 推論バンドル学習) のテスト。

合成 aug CSV (sim_* 3列付き) で軽量に検証する (実66動画データは使わない)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_exchange_triple_comparison import SIM_FEATURE_COLS
from scripts.train_exchange_stacking_rt import train_and_save_stacking_bundle
from src.exchange_predictor import load_exchange_model, predict_exchange_event

INDICATOR_BASES = ["current_max_chain", "board_ojama_count", "death_margin"]


def _make_synthetic_aug_csv(tmp_path, n: int = 150, seed: int = 3) -> "object":
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "video_id": rng.choice([f"video_c{i}" for i in range(10)], size=n),
        "game_idx": rng.integers(0, 5, size=n),
        "t_sec": rng.uniform(0, 3000, size=n),
        "phase": rng.choice(["序", "中", "終"], size=n, p=[0.2, 0.3, 0.5]),
        "fire_side": rng.choice(["1P", "2P"], size=n),
    }
    for prefix in ("fire_", "opp_", "diff_"):
        for base in INDICATOR_BASES:
            data[f"{prefix}{base}"] = rng.normal(size=n)
    data["taiou_success"] = rng.integers(0, 2, size=n)
    data["net_ojama_after"] = rng.normal(loc=50.0, scale=30.0, size=n)
    data["sim_k_hands"] = rng.integers(1, 5, size=n).astype(float)
    data["sim_expected_counter_ojama"] = rng.normal(loc=100.0, scale=50.0, size=n)
    data["sim_damage_score"] = rng.uniform(0.0, 1.0, size=n)
    df = pd.DataFrame(data)
    path = tmp_path / "synthetic_aug.csv"
    df.to_csv(path, index=False)
    return path


def test_train_and_save_stacking_bundle_roundtrip(tmp_path) -> None:
    """学習→保存→ロードで sim_feature_cols 付き44特徴量バンドルが得られること。"""
    aug_csv = _make_synthetic_aug_csv(tmp_path)
    save_path = tmp_path / "stacking.joblib"
    train_and_save_stacking_bundle(aug_csv, save_path, model_date="2026-08-03")

    model = load_exchange_model(save_path)
    assert model.sim_feature_cols == tuple(SIM_FEATURE_COLS)
    assert set(model.indicator_bases) == set(INDICATOR_BASES)
    assert len(model.feature_names) == len(INDICATOR_BASES) * 3 + 3 + 2 + len(SIM_FEATURE_COLS)


def test_stacking_bundle_predicts_without_error(tmp_path) -> None:
    """保存済みバンドルで predict_exchange_event が例外なく実行できること。"""
    aug_csv = _make_synthetic_aug_csv(tmp_path)
    save_path = tmp_path / "stacking.joblib"
    train_and_save_stacking_bundle(aug_csv, save_path)
    model = load_exchange_model(save_path)

    features: dict = {"phase": "中", "fire_side": "1P"}
    for prefix in ("fire_", "opp_", "diff_"):
        for base in INDICATOR_BASES:
            features[f"{prefix}{base}"] = 0.5
    for col in SIM_FEATURE_COLS:
        features[col] = 10.0
    prob, pred = predict_exchange_event(model, features)
    assert 0.0 <= prob <= 1.0
    assert isinstance(pred, float)


def test_stacking_bundle_missing_sim_key_raises(tmp_path) -> None:
    """sim_* キーが欠けていれば0埋めせず例外にする (誤判定防止)。"""
    aug_csv = _make_synthetic_aug_csv(tmp_path)
    save_path = tmp_path / "stacking.joblib"
    train_and_save_stacking_bundle(aug_csv, save_path)
    model = load_exchange_model(save_path)

    features: dict = {"phase": "中", "fire_side": "1P"}
    for prefix in ("fire_", "opp_", "diff_"):
        for base in INDICATOR_BASES:
            features[f"{prefix}{base}"] = 0.5
    # sim_* を意図的に欠落させる
    with pytest.raises(KeyError):
        predict_exchange_event(model, features)
