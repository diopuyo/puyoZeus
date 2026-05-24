"""epochs_calculator の単体テスト (= cycle 50、 2026-05-21)."""
from __future__ import annotations

from src.epochs_calculator import (
    MAX_EPOCHS,
    MIN_EPOCHS,
    TARGET_EPOCH_SIZE,
    epochs_for_seed_count,
    report_epochs_for_seed,
)


def test_zero_or_negative() -> None:
    assert epochs_for_seed_count(0) == MIN_EPOCHS
    assert epochs_for_seed_count(-100) == MIN_EPOCHS


def test_baseline_anchor() -> None:
    """8 動画 60K = baseline → MIN_EPOCHS."""
    assert epochs_for_seed_count(TARGET_EPOCH_SIZE) == MIN_EPOCHS


def test_phase_l_scale() -> None:
    """38 動画 280K = baseline 4.7 倍 → ~23 epochs (= Phase L 5 不足)."""
    e = epochs_for_seed_count(280000)
    assert e > MIN_EPOCHS * 4
    assert e <= MAX_EPOCHS


def test_clamp_max() -> None:
    """巨大データ (= 1M) でも MAX で clamp."""
    assert epochs_for_seed_count(1_000_000) == MAX_EPOCHS


def test_clamp_min() -> None:
    """微小データ (= 5K) でも MIN_EPOCHS."""
    assert epochs_for_seed_count(5000) == MIN_EPOCHS


def test_report_structure() -> None:
    r = report_epochs_for_seed(120000)
    assert r["seed_count"] == 120000
    assert r["recommended_epochs"] == epochs_for_seed_count(120000)
    assert r["min_epochs"] == MIN_EPOCHS
    assert r["max_epochs"] == MAX_EPOCHS
