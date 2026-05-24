"""Phase H2 TimeseriesWrapper テスト.

15 テストで以下を確認:
    - 履歴管理 (古い entries が drop)
    - Δ 計算正確性
    - 加速度計算
    - max/min/mean の正確性
    - 履歴空時のハンドリング
    - 単発 update での動作 (履歴 1 entry のみ)
    - 270 features 列数
    - reset 動作
"""
from __future__ import annotations

import pytest

from src.board import COLOR_RED, Board
from src.indicators import (
    ALL_INDICATOR_NAMES,
    EXTRA_INDICATOR_NAMES,
    IndicatorCalculator,
    IndicatorResult,
    IndicatorSet,
)
from src.timeseries_indicator_wrapper import (
    ALL_AXES,
    AXIS_ACCEL,
    AXIS_DELTA,
    AXIS_HIST_MAX,
    AXIS_HIST_MEAN,
    AXIS_HIST_MIN,
    AXIS_STATIC,
    DEFAULT_HISTORY_SEC,
    DELTA_EPSILON,
    NEUTRAL_DELTA,
    TIMESERIES_FEATURE_NAMES,
    TIMESERIES_INDICATOR_NAMES,
    TimeseriesEntry,
    TimeseriesWrapper,
    build_timeseries_feature_names,
    extract_static_vector,
    indicator_value,
)

# ============================
# fixtures
# ============================

# テスト用に「全指標を同一スカラー値で埋めた」IndicatorSet を生成するヘルパー。
# IndicatorResult は最低限 name/score/raw_value のみ持てば十分。


def _make_indicator_set(value: float) -> IndicatorSet:
    """全 45 指標を value で埋めた IndicatorSet を返す."""
    results: dict[str, IndicatorResult] = {}
    for name in ALL_INDICATOR_NAMES:
        results[name] = IndicatorResult(
            name=name, score=value, raw_value=value,
        )
    for name in EXTRA_INDICATOR_NAMES:
        results[name] = IndicatorResult(
            name=name, score=value, raw_value=value,
        )
    return IndicatorSet(results=results, next_acceptance=value)


# ============================
# 基本構造テスト
# ============================


def test_indicator_count_at_least_45() -> None:
    """45+ 指標が定義されていることを確認.

    2026-05-22: 過去 cycle で indicator が 45 → 47 に増えている (= EXTRA_INDICATOR_NAMES
    末尾追加方針通り)。 厳密 == から >= 45 に緩和。 cycle 56 以降の追加は受け入れる。
    """
    assert len(TIMESERIES_INDICATOR_NAMES) >= 45


def test_axes_count_6() -> None:
    """6 軸が定義されていることを確認."""
    assert len(ALL_AXES) == 6


def test_feature_names_count_matches_indicators() -> None:
    """indicator × 軸 = feature 数."""
    names = build_timeseries_feature_names()
    expected = len(TIMESERIES_INDICATOR_NAMES) * 6
    assert len(names) == expected
    assert TIMESERIES_FEATURE_NAMES == names


def test_feature_names_format() -> None:
    """列名は indicator__axis 形式."""
    names = TIMESERIES_FEATURE_NAMES
    assert "main_chain_maturity__static" in names
    assert "main_chain_maturity__delta" in names
    assert "main_chain_maturity__hist_mean" in names


# ============================
# indicator_value / extract_static_vector
# ============================


def test_indicator_value_results() -> None:
    """results dict に入っている指標値を取得できる."""
    iset = _make_indicator_set(0.7)
    assert indicator_value(iset, "main_chain_maturity") == 0.7


def test_indicator_value_next_acceptance_attribute() -> None:
    """next_acceptance は属性 fallback で取得."""
    iset = IndicatorSet(results={}, next_acceptance=0.42)
    assert indicator_value(iset, "next_acceptance") == 0.42


def test_extract_static_vector_matches_indicator_count() -> None:
    """extract_static_vector は indicator 数と一致 (= cycle 56 以降の追加対応)."""
    iset = _make_indicator_set(0.3)
    vec = extract_static_vector(iset)
    assert len(vec) == len(TIMESERIES_INDICATOR_NAMES)
    assert all(v == 0.3 for v in vec.values())


# ============================
# 単発 update での動作 (履歴 1 entry)
# ============================


def test_first_update_delta_neutral() -> None:
    """履歴 1 entry のみの場合、delta は NEUTRAL_DELTA."""
    w = TimeseriesWrapper()
    iset = _make_indicator_set(0.5)
    w.update(0.0, iset)
    feats = w.expand_features(iset)
    assert feats["main_chain_maturity__delta"] == NEUTRAL_DELTA
    assert feats["main_chain_maturity__accel"] == NEUTRAL_DELTA


def test_first_update_hist_stats_equal_static() -> None:
    """履歴 1 entry なら hist_max/min/mean は static と一致."""
    w = TimeseriesWrapper()
    iset = _make_indicator_set(0.5)
    w.update(0.0, iset)
    feats = w.expand_features(iset)
    assert feats["main_chain_maturity__static"] == 0.5
    assert feats["main_chain_maturity__hist_max"] == 0.5
    assert feats["main_chain_maturity__hist_min"] == 0.5
    assert feats["main_chain_maturity__hist_mean"] == 0.5


# ============================
# Δ 計算
# ============================


def test_delta_basic() -> None:
    """delta = (curr - prev) / max(|prev|, eps)."""
    w = TimeseriesWrapper()
    w.update(0.0, _make_indicator_set(0.4))
    w.update(0.6, _make_indicator_set(0.6))
    feats = w.expand_features(_make_indicator_set(0.6))
    # (0.6 - 0.4) / max(|0.4|, eps) = 0.2 / 0.4 = 0.5
    assert feats["main_chain_maturity__delta"] == pytest.approx(0.5)


def test_delta_prev_zero_uses_epsilon() -> None:
    """prev=0 の場合 epsilon で割って大きな delta になる."""
    w = TimeseriesWrapper()
    w.update(0.0, _make_indicator_set(0.0))
    w.update(0.6, _make_indicator_set(0.5))
    feats = w.expand_features(_make_indicator_set(0.5))
    expected = 0.5 / DELTA_EPSILON
    assert feats["main_chain_maturity__delta"] == pytest.approx(expected)


# ============================
# 加速度計算
# ============================


def test_accel_neutral_with_two_entries() -> None:
    """履歴 2 entries なら accel は NEUTRAL_DELTA (delta_prev 不能)."""
    w = TimeseriesWrapper()
    w.update(0.0, _make_indicator_set(0.4))
    w.update(0.6, _make_indicator_set(0.6))
    feats = w.expand_features(_make_indicator_set(0.6))
    assert feats["main_chain_maturity__accel"] == NEUTRAL_DELTA


def test_accel_basic_three_entries() -> None:
    """履歴 3 entries で accel = delta_curr - delta_prev."""
    w = TimeseriesWrapper()
    # t0: 0.4
    w.update(0.0, _make_indicator_set(0.4))
    # t1: 0.5  -> delta_prev = (0.5 - 0.4)/0.4 = 0.25
    w.update(0.6, _make_indicator_set(0.5))
    # t2: 0.6  -> delta_curr = (0.6 - 0.5)/0.5 = 0.2
    iset_curr = _make_indicator_set(0.6)
    w.update(1.2, iset_curr)
    feats = w.expand_features(iset_curr)
    # accel = 0.2 - 0.25 = -0.05
    assert feats["main_chain_maturity__accel"] == pytest.approx(-0.05)


# ============================
# 履歴 max/min/mean
# ============================


def test_hist_max_min_mean_three_entries() -> None:
    """3 entries の max/min/mean が正しく計算される."""
    w = TimeseriesWrapper()
    w.update(0.0, _make_indicator_set(0.2))
    w.update(0.6, _make_indicator_set(0.8))
    iset_curr = _make_indicator_set(0.5)
    w.update(1.2, iset_curr)
    feats = w.expand_features(iset_curr)
    assert feats["main_chain_maturity__hist_max"] == pytest.approx(0.8)
    assert feats["main_chain_maturity__hist_min"] == pytest.approx(0.2)
    # mean = (0.2 + 0.8 + 0.5) / 3 = 0.5
    assert feats["main_chain_maturity__hist_mean"] == pytest.approx(0.5)


# ============================
# 履歴管理 (古い entries の drop)
# ============================


def test_history_drops_old_entries() -> None:
    """history_sec を超えた entries は drop される."""
    w = TimeseriesWrapper(history_sec=10.0)
    w.update(0.0, _make_indicator_set(0.1))
    w.update(5.0, _make_indicator_set(0.5))
    w.update(11.0, _make_indicator_set(0.9))
    # t=11 から見て t=0 は 11 秒前 = cutoff (1.0) より古いので drop
    # t=5 は 6 秒前 = 残る
    timestamps = [e.timestamp for e in w.history]
    assert 0.0 not in timestamps
    assert 5.0 in timestamps
    assert 11.0 in timestamps


def test_reset_clears_history() -> None:
    """reset() で履歴が空になる."""
    w = TimeseriesWrapper()
    w.update(0.0, _make_indicator_set(0.5))
    w.update(0.6, _make_indicator_set(0.6))
    assert len(w.history) == 2
    w.reset()
    assert len(w.history) == 0


# ============================
# 統合テスト (実 IndicatorSet で動作確認)
# ============================


def test_integration_with_real_indicator_set() -> None:
    """実 IndicatorCalculator で計算した IndicatorSet で動作する."""
    b = Board()
    b.set(12, 0, COLOR_RED)
    b.set(12, 1, COLOR_RED)
    b.set(12, 2, COLOR_RED)
    b.set(12, 3, COLOR_RED)
    calc = IndicatorCalculator()
    iset = calc.compute_all(b)
    w = TimeseriesWrapper()
    w.update(0.0, iset)
    feats = w.expand_features(iset)
    # 270 features 揃う
    assert len(feats) == len(TIMESERIES_INDICATOR_NAMES) * 6
    # 値はすべて float
    for v in feats.values():
        assert isinstance(v, float)


def test_expand_features_all_270_keys() -> None:
    """expand_features の戻り値は 270 key."""
    w = TimeseriesWrapper()
    iset = _make_indicator_set(0.5)
    w.update(0.0, iset)
    feats = w.expand_features(iset)
    assert len(feats) == len(TIMESERIES_INDICATOR_NAMES) * 6
    for name in TIMESERIES_FEATURE_NAMES:
        assert name in feats


def test_history_sec_default() -> None:
    """デフォルト履歴秒数が 30."""
    w = TimeseriesWrapper()
    assert w.history_sec == DEFAULT_HISTORY_SEC == 30.0
