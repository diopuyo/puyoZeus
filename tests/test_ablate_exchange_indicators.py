"""scripts/ablate_exchange_indicators.py の単体テスト (Step4 アブレーション)。

対象: 直近発火イベントの実時刻突合 (game_idx不使用・未来イベント参照禁止)、
neutral/decay 既定値の合成、tau感度の単調性、既存 pair_sides_for_win との
統合 (自動 _1p/_2p suffix 付与)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.ablate_exchange_indicators import (
    DELTA_LAST_BASE_FMT,
    NEUTRAL_DELTA,
    NEUTRAL_PROB,
    PROB_LAST_BASE_FMT,
    _build_verdict,
    _decay_weight,
    assign_phase_by_tsumo_tertile,
    attach_last_event_features,
    build_auc_diff_table,
    lookup_last_events,
)
from scripts.model_indicator_win import DEFAULT_MAX_TDIFF, build_features, pair_sides_for_win


def _events(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    """(video_id, side, t_sec, prob, delta) タプルのリストから events DataFrame を作る。"""
    return pd.DataFrame(rows, columns=["video_id", "side", "t_sec", "prob", "delta"])


# =============================================================================
# lookup_last_events: 実時刻突合 (backward-only、未来イベント参照禁止)
# =============================================================================


def test_直近過去イベントのみ拾い未来イベントは無視する() -> None:
    events = _events([
        ("video_a", "1P", 10.0, 0.3, 5.0),
        ("video_a", "1P", 50.0, 0.8, 20.0),  # これは未来なので t=30 の行から見えてはならない
    ])
    gap, prob, delta = lookup_last_events(
        np.array(["video_a"]), np.array(["1P"]), np.array([30.0]), events,
    )
    assert gap[0] == pytest.approx(20.0)
    assert prob[0] == pytest.approx(0.3)
    assert delta[0] == pytest.approx(5.0)


def test_イベントより前の時刻はNaNになる() -> None:
    events = _events([("video_a", "1P", 100.0, 0.9, 30.0)])
    gap, prob, delta = lookup_last_events(
        np.array(["video_a"]), np.array(["1P"]), np.array([5.0]), events,
    )
    assert np.isnan(gap[0]) and np.isnan(prob[0]) and np.isnan(delta[0])


def test_別動画_別sideのイベントは混同されない() -> None:
    events = _events([
        ("video_a", "1P", 10.0, 0.1, 1.0),
        ("video_a", "2P", 10.0, 0.9, 9.0),   # 同じ動画でも side違いは見えてはならない
        ("video_b", "1P", 10.0, 0.5, 5.0),   # 別動画は見えてはならない
    ])
    gap, prob, delta = lookup_last_events(
        np.array(["video_a"]), np.array(["1P"]), np.array([20.0]), events,
    )
    assert prob[0] == pytest.approx(0.1)
    assert delta[0] == pytest.approx(1.0)


# =============================================================================
# neutral / decay 既定値の合成
# =============================================================================


def test_neutral方式はイベント無しなら中立値になる() -> None:
    df = pd.DataFrame({
        "video_id": ["video_c"], "side": ["1P"], "t_sec": [5.0],
    })
    events = _events([])
    out = attach_last_event_features(df, events, tau_values=(10.0,))
    assert out[PROB_LAST_BASE_FMT.format(mode="neutral")].iloc[0] == pytest.approx(NEUTRAL_PROB)
    assert out[DELTA_LAST_BASE_FMT.format(mode="neutral")].iloc[0] == pytest.approx(NEUTRAL_DELTA)


def test_neutral方式はイベントありなら時刻減衰せず生値を保持する() -> None:
    df = pd.DataFrame({
        "video_id": ["video_c"], "side": ["1P"], "t_sec": [1000.0],  # 大きく時間が経っていても
    })
    events = _events([("video_c", "1P", 5.0, 0.77, 12.0)])
    out = attach_last_event_features(df, events, tau_values=(10.0,))
    assert out[PROB_LAST_BASE_FMT.format(mode="neutral")].iloc[0] == pytest.approx(0.77)
    assert out[DELTA_LAST_BASE_FMT.format(mode="neutral")].iloc[0] == pytest.approx(12.0)


def test_decay方式はgapゼロなら生値と一致する() -> None:
    df = pd.DataFrame({"video_id": ["video_c"], "side": ["1P"], "t_sec": [5.0]})
    events = _events([("video_c", "1P", 5.0, 0.77, 12.0)])
    out = attach_last_event_features(df, events, tau_values=(10.0,))
    assert out[PROB_LAST_BASE_FMT.format(mode="decay_tau10")].iloc[0] == pytest.approx(0.77, abs=1e-6)
    assert out[DELTA_LAST_BASE_FMT.format(mode="decay_tau10")].iloc[0] == pytest.approx(12.0, abs=1e-6)


def test_decay方式はgapが十分大きいと中立値に収束する() -> None:
    df = pd.DataFrame({"video_id": ["video_c"], "side": ["1P"], "t_sec": [100000.0]})
    events = _events([("video_c", "1P", 5.0, 0.9, 40.0)])
    out = attach_last_event_features(df, events, tau_values=(10.0,))
    assert out[PROB_LAST_BASE_FMT.format(mode="decay_tau10")].iloc[0] == pytest.approx(NEUTRAL_PROB, abs=1e-6)
    assert out[DELTA_LAST_BASE_FMT.format(mode="decay_tau10")].iloc[0] == pytest.approx(NEUTRAL_DELTA, abs=1e-6)


def test_decay方式はイベント無しなら中立値になる() -> None:
    df = pd.DataFrame({"video_id": ["video_c"], "side": ["1P"], "t_sec": [5.0]})
    events = _events([])
    out = attach_last_event_features(df, events, tau_values=(10.0, 20.0))
    for tau in (10, 20):
        assert out[PROB_LAST_BASE_FMT.format(mode=f"decay_tau{tau}")].iloc[0] == pytest.approx(NEUTRAL_PROB)
        assert out[DELTA_LAST_BASE_FMT.format(mode=f"decay_tau{tau}")].iloc[0] == pytest.approx(NEUTRAL_DELTA)


def test_tau感度_大きいtauほど生値に近く残る単調性() -> None:
    """同じgapなら、tauが大きいほど減衰が緩やかで生値に近い値になるはず。"""
    df = pd.DataFrame({"video_id": ["video_c"], "side": ["1P"], "t_sec": [25.0]})
    events = _events([("video_c", "1P", 5.0, 1.0, 50.0)])  # gap=20秒
    out = attach_last_event_features(df, events, tau_values=(10.0, 20.0, 40.0))
    v10 = out[PROB_LAST_BASE_FMT.format(mode="decay_tau10")].iloc[0]
    v20 = out[PROB_LAST_BASE_FMT.format(mode="decay_tau20")].iloc[0]
    v40 = out[PROB_LAST_BASE_FMT.format(mode="decay_tau40")].iloc[0]
    # 生値(1.0)に近い順は tau40 > tau20 > tau10 (中立値0.5からの距離で比較)
    assert abs(v40 - 1.0) < abs(v20 - 1.0) < abs(v10 - 1.0)


def test_decay_weight関数は境界値で正しい() -> None:
    assert _decay_weight(np.array([0.0]), 10.0)[0] == pytest.approx(1.0)
    assert _decay_weight(np.array([10.0]), 10.0)[0] == pytest.approx(np.exp(-1.0))


# =============================================================================
# pair_sides_for_win との統合 (自動 _1p/_2p suffix 付与を確認)
# =============================================================================


def test_pair_sides_for_winが新列を自動でsuffix付与する() -> None:
    df = pd.DataFrame({
        "video_id": ["video_x", "video_x"],
        "game_idx": [0, 0],
        "t_sec": [100.0, 100.2],
        "side": ["1P", "2P"],
        "won": [1, 0],
    })
    events = _events([
        ("video_x", "1P", 90.0, 0.6, 10.0),
        ("video_x", "2P", 95.0, 0.4, -5.0),
    ])
    with_feats = attach_last_event_features(df, events, tau_values=(10.0,))
    paired = pair_sides_for_win(with_feats, DEFAULT_MAX_TDIFF)

    assert len(paired) == 1
    base_neutral = PROB_LAST_BASE_FMT.format(mode="neutral")
    assert f"{base_neutral}_1p" in paired.columns
    assert f"{base_neutral}_2p" in paired.columns
    assert paired[f"{base_neutral}_1p"].iloc[0] == pytest.approx(0.6)
    assert paired[f"{base_neutral}_2p"].iloc[0] == pytest.approx(0.4)

    # build_features に base名を渡すと 1p/2p/diff の3列が自動生成されること
    feat = build_features(paired, [base_neutral])
    assert set(feat.columns) == {f"{base_neutral}_1p", f"{base_neutral}_2p", f"{base_neutral}_diff"}
    assert feat[f"{base_neutral}_diff"].iloc[0] == pytest.approx(0.6 - 0.4)


# =============================================================================
# 位相割当
# =============================================================================


def test_位相割当は三分位で序中終に分かれる() -> None:
    tsumo = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 30.0, 31.0, 32.0])
    labels, q_low, q_high = assign_phase_by_tsumo_tertile(tsumo)
    assert set(labels) == {"序", "中", "終"}
    assert (labels[tsumo <= q_low] == "序").all()
    assert (labels[tsumo > q_high] == "終").all()


# =============================================================================
# AUC差CI + 所見判定 (盛らない: CIが0を跨げば「効果を確認できず」)
# =============================================================================


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_同一予測ならAUC差はゼロ近傍でCIが0を跨ぐ() -> None:
    from scripts.ablate_exchange_indicators import ConfigResult

    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    proba = rng.random(n)
    video_ids = np.array([f"v{i % 5}" for i in range(n)])
    phase_labels = np.full(n, "中", dtype=object)

    baseline = ConfigResult("baseline", 10, proba, y, video_ids, phase_labels)
    same_as_baseline = ConfigResult("same", 10, proba.copy(), y, video_ids, phase_labels)

    diff_table = build_auc_diff_table(baseline, [same_as_baseline], n_resamples=100)
    overall = diff_table.loc[diff_table["範囲"] == "全体"].iloc[0]
    assert overall["AUC差(点推定)"] == pytest.approx(0.0, abs=1e-9)
    assert overall["判定"] == "効果を確認できず (CIが0を跨ぐ)"

    verdict = _build_verdict(diff_table)
    assert "非推奨" in verdict


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_有意な悪化は改善と混同されず非推奨と判定される_回帰テスト() -> None:
    """[既知バグ修正の固定化] 2026-08-03 smoke走行で発見:

    CIが0を跨がない=「有意」を改善/悪化の区別なしに「条件付き採用推奨」と
    誤判定していた (decay_tau10がbaselineより有意に悪化していたのに
    「改善が確認された」と表示するバグ)。符号を区別し、悪化のみが有意な
    場合は必ず非推奨と判定されることを固定化する。
    """
    from scripts.ablate_exchange_indicators import ConfigResult

    rng = np.random.default_rng(1)
    n = 400
    y = rng.integers(0, 2, size=n)
    # baseline の方が明確に良い予測 (y と強く相関) / other は弱い予測にして
    # 「other - baseline」の AUC 差が確実に負かつ有意になるようにする
    proba_baseline = np.clip(y.astype(float) * 0.9 + rng.normal(0, 0.05, n), 0.0, 1.0)
    proba_worse = np.clip(y.astype(float) * 0.5 + rng.normal(0, 0.3, n), 0.0, 1.0)
    video_ids = np.array([f"v{i % 10}" for i in range(n)])
    phase_labels = np.full(n, "中", dtype=object)

    baseline = ConfigResult("baseline", 10, proba_baseline, y, video_ids, phase_labels)
    worse = ConfigResult("worse", 10, proba_worse, y, video_ids, phase_labels)

    diff_table = build_auc_diff_table(baseline, [worse], n_resamples=300)
    overall = diff_table.loc[diff_table["範囲"] == "全体"].iloc[0]
    assert overall["AUC差(点推定)"] < 0.0
    assert overall["判定"] == "有意に悪化 (CIが0より下)"

    verdict = _build_verdict(diff_table)
    assert "非推奨" in verdict
    assert "改善" not in verdict
