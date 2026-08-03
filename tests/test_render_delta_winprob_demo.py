"""scripts/render_delta_winprob_demo.py のテスト (純関数のみ、cv2/動画I/Oは対象外)。

速報タイミング計算 (ignition_time_for_event) とバッジ表示ロジック
(compute_display_state) の合成テスト最小限。cv2.VideoCapture 等の実I/Oは
重い/環境依存のためユニットテスト対象から除外する(README_LABELING.md 方針同様)。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.render_delta_winprob_demo import (
    BADGE_TEXT_DISPLAY_SEC,
    FireEventView,
    _latest_event_at_or_before,
    _to_1p_view,
    build_fire_event_views,
    compute_display_state,
    ignition_time_for_event,
    select_video_segment,
    stable_value_at,
)


# =============================================================================
# ignition_time_for_event
# =============================================================================

def test_ignition_time_subtracts_chain_anim_duration() -> None:
    """t_sec から CHAIN_ANIM_PER_STEP_SEC(=0.4)*連鎖数 を引いた値になる。"""
    t = ignition_time_for_event(t_sec=100.0, approx_fire_chains=5.0)
    assert t == pytest.approx(100.0 - 0.4 * 5.0)


def test_ignition_time_zero_chains_clamped() -> None:
    """連鎖数0以下でも例外にならず t_sec そのまま(クランプ)を返す。"""
    t = ignition_time_for_event(t_sec=50.0, approx_fire_chains=0.0)
    assert t == pytest.approx(50.0)


# =============================================================================
# _to_1p_view
# =============================================================================

def test_to_1p_view_1p_side_passthrough() -> None:
    assert _to_1p_view(72.0, "1P") == pytest.approx(72.0)


def test_to_1p_view_2p_side_complement() -> None:
    assert _to_1p_view(72.0, "2P") == pytest.approx(28.0)


# =============================================================================
# stable_value_at (forward-fill)
# =============================================================================

def test_stable_value_at_before_first_sample_is_none() -> None:
    t_arr = np.array([10.0, 20.0, 30.0])
    v_arr = np.array([40.0, 60.0, 55.0])
    assert stable_value_at(t_arr, v_arr, 5.0) is None


def test_stable_value_at_forward_fill_between_samples() -> None:
    t_arr = np.array([10.0, 20.0, 30.0])
    v_arr = np.array([40.0, 60.0, 55.0])
    # 20と30の間は直近(20時点=60.0)を保持する(補間しない)
    assert stable_value_at(t_arr, v_arr, 25.0) == pytest.approx(60.0)


def test_stable_value_at_after_last_sample_holds_last() -> None:
    t_arr = np.array([10.0, 20.0, 30.0])
    v_arr = np.array([40.0, 60.0, 55.0])
    assert stable_value_at(t_arr, v_arr, 999.0) == pytest.approx(55.0)


# =============================================================================
# build_fire_event_views
# =============================================================================

def _make_events_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"t_sec": 100.0, "approx_fire_chains": 5.0, "fire_side": "1P",
         "winprob_before": 45.0, "winprob_after": 70.0, "delta_winprob": 25.0},
        {"t_sec": 120.0, "approx_fire_chains": 3.0, "fire_side": "2P",
         "winprob_before": 60.0, "winprob_after": 30.0, "delta_winprob": 30.0},
    ])


def test_build_fire_event_views_sorted_by_ignition_and_1p_normalized() -> None:
    views = build_fire_event_views(_make_events_df())
    assert len(views) == 2
    # ignition_sec 昇順であること
    assert views[0].ignition_sec < views[1].ignition_sec
    # 1件目(1P発火)はそのまま、2件目(2P発火)は補数変換されていること
    ev2p = [v for v in views if v.fire_side == "2P"][0]
    assert ev2p.winprob_before_1p == pytest.approx(100.0 - 60.0)
    assert ev2p.winprob_after_1p == pytest.approx(100.0 - 30.0)


# =============================================================================
# compute_display_state (統合的な合成テスト)
# =============================================================================

@pytest.fixture()
def sample_events() -> list[FireEventView]:
    return build_fire_event_views(_make_events_df())


@pytest.fixture()
def sample_timeline() -> tuple[np.ndarray, np.ndarray]:
    # STABLE推移: t=90で45%, t=130で70%(旧方式はここまで反映が遅れる想定)
    return np.array([90.0, 130.0]), np.array([45.0, 70.0])


def test_before_any_stable_sample_is_waiting(sample_events, sample_timeline) -> None:
    t_arr, v_arr = sample_timeline
    state = compute_display_state(sample_events, t_arr, v_arr, t=50.0)
    assert state.waiting is True
    assert state.badge is None


def test_before_ignition_shows_old_stale_value(sample_events, sample_timeline) -> None:
    """発火検知(ignition_sec)より前は旧来のSTABLE値のまま(ジャンプなし)。"""
    t_arr, v_arr = sample_timeline
    ev0 = sample_events[0]
    state = compute_display_state(sample_events, t_arr, v_arr, t=ev0.ignition_sec - 0.5)
    assert state.jump_active is False
    assert state.winprob_1p == pytest.approx(45.0)


def test_during_chain_animation_jumps_to_predicted_after(sample_events, sample_timeline) -> None:
    """発火検知後〜連鎖終了前は予測後勝率へ即座にジャンプして保持する。"""
    t_arr, v_arr = sample_timeline
    ev0 = sample_events[0]
    mid_t = (ev0.ignition_sec + ev0.fire_end_sec) / 2.0
    state = compute_display_state(sample_events, t_arr, v_arr, t=mid_t)
    assert state.jump_active is True
    assert state.winprob_1p == pytest.approx(ev0.winprob_after_1p)
    assert state.badge is not None
    assert state.badge.fire_side == "1P"


def test_badge_disappears_after_display_window(sample_events, sample_timeline) -> None:
    """バッジ表示は BADGE_TEXT_DISPLAY_SEC 経過後に消える。"""
    t_arr, v_arr = sample_timeline
    ev0 = sample_events[0]
    state = compute_display_state(
        sample_events, t_arr, v_arr, t=ev0.ignition_sec + BADGE_TEXT_DISPLAY_SEC + 1.0)
    assert state.badge is None


def test_after_chain_end_falls_back_to_real_stable_line(sample_events, sample_timeline) -> None:
    """連鎖終了(fire_end_sec)後は実測STABLE推移(データ駆動)にそのまま切り替わる。"""
    t_arr, v_arr = sample_timeline
    ev0 = sample_events[0]
    # ev0.fire_end_sec=100.0 なので、直後(t=101)は timeline側の t=90 サンプル値
    # (次のSTABLEサンプルt=130まではまだ古い値のまま=forward-fill)。
    state = compute_display_state(sample_events, t_arr, v_arr, t=ev0.fire_end_sec + 1.0)
    assert state.jump_active is False
    assert state.winprob_1p == pytest.approx(45.0)


def test_second_event_overrides_first_when_later(sample_events, sample_timeline) -> None:
    """2件目(2P発火)の検知後は2件目が最新イベントとして選ばれる。"""
    t_arr, v_arr = sample_timeline
    ev1 = sample_events[1]
    mid_t = (ev1.ignition_sec + ev1.fire_end_sec) / 2.0
    state = compute_display_state(sample_events, t_arr, v_arr, t=mid_t)
    assert state.jump_active is True
    assert state.winprob_1p == pytest.approx(ev1.winprob_after_1p)
    assert state.badge.fire_side == "2P"


def test_latest_event_at_or_before_returns_none_when_too_early(sample_events) -> None:
    ev0 = sample_events[0]
    assert _latest_event_at_or_before(sample_events, ev0.ignition_sec - 10.0) is None


# =============================================================================
# select_video_segment (バッファ・隣接試合マージン)
# =============================================================================

def _fake_cache(t_by_game: dict[int, tuple[float, float]]) -> SimpleNamespace:
    """r1p.game_idx / r1p.t_sec のみを持つ最小限の偽 npz キャッシュを作る。"""
    game_idx_list: list[int] = []
    t_list: list[float] = []
    for gi, (t0, t1) in t_by_game.items():
        game_idx_list += [gi, gi]
        t_list += [t0, t1]
    r1p = SimpleNamespace(game_idx=np.array(game_idx_list), t_sec=np.array(t_list))
    return SimpleNamespace(r1p=r1p)


def test_select_video_segment_applies_start_end_buffer() -> None:
    cache = _fake_cache({16: (1153.3, 1237.5)})
    start, end = select_video_segment(cache, 16)
    assert start == pytest.approx(1153.3 - 8.0)
    assert end == pytest.approx(1237.5 + 8.0)


def test_select_video_segment_respects_neighbor_game_guard() -> None:
    cache = _fake_cache({15: (1101.3, 1143.7), 16: (1153.3, 1237.5), 17: (1244.5, 1322.2)})
    start, end = select_video_segment(cache, 16)
    # 終了側は次試合開始(1244.5)より NEIGHBOR_GAME_GUARD_SEC(1.0) 手前で頭打ち
    assert end == pytest.approx(1244.5 - 1.0)
    # 開始側は前試合終了(1143.7)より NEIGHBOR_GAME_GUARD_SEC(1.0) 後ろが下限
    assert start == pytest.approx(1153.3 - 8.0)  # バッファの方が緩い(1144.7超え)


def test_select_video_segment_missing_game_raises() -> None:
    cache = _fake_cache({16: (1153.3, 1237.5)})
    with pytest.raises(ValueError):
        select_video_segment(cache, 99)
