"""エフェクト有無セルラベルシート第3弾準備 (scripts/build_effect_cell_label_sheet_v3.py)
の単体テスト。

実npz/実動画は使わず、合成データのみで窓検出・候補プール収集・ラウンドロビン
選定ロジックを検証する (第1弾 tests/test_build_effect_cell_label_sheet.py と対を成す)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_effect_cell_label_sheet_v3 import (
    BASELINE_TARGET_TOTAL,
    BURST_TARGET_TOTAL,
    CSV_HEADER,
    LAYER_BASELINE,
    LAYER_BURST,
    LAYER_SMOKE,
    LAYER_TELOP_NEGATIVE,
    LAYER_ZENKESHI,
    MAX_CANDIDATES_PER_VIDEO,
    SMOKE_TARGET_TOTAL,
    TELOP_NEGATIVE_TARGET_TOTAL,
    ZENKESHI_TARGET_TOTAL,
    EffectFrameCandidateV3,
    _NpzSideIndex,
    _frame_basename,
    _row_from_candidate,
    _unsafe_intervals_for_video,
    collect_baseline_pool,
    collect_burst_pool,
    collect_telop_negative_pool,
    find_ojama_increase_events,
    find_zenkeshi_events,
    round_robin_select,
    summarize_selection,
)
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA


# =============================================================================
# 定数整合性 (5層合計100 / 動画分散20本以上)
# =============================================================================


class TestConstantsConsistency:
    def test_target_totals_sum_to_one_hundred(self) -> None:
        total = (
            BURST_TARGET_TOTAL + SMOKE_TARGET_TOTAL + TELOP_NEGATIVE_TARGET_TOTAL
            + ZENKESHI_TARGET_TOTAL + BASELINE_TARGET_TOTAL
        )
        assert total == 100

    def test_max_per_video_allows_at_least_twenty_videos(self) -> None:
        total = (
            BURST_TARGET_TOTAL + SMOKE_TARGET_TOTAL + TELOP_NEGATIVE_TARGET_TOTAL
            + ZENKESHI_TARGET_TOTAL + BASELINE_TARGET_TOTAL
        )
        assert total / MAX_CANDIDATES_PER_VIDEO >= 20

    def test_telop_negative_layer_is_new_vs_first_batch(self) -> None:
        # 第1弾には無かったテロップ負例層が今回追加されている
        assert LAYER_TELOP_NEGATIVE not in (LAYER_BURST, LAYER_SMOKE, LAYER_BASELINE)


# =============================================================================
# find_ojama_increase_events (npz由来の実増加イベント検出)
# =============================================================================


def _make_npz_index(ojama_counts_per_snapshot: list[int], t_secs: list[float]) -> _NpzSideIndex:
    """可視領域のおじゃま総数だけを ojama_counts_per_snapshot で指定した合成npzを作る。"""
    n = len(ojama_counts_per_snapshot)
    grids = np.full((n, BOARD_ROWS, BOARD_COLS), COLOR_EMPTY, dtype=np.int64)
    for i, n_ojama in enumerate(ojama_counts_per_snapshot):
        flat = grids[i, 1:, :].reshape(-1)
        flat[:n_ojama] = COLOR_OJAMA
        grids[i, 1:, :] = flat.reshape(BOARD_ROWS - 1, BOARD_COLS)
    return _NpzSideIndex(
        t_secs=np.array(t_secs, dtype=np.float64), grids=grids,
        game_idxs=np.zeros(n, dtype=np.int64),
    )


class TestFindOjamaIncreaseEvents:
    def test_detects_real_increase(self) -> None:
        idx = _make_npz_index([0, 0, 6], [10.0, 20.0, 21.0])
        events = find_ojama_increase_events(idx)
        assert events == [21.0]

    def test_no_events_when_flat(self) -> None:
        idx = _make_npz_index([3, 3, 3], [10.0, 20.0, 30.0])
        assert find_ojama_increase_events(idx) == []

    def test_ignores_decrease(self) -> None:
        idx = _make_npz_index([6, 0], [10.0, 11.0])
        assert find_ojama_increase_events(idx) == []

    def test_ignores_gap_exceeding_max(self) -> None:
        idx = _make_npz_index([0, 6], [10.0, 100.0])  # gap=90秒 > SMOKE_MAX_GAP_SEC
        assert find_ojama_increase_events(idx) == []


# =============================================================================
# find_zenkeshi_events (npz由来の全消し遷移検出)
# =============================================================================


class TestFindZenkeshiEvents:
    def test_detects_transition_to_all_empty(self) -> None:
        idx = _make_npz_index([3, 0], [10.0, 11.0])
        assert find_zenkeshi_events(idx) == [11.0]

    def test_no_event_when_already_empty(self) -> None:
        idx = _make_npz_index([0, 0], [10.0, 11.0])
        assert find_zenkeshi_events(idx) == []

    def test_no_event_across_game_boundary(self) -> None:
        idx = _make_npz_index([3, 0], [10.0, 11.0])
        idx.game_idxs[:] = [0, 1]  # 試合跨ぎは全消しでない
        assert find_zenkeshi_events(idx) == []


# =============================================================================
# collect_burst_pool / collect_telop_negative_pool (窓の式)
# =============================================================================


def _make_fire_df() -> pd.DataFrame:
    return pd.DataFrame({
        "video_id": ["video_c18"], "t_sec": [100.0], "fire_side": ["1P"],
        "approx_fire_chains": [5.0], "is_synthetic_terminal_event": [0],
    })


class TestCollectBurstPool:
    def test_uses_opponent_side(self) -> None:
        rng = np.random.default_rng(1)
        out = collect_burst_pool(_make_fire_df(), rng, n_oversample=1)
        assert len(out) == 1
        assert out[0].side == "2P"  # fire_sideの相手側が受け側
        assert out[0].layer == LAYER_BURST

    def test_sample_time_within_task_specified_window(self) -> None:
        rng = np.random.default_rng(1)
        out = collect_burst_pool(_make_fire_df(), rng, n_oversample=1)
        # 窓 = [t_sec - chain*0.4 - 1, t_sec + 1] = [97.0, 101.0] (chain=5, step=0.4)
        assert 97.0 <= out[0].t_sec <= 101.0


class TestCollectTelopNegativePool:
    def test_uses_fire_side_itself(self) -> None:
        rng = np.random.default_rng(1)
        out = collect_telop_negative_pool(_make_fire_df(), rng, n_oversample=1)
        assert len(out) == 1
        assert out[0].side == "1P"  # 発火した側自身の盤面
        assert out[0].layer == LAYER_TELOP_NEGATIVE

    def test_sample_time_within_own_chain_window(self) -> None:
        rng = np.random.default_rng(1)
        out = collect_telop_negative_pool(_make_fire_df(), rng, n_oversample=1)
        # 窓 = [t_sec - chain*0.4, t_sec] = [98.0, 100.0]、post marginなし
        assert 98.0 <= out[0].t_sec <= 100.0


# =============================================================================
# _unsafe_intervals_for_video / collect_baseline_pool (平穏対照の除外ロジック)
# =============================================================================


class TestUnsafeIntervalsForVideo:
    def test_includes_fire_event_margin(self) -> None:
        fire_df = pd.DataFrame({"video_id": ["video_c18"], "t_sec": [50.0]})
        intervals = _unsafe_intervals_for_video("c18", fire_df, [], [])
        assert intervals == [(45.0, 55.0)]

    def test_includes_smoke_and_zenkeshi_pools(self) -> None:
        fire_df = pd.DataFrame({"video_id": [], "t_sec": []})
        smoke_pool = [EffectFrameCandidateV3(video_stem="c18", side="1P", t_sec=10.0, layer=LAYER_SMOKE)]
        zenkeshi_pool = [EffectFrameCandidateV3(video_stem="c18", side="2P", t_sec=20.0, layer=LAYER_ZENKESHI)]
        intervals = _unsafe_intervals_for_video("c18", fire_df, smoke_pool, zenkeshi_pool)
        assert (5.0, 15.0) in intervals
        assert (15.0, 25.0) in intervals


class TestCollectBaselinePool:
    def test_excludes_unsafe_snapshots(self) -> None:
        # t=50 は発火イベント(t=50)の±5秒以内で不安全、t=200は安全
        idx = _make_npz_index([0, 0], [50.0, 200.0])

        def fake_loader(stem: str, side: str):
            return idx

        import scripts.build_effect_cell_label_sheet_v3 as sheet_mod
        original = sheet_mod.load_npz_side_index
        sheet_mod.load_npz_side_index = fake_loader
        try:
            fire_df = pd.DataFrame({"video_id": ["video_c18"], "t_sec": [50.0]})
            rng = np.random.default_rng(1)
            out = collect_baseline_pool(["c18"], fire_df, [], [], rng, n_oversample=10)
        finally:
            sheet_mod.load_npz_side_index = original
        assert all(c.t_sec == 200.0 for c in out)
        assert all(c.layer == LAYER_BASELINE for c in out)


# =============================================================================
# round_robin_select (動画あたり上限・偏り抑制、第1弾から流用)
# =============================================================================


def _make_candidates(video_stems: list[str], layer: str = LAYER_BURST) -> list[EffectFrameCandidateV3]:
    return [
        EffectFrameCandidateV3(video_stem=v, side="1P", t_sec=float(i), layer=layer)
        for i, v in enumerate(video_stems)
    ]


class TestRoundRobinSelect:
    def test_respects_max_per_video_cap(self) -> None:
        pool = _make_candidates(["c1"] * 5)
        selected = round_robin_select(pool, n_want=10, max_per_video=2, usage={})
        assert len(selected) == 2

    def test_spreads_across_videos(self) -> None:
        pool = _make_candidates(["c1", "c2", "c3", "c1", "c2", "c3"])
        selected = round_robin_select(pool, n_want=3, max_per_video=2, usage={})
        assert len({c.video_stem for c in selected}) == 3

    def test_shared_usage_dict_persists_across_calls(self) -> None:
        usage: dict[str, int] = {}
        pool_a = _make_candidates(["c1"] * 3, layer=LAYER_BURST)
        pool_b = _make_candidates(["c1"] * 3, layer=LAYER_SMOKE)
        selected_a = round_robin_select(pool_a, n_want=3, max_per_video=2, usage=usage)
        selected_b = round_robin_select(pool_b, n_want=3, max_per_video=2, usage=usage)
        assert len(selected_a) == 2
        assert len(selected_b) == 0

    def test_empty_pool_returns_empty(self) -> None:
        assert round_robin_select([], n_want=5, max_per_video=2, usage={}) == []


# =============================================================================
# _frame_basename / _row_from_candidate
# =============================================================================


class TestFrameBasename:
    def test_includes_video_time_side_layer(self) -> None:
        c = EffectFrameCandidateV3(video_stem="c18", side="2P", t_sec=101.3, layer=LAYER_BURST)
        assert _frame_basename(c) == "c18_t101.30_2P_burst"

    def test_different_layers_produce_different_basenames(self) -> None:
        base = dict(video_stem="c18", side="1P", t_sec=1.0)
        burst = EffectFrameCandidateV3(layer=LAYER_BURST, **base)
        telop = EffectFrameCandidateV3(layer=LAYER_TELOP_NEGATIVE, **base)
        assert _frame_basename(burst) != _frame_basename(telop)


class TestRowFromCandidate:
    def test_marks_failed_fetch_with_placeholder(self) -> None:
        c = EffectFrameCandidateV3(video_stem="c18", side="1P", t_sec=1.0, layer=LAYER_BASELINE)
        row = _row_from_candidate(c, None, None)
        assert row["image_full_frame"] == "(取得失敗)"
        assert row["image_board_crop"] == "(取得失敗)"

    def test_row_matches_csv_header_keys(self) -> None:
        c = EffectFrameCandidateV3(video_stem="c18", side="1P", t_sec=1.0, layer=LAYER_BURST, note="test")
        row = _row_from_candidate(c, None, None)
        assert set(row.keys()) == set(CSV_HEADER)


# =============================================================================
# summarize_selection
# =============================================================================


class TestSummarizeSelection:
    def test_reports_layer_and_video_breakdown(self) -> None:
        selected = _make_candidates(["c1", "c2", "c1"], layer=LAYER_BURST)
        summary = summarize_selection(selected)
        assert "選定候補数: 3 件" in summary
        assert "動画数: 2 本" in summary
        assert LAYER_BURST in summary
