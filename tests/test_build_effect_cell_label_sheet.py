"""エフェクト有無セルラベルシート準備 (scripts/build_effect_cell_label_sheet.py) の単体テスト。

実npz/実動画は使わず、合成データのみで候補集約・ラウンドロビン選定ロジックを
検証する (study_effect_signature側の窓検出そのものは調査専用スクリプトの
既存責務であり、本テストの対象外)。
"""
from __future__ import annotations

from dataclasses import dataclass

from scripts.build_effect_cell_label_sheet import (
    BASELINE_TARGET_TOTAL,
    BURST_TARGET_PER_BIN,
    CSV_HEADER,
    EFFECT_LABEL_VIDEO_STEMS,
    EXTRA_VIDEO_STEMS,
    LAYER_BASELINE,
    LAYER_BURST,
    LAYER_SMOKE,
    MAX_CANDIDATES_PER_VIDEO,
    SMOKE_TARGET_TOTAL,
    STUDY_VIDEO_STEMS,
    EffectFrameCandidate,
    _frame_basename,
    _row_from_candidate,
    group_records_to_frames,
    round_robin_select,
    summarize_selection,
)


@dataclass
class _FakeCellRecord:
    """study.CellRecordの必要フィールドだけを持つダミー (テスト専用)。"""

    video_stem: str
    side: str
    t_sec: float
    chain_bin: str = ""


# =============================================================================
# group_records_to_frames (フレーム単位の重複排除)
# =============================================================================


class TestGroupRecordsToFrames:
    def test_dedupes_same_frame_cells(self) -> None:
        # 同じ (video,side,t_sec) の18セル分レコードは1フレームに集約される
        records = [_FakeCellRecord("c18", "2P", 100.0) for _ in range(18)]
        frames = group_records_to_frames(records, LAYER_BURST)
        assert len(frames) == 1
        assert frames[0].layer == LAYER_BURST

    def test_distinguishes_different_t_sec(self) -> None:
        records = [_FakeCellRecord("c18", "2P", 100.0), _FakeCellRecord("c18", "2P", 200.0)]
        frames = group_records_to_frames(records, LAYER_SMOKE)
        assert len(frames) == 2

    def test_distinguishes_different_chain_bin(self) -> None:
        # 同じ時刻でも連鎖規模帯が違えば別窓として扱う
        records = [
            _FakeCellRecord("c18", "2P", 100.001, chain_bin="2-3"),
            _FakeCellRecord("c18", "2P", 100.001, chain_bin="4-6"),
        ]
        frames = group_records_to_frames(records, LAYER_BURST)
        assert len(frames) == 2

    def test_rounds_t_sec_to_avoid_float_noise_duplicates(self) -> None:
        records = [_FakeCellRecord("c18", "2P", 100.001), _FakeCellRecord("c18", "2P", 100.004)]
        frames = group_records_to_frames(records, LAYER_BASELINE)
        assert len(frames) == 1

    def test_empty_records_returns_empty(self) -> None:
        assert group_records_to_frames([], LAYER_BURST) == []


# =============================================================================
# round_robin_select (動画あたり上限・偏り抑制)
# =============================================================================


def _make_candidates(video_stems: list[str], layer: str = LAYER_BURST) -> list[EffectFrameCandidate]:
    return [
        EffectFrameCandidate(video_stem=v, side="1P", t_sec=float(i), layer=layer, chain_bin="")
        for i, v in enumerate(video_stems)
    ]


class TestRoundRobinSelect:
    def test_respects_max_per_video_cap(self) -> None:
        pool = _make_candidates(["c1"] * 5)
        usage: dict[str, int] = {}
        selected = round_robin_select(pool, n_want=10, max_per_video=2, usage=usage)
        assert len(selected) == 2

    def test_spreads_across_videos(self) -> None:
        pool = _make_candidates(["c1", "c2", "c3", "c1", "c2", "c3"])
        usage: dict[str, int] = {}
        selected = round_robin_select(pool, n_want=3, max_per_video=2, usage=usage)
        by_video = {c.video_stem for c in selected}
        assert len(by_video) == 3  # 1動画に偏らず3動画全てから1件ずつ

    def test_shared_usage_dict_persists_across_calls(self) -> None:
        # 複数レイヤーを跨いで同じusage辞書を使うと、合計上限が守られる
        usage: dict[str, int] = {}
        pool_a = _make_candidates(["c1"] * 3, layer=LAYER_BURST)
        pool_b = _make_candidates(["c1"] * 3, layer=LAYER_SMOKE)
        selected_a = round_robin_select(pool_a, n_want=3, max_per_video=2, usage=usage)
        selected_b = round_robin_select(pool_b, n_want=3, max_per_video=2, usage=usage)
        assert len(selected_a) == 2
        assert len(selected_b) == 0  # 既に上限2件消化済み

    def test_empty_pool_returns_empty(self) -> None:
        assert round_robin_select([], n_want=5, max_per_video=2, usage={}) == []

    def test_insufficient_pool_degrades_gracefully(self) -> None:
        pool = _make_candidates(["c1", "c2"])
        selected = round_robin_select(pool, n_want=10, max_per_video=5, usage={})
        assert len(selected) == 2


# =============================================================================
# _frame_basename / _row_from_candidate
# =============================================================================


class TestFrameBasename:
    def test_includes_video_time_side_layer(self) -> None:
        c = EffectFrameCandidate(video_stem="c18", side="2P", t_sec=101.3, layer=LAYER_BURST, chain_bin="2-3")
        assert _frame_basename(c) == "c18_t101.30_2P_burst"

    def test_different_layers_produce_different_basenames(self) -> None:
        base_kwargs = dict(video_stem="c18", side="1P", t_sec=1.0, chain_bin="")
        burst = EffectFrameCandidate(layer=LAYER_BURST, **base_kwargs)
        smoke = EffectFrameCandidate(layer=LAYER_SMOKE, **base_kwargs)
        assert _frame_basename(burst) != _frame_basename(smoke)


class TestRowFromCandidate:
    def test_marks_failed_fetch_with_placeholder(self) -> None:
        c = EffectFrameCandidate(video_stem="c18", side="1P", t_sec=1.0, layer=LAYER_BASELINE, chain_bin="")
        row = _row_from_candidate(c, None, None)
        assert row["image_full_frame"] == "(取得失敗)"
        assert row["image_board_crop"] == "(取得失敗)"

    def test_row_matches_csv_header_keys(self) -> None:
        c = EffectFrameCandidate(video_stem="c18", side="1P", t_sec=1.0, layer=LAYER_BURST, chain_bin="2-3")
        row = _row_from_candidate(c, None, None)
        assert set(row.keys()) == set(CSV_HEADER)


# =============================================================================
# 定数の整合性 (動画分散・件数配分)
# =============================================================================


class TestConstantsConsistency:
    def test_video_pool_has_no_duplicates(self) -> None:
        assert len(EFFECT_LABEL_VIDEO_STEMS) == len(set(EFFECT_LABEL_VIDEO_STEMS))

    def test_video_pool_is_study_plus_extra(self) -> None:
        assert set(EFFECT_LABEL_VIDEO_STEMS) == set(STUDY_VIDEO_STEMS) | set(EXTRA_VIDEO_STEMS)

    def test_extra_videos_are_new_not_overlapping_study(self) -> None:
        # 効果調査と同じ動画に偏らないよう新規動画を含める、という要件の検証
        assert set(EXTRA_VIDEO_STEMS).isdisjoint(set(STUDY_VIDEO_STEMS))
        assert len(EXTRA_VIDEO_STEMS) >= 1

    def test_pool_has_at_least_six_videos(self) -> None:
        assert len(EFFECT_LABEL_VIDEO_STEMS) >= 6

    def test_target_totals_sum_to_forty(self) -> None:
        assert BURST_TARGET_PER_BIN * 3 + SMOKE_TARGET_TOTAL + BASELINE_TARGET_TOTAL == 40

    def test_max_per_video_allows_at_least_six_distinct_videos(self) -> None:
        total = BURST_TARGET_PER_BIN * 3 + SMOKE_TARGET_TOTAL + BASELINE_TARGET_TOTAL
        assert total / MAX_CANDIDATES_PER_VIDEO >= 6


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
