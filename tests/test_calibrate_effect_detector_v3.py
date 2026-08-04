"""案B閾値 統合較正 v3 (scripts/calibrate_effect_detector_v3.py) の単体テスト。

実ラベルCSV/実動画は使わず、合成データのみで
突合ロジック・行分布診断・窓スコア集約・誤発火判定を検証する。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_effect_detector_v3 import (
    BURST_ROW_MAX,
    BURST_ROW_MIN,
    EFFECT_STATE_BURST,
    EFFECT_STATE_NONE,
    EFFECT_STATE_SMOKE,
    LABEL_UNKNOWN,
    STATUS_MARKED,
    STATUS_NO_EFFECT,
    STATUS_OUT_OF_SCOPE,
    STATUS_SKIP,
    STATUS_UNLABELED,
    FrameRecord,
    band_max,
    band_topk_mean,
    best_youden_threshold,
    build_all_frame_records,
    build_burst_frame_scores,
    build_burst_roc_table,
    build_out_of_scope_misfire_table,
    classify_row_status,
    compute_true_label_row_histogram,
    decode_effect_grid_or_none,
    frame_image_path,
    has_true_effect_in_band,
    max_tpr_at_zero_fp,
    video_stem_from_id,
)
from src.board import HIDDEN_ROWS


# =============================================================================
# classify_row_status / video_stem_from_id / frame_image_path
# =============================================================================


class TestClassifyRowStatus:
    def test_blank_status_is_unlabeled(self) -> None:
        assert classify_row_status({"status": ""}) == STATUS_UNLABELED

    def test_out_of_scope_passes_through(self) -> None:
        assert classify_row_status({"status": STATUS_OUT_OF_SCOPE}) == STATUS_OUT_OF_SCOPE

    def test_missing_key_is_unlabeled(self) -> None:
        assert classify_row_status({}) == STATUS_UNLABELED


class TestVideoStemFromId:
    def test_strips_prefix(self) -> None:
        assert video_stem_from_id("video_c50") == "c50"

    def test_keeps_id_without_prefix(self) -> None:
        assert video_stem_from_id("c50") == "c50"


class TestFrameImagePath:
    def test_builds_expected_basename(self) -> None:
        row = {"video_id": "video_c50", "t_sec": "749.36", "side": "1P", "layer": "burst"}
        path = frame_image_path(row, Path("frames"))
        assert path == Path("frames/c50_t749.36_1P_burst_full.png")


# =============================================================================
# decode_effect_grid_or_none (out_of_scope等の空文字対応が第3弾の新機能)
# =============================================================================


class TestDecodeEffectGridOrNone:
    def test_empty_string_returns_none(self) -> None:
        assert decode_effect_grid_or_none("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert decode_effect_grid_or_none("   ") is None

    def test_mixed_states_decode_correctly(self) -> None:
        rows = ["100000"] + ["000000"] * 10 + ["000002"]
        grid = decode_effect_grid_or_none("/".join(rows))
        assert grid is not None
        assert grid[0, 0] == EFFECT_STATE_BURST
        assert grid[11, 5] == EFFECT_STATE_SMOKE


# =============================================================================
# max_tpr_at_zero_fp / best_youden_threshold
# =============================================================================


class TestMaxTprAtZeroFp:
    def test_perfectly_separable(self) -> None:
        pos = np.array([10.0, 11.0, 12.0])
        neg = np.array([1.0, 2.0, 3.0])
        result = max_tpr_at_zero_fp(pos, neg)
        assert result["tpr_at_zero_fp"] == 1.0
        assert result["threshold"] == 3.0

    def test_empty_pos_returns_nan(self) -> None:
        result = max_tpr_at_zero_fp(np.array([]), np.array([1.0]))
        assert np.isnan(result["tpr_at_zero_fp"])


class TestBestYoudenThreshold:
    def test_perfectly_separable_gives_zero_fpr(self) -> None:
        pos = np.array([10.0, 11.0, 12.0])
        neg = np.array([1.0, 2.0, 3.0])
        result = best_youden_threshold(pos, neg)
        assert result["tpr"] == 1.0
        assert result["fpr"] == 0.0

    def test_empty_input_returns_nan(self) -> None:
        result = best_youden_threshold(np.array([]), np.array([1.0]))
        assert np.isnan(result["tpr"])


# =============================================================================
# band_max / band_topk_mean / has_true_effect_in_band
# (burst窓 row1-3 内外の切り分けが本スクリプトの核心ロジック)
# =============================================================================


def _make_record(
    status: str = STATUS_MARKED, label_grid: np.ndarray | None = None,
    v_mean_grid: np.ndarray | None = None, layer: str = "burst",
) -> FrameRecord:
    """テスト用の合成FrameRecord (可視12行×6列)。"""
    grid = label_grid if label_grid is not None else np.zeros((12, 6), dtype=np.int64)
    v_mean = v_mean_grid if v_mean_grid is not None else np.zeros((12, 6))
    features = {
        "v_mean": v_mean, "v_max": np.zeros((12, 6)), "s_mean": np.zeros((12, 6)),
        "s_min": np.zeros((12, 6)), "specular_ratio": np.zeros((12, 6)),
        "bright_ratio": np.zeros((12, 6)),
    }
    return FrameRecord(
        source="test", video_stem="c1", side="1P", t_sec=1.0, layer=layer, note="",
        status=status, label_grid=grid, features=features,
    )


class TestBandAggregation:
    def test_band_max_picks_max_within_band_only(self) -> None:
        v_mean = np.zeros((12, 6))
        vis_row_in_band = BURST_ROW_MIN - HIDDEN_ROWS  # abs_row=BURST_ROW_MIN
        v_mean[vis_row_in_band, 0] = 200.0
        v_mean[BURST_ROW_MAX + 5 - HIDDEN_ROWS, 0] = 999.0  # 行帯外、無視されるべき
        rec = _make_record(v_mean_grid=v_mean)
        assert band_max(rec, "v_mean", BURST_ROW_MIN, BURST_ROW_MAX) == 200.0

    def test_band_topk_mean_averages_top_k(self) -> None:
        v_mean = np.zeros((12, 6))
        vis_row = BURST_ROW_MIN - HIDDEN_ROWS
        v_mean[vis_row, 0:3] = [10.0, 20.0, 30.0]
        rec = _make_record(v_mean_grid=v_mean)
        result = band_topk_mean(rec, "v_mean", BURST_ROW_MIN, BURST_ROW_MAX, k=2)
        assert result == pytest.approx((20.0 + 30.0) / 2)

    def test_empty_band_returns_nan(self) -> None:
        rec = _make_record()
        # row_min/max が可視範囲外 -> 該当セルなし
        assert np.isnan(band_max(rec, "v_mean", 100, 101))


class TestHasTrueEffectInBand:
    def test_detects_burst_cell_in_band(self) -> None:
        grid = np.zeros((12, 6), dtype=np.int64)
        grid[BURST_ROW_MIN - HIDDEN_ROWS, 0] = EFFECT_STATE_BURST
        rec = _make_record(label_grid=grid)
        assert has_true_effect_in_band(rec, EFFECT_STATE_BURST, BURST_ROW_MIN, BURST_ROW_MAX)

    def test_burst_cell_outside_band_not_detected(self) -> None:
        grid = np.zeros((12, 6), dtype=np.int64)
        grid[11, 0] = EFFECT_STATE_BURST  # abs_row=HIDDEN_ROWS+11、burst帯外
        rec = _make_record(label_grid=grid)
        assert not has_true_effect_in_band(rec, EFFECT_STATE_BURST, BURST_ROW_MIN, BURST_ROW_MAX)

    def test_unknown_label_never_counts_as_effect(self) -> None:
        grid = np.full((12, 6), LABEL_UNKNOWN, dtype=np.int64)
        rec = _make_record(label_grid=grid, status=STATUS_OUT_OF_SCOPE)
        assert not has_true_effect_in_band(rec, EFFECT_STATE_BURST, BURST_ROW_MIN, BURST_ROW_MAX)
        assert not has_true_effect_in_band(rec, EFFECT_STATE_SMOKE, BURST_ROW_MIN, BURST_ROW_MAX)


# =============================================================================
# build_burst_frame_scores / build_burst_roc_table
# =============================================================================


class TestBuildBurstFrameScores:
    def test_marked_frame_with_burst_in_band_is_positive(self) -> None:
        grid = np.zeros((12, 6), dtype=np.int64)
        grid[BURST_ROW_MIN - HIDDEN_ROWS, 0] = EFFECT_STATE_BURST
        rec = _make_record(label_grid=grid, status=STATUS_MARKED)
        df = build_burst_frame_scores([rec])
        assert bool(df.iloc[0]["is_true_burst"]) is True

    def test_marked_frame_with_smoke_only_is_negative_for_burst(self) -> None:
        grid = np.zeros((12, 6), dtype=np.int64)
        grid[BURST_ROW_MIN - HIDDEN_ROWS, 0] = EFFECT_STATE_SMOKE
        rec = _make_record(label_grid=grid, status=STATUS_MARKED)
        df = build_burst_frame_scores([rec])
        assert bool(df.iloc[0]["is_true_burst"]) is False

    def test_out_of_scope_excluded_from_roc_training(self) -> None:
        rec = _make_record(status=STATUS_OUT_OF_SCOPE)
        df = build_burst_frame_scores([rec])
        assert df.empty


class TestBuildBurstRocTable:
    def test_separable_score_gets_high_auc(self) -> None:
        v_mean_pos = np.zeros((12, 6))
        vis_row = BURST_ROW_MIN - HIDDEN_ROWS
        v_mean_pos[vis_row, 0] = 250.0
        grid_pos = np.zeros((12, 6), dtype=np.int64)
        grid_pos[vis_row, 0] = EFFECT_STATE_BURST
        rec_pos = _make_record(status=STATUS_MARKED, label_grid=grid_pos, v_mean_grid=v_mean_pos)

        v_mean_neg = np.zeros((12, 6))
        v_mean_neg[vis_row, 0] = 10.0
        rec_neg = _make_record(status=STATUS_NO_EFFECT, v_mean_grid=v_mean_neg)

        score_df = build_burst_frame_scores([rec_pos, rec_neg])
        roc = build_burst_roc_table(score_df)
        best = roc.iloc[0]
        assert best["auc"] == 1.0


# =============================================================================
# build_out_of_scope_misfire_table (テロップ判別の検証、self-chainゲートの核心)
# =============================================================================


class TestBuildOutOfScopeMisfireTable:
    def test_telop_negative_layer_flagged_as_self_chain(self) -> None:
        v_mean = np.zeros((12, 6))
        rec = _make_record(status=STATUS_OUT_OF_SCOPE, v_mean_grid=v_mean, layer="telop_negative")
        table = build_out_of_scope_misfire_table([rec], "v_mean_max", threshold=100.0)
        assert bool(table.iloc[0]["is_self_chain_scenario"]) is True

    def test_score_above_threshold_marked_fired(self) -> None:
        v_mean = np.zeros((12, 6))
        vis_row = BURST_ROW_MIN - HIDDEN_ROWS
        v_mean[vis_row, 0] = 300.0
        rec = _make_record(status=STATUS_OUT_OF_SCOPE, v_mean_grid=v_mean, layer="burst")
        table = build_out_of_scope_misfire_table([rec], "v_mean_max", threshold=100.0)
        assert bool(table.iloc[0]["fired"]) is True

    def test_skip_and_no_effect_layer_rows_excluded(self) -> None:
        rec_skip = _make_record(status=STATUS_SKIP, layer="burst")
        table = build_out_of_scope_misfire_table([rec_skip], "v_mean_max", threshold=100.0)
        assert table.empty


# =============================================================================
# compute_true_label_row_histogram (burst/smoke行帯導出の根拠データ)
# =============================================================================


class TestComputeTrueLabelRowHistogram:
    def test_counts_by_abs_row_and_label(self) -> None:
        grid = np.zeros((12, 6), dtype=np.int64)
        grid[0, 0] = EFFECT_STATE_BURST  # abs_row = HIDDEN_ROWS
        grid[5, 0] = EFFECT_STATE_SMOKE  # abs_row = HIDDEN_ROWS+5
        rec = _make_record(status=STATUS_MARKED, label_grid=grid)
        hist = compute_true_label_row_histogram([rec])
        assert hist.loc[HIDDEN_ROWS, EFFECT_STATE_BURST] == 1
        assert hist.loc[HIDDEN_ROWS + 5, EFFECT_STATE_SMOKE] == 1

    def test_non_marked_frames_excluded(self) -> None:
        grid = np.zeros((12, 6), dtype=np.int64)
        grid[0, 0] = EFFECT_STATE_BURST
        rec = _make_record(status=STATUS_NO_EFFECT, label_grid=grid)
        hist = compute_true_label_row_histogram([rec])
        assert hist.empty


# =============================================================================
# build_all_frame_records (画像との突合、skip/unlabeled除外、out_of_scope対応)
# =============================================================================


class TestBuildAllFrameRecords:
    def _write_frame(self, tmp_path: Path, name: str) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / name), frame)

    def test_out_of_scope_row_with_empty_grid_produces_unknown_labels(self, tmp_path: Path) -> None:
        self._write_frame(tmp_path, "c1_t1.00_1P_burst_full.png")
        csv_path = tmp_path / "labeling_result.csv"
        csv_path.write_text(
            "video_id,t_sec,side,layer,note,status,effect_grid,marked_cells\n"
            "video_c1,1.00,1P,burst,テスト,out_of_scope,,0\n",
            encoding="utf-8",
        )
        records, counts, warnings = build_all_frame_records(
            (("test", csv_path, tmp_path),),
        )
        assert len(records) == 1
        assert counts[STATUS_OUT_OF_SCOPE] == 1
        assert warnings == []
        assert bool(np.all(records[0].label_grid == LABEL_UNKNOWN))

    def test_skip_row_excluded_from_records(self, tmp_path: Path) -> None:
        self._write_frame(tmp_path, "c1_t1.00_1P_burst_full.png")
        csv_path = tmp_path / "labeling_result.csv"
        csv_path.write_text(
            "video_id,t_sec,side,layer,note,status,effect_grid,marked_cells\n"
            "video_c1,1.00,1P,burst,,skip,,0\n",
            encoding="utf-8",
        )
        records, counts, _warn = build_all_frame_records((("test", csv_path, tmp_path),))
        assert records == []
        assert counts[STATUS_SKIP] == 1

    def test_missing_image_warns_and_skips(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labeling_result.csv"
        csv_path.write_text(
            "video_id,t_sec,side,layer,note,status,effect_grid,marked_cells\n"
            "video_missing,1.00,1P,burst,,no_effect," + "/".join(["000000"] * 12) + ",0\n",
            encoding="utf-8",
        )
        records, _counts, warnings = build_all_frame_records((("test", csv_path, tmp_path),))
        assert records == []
        assert len(warnings) == 1
