"""案B閾値本較正 (scripts/calibrate_effect_detector.py) の単体テスト。

実ラベルCSV/実動画は使わず、合成データのみで突合ロジック・ROC動作点計算・
窓レベル集計を検証する。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_effect_detector import (
    EFFECT_STATE_BURST,
    EFFECT_STATE_NONE,
    EFFECT_STATE_SMOKE,
    STATUS_MARKED,
    STATUS_NO_EFFECT,
    STATUS_SKIP,
    STATUS_UNLABELED,
    LabeledCellSample,
    build_labeled_cell_samples,
    classify_row_status,
    compute_frame_level_auc,
    compute_ojama_vs_smoke_separability,
    compute_window_level_flag_counts,
    count_effect_cells_by_layer,
    decode_effect_grid,
    frame_image_path,
    max_tpr_at_zero_fp,
    samples_to_dataframe,
    summarize_frame_status,
    video_stem_from_id,
)
from src.board import HIDDEN_ROWS


# =============================================================================
# classify_row_status / video_stem_from_id / frame_image_path
# =============================================================================


class TestClassifyRowStatus:
    def test_blank_status_is_unlabeled(self) -> None:
        assert classify_row_status({"status": ""}) == STATUS_UNLABELED

    def test_missing_key_is_unlabeled(self) -> None:
        assert classify_row_status({}) == STATUS_UNLABELED

    def test_whitespace_only_is_unlabeled(self) -> None:
        assert classify_row_status({"status": "  "}) == STATUS_UNLABELED

    def test_explicit_status_passes_through(self) -> None:
        assert classify_row_status({"status": STATUS_NO_EFFECT}) == STATUS_NO_EFFECT
        assert classify_row_status({"status": STATUS_MARKED}) == STATUS_MARKED
        assert classify_row_status({"status": STATUS_SKIP}) == STATUS_SKIP


class TestVideoStemFromId:
    def test_strips_prefix(self) -> None:
        assert video_stem_from_id("video_c18") == "c18"

    def test_keeps_id_without_prefix(self) -> None:
        assert video_stem_from_id("c18") == "c18"


class TestFrameImagePath:
    def test_builds_expected_basename(self) -> None:
        row = {"video_id": "video_c18", "t_sec": "100.00", "side": "2P", "layer": "burst"}
        path = frame_image_path(row, Path("frames"))
        assert path == Path("frames/c18_t100.00_2P_burst_full.png")


# =============================================================================
# decode_effect_grid (encode_effect_grid の逆変換、JS側encodeGridStringと対)
# =============================================================================


class TestDecodeEffectGrid:
    def test_all_zero_grid(self) -> None:
        encoded = "/".join(["000000"] * 12)
        grid = decode_effect_grid(encoded)
        assert grid.shape == (12, 6)
        assert np.all(grid == EFFECT_STATE_NONE)

    def test_mixed_states_decode_correctly(self) -> None:
        rows = ["100000"] + ["000000"] * 10 + ["000002"]
        grid = decode_effect_grid("/".join(rows))
        assert grid[0, 0] == EFFECT_STATE_BURST
        assert grid[11, 5] == EFFECT_STATE_SMOKE
        assert grid[5, 3] == EFFECT_STATE_NONE


# =============================================================================
# max_tpr_at_zero_fp (ゼロFP保証・方向自動判定)
# =============================================================================


class TestMaxTprAtZeroFp:
    def test_perfectly_separable_ascending(self) -> None:
        pos = np.array([10.0, 11.0, 12.0])
        neg = np.array([1.0, 2.0, 3.0])
        result = max_tpr_at_zero_fp(pos, neg)
        assert result["tpr_at_zero_fp"] == 1.0
        assert result["threshold"] == 3.0

    def test_perfectly_separable_descending_direction(self) -> None:
        # 特徴量が「小さいほど陽性」の向きでも自動判定できる
        pos = np.array([1.0, 2.0, 3.0])
        neg = np.array([10.0, 11.0, 12.0])
        result = max_tpr_at_zero_fp(pos, neg)
        assert result["tpr_at_zero_fp"] == 1.0
        assert result["direction"] == -1.0

    def test_overlapping_distributions_reduces_tpr(self) -> None:
        pos = np.array([1.0, 5.0, 10.0])
        neg = np.array([2.0, 3.0, 4.0])
        result = max_tpr_at_zero_fp(pos, neg)
        # neg最大値4.0を超えるのはpos中5.0と10.0 (1.0は超えない)
        assert result["tpr_at_zero_fp"] == pytest.approx(2 / 3)

    def test_empty_pos_returns_nan(self) -> None:
        result = max_tpr_at_zero_fp(np.array([]), np.array([1.0, 2.0]))
        assert np.isnan(result["tpr_at_zero_fp"])

    def test_empty_neg_returns_nan(self) -> None:
        result = max_tpr_at_zero_fp(np.array([1.0]), np.array([]))
        assert np.isnan(result["tpr_at_zero_fp"])


# =============================================================================
# summarize_frame_status / count_effect_cells_by_layer
# =============================================================================


class TestSummarizeFrameStatus:
    def test_counts_by_layer_and_status(self) -> None:
        rows = [
            {"layer": "burst", "status": "no_effect"},
            {"layer": "burst", "status": "marked"},
            {"layer": "burst", "status": ""},
            {"layer": "smoke", "status": "skip"},
        ]
        table = summarize_frame_status(rows)
        assert table.loc["burst", STATUS_NO_EFFECT] == 1
        assert table.loc["burst", STATUS_MARKED] == 1
        assert table.loc["burst", STATUS_UNLABELED] == 1
        assert table.loc["smoke", STATUS_SKIP] == 1

    def test_missing_status_columns_filled_with_zero(self) -> None:
        rows = [{"layer": "baseline", "status": "no_effect"}]
        table = summarize_frame_status(rows)
        assert table.loc["baseline", STATUS_MARKED] == 0
        assert table.loc["baseline", STATUS_SKIP] == 0


def _make_sample(layer_hint: str = "burst", label: int = 0) -> LabeledCellSample:
    return LabeledCellSample(
        video_stem="c1", side="1P", t_sec=1.0, layer_hint=layer_hint, chain_bin="",
        frame_status="marked", row=1, col=0, label=label,
        v_mean=100.0, v_max=150.0, s_mean=80.0, s_min=50.0,
        specular_ratio=0.0, bright_ratio=0.0,
    )


class TestCountEffectCellsByLayer:
    def test_pivots_layer_by_label(self) -> None:
        samples = [
            _make_sample("burst", EFFECT_STATE_NONE),
            _make_sample("burst", EFFECT_STATE_BURST),
            _make_sample("smoke", EFFECT_STATE_SMOKE),
        ]
        table = count_effect_cells_by_layer(samples)
        assert table.loc["burst", EFFECT_STATE_NONE] == 1
        assert table.loc["burst", EFFECT_STATE_BURST] == 1
        assert table.loc["smoke", EFFECT_STATE_SMOKE] == 1

    def test_empty_samples_returns_empty_dataframe(self) -> None:
        assert count_effect_cells_by_layer([]).empty


# =============================================================================
# compute_ojama_vs_smoke_separability
# =============================================================================


class TestOjamaVsSmokeSeparability:
    def test_separable_feature_gets_high_auc(self) -> None:
        samples = [
            LabeledCellSample(
                video_stem="c1", side="1P", t_sec=float(i), layer_hint="smoke", chain_bin="",
                frame_status="marked", row=1, col=0, label=EFFECT_STATE_SMOKE,
                v_mean=200.0 + i, v_max=0, s_mean=0, s_min=0, specular_ratio=0, bright_ratio=0,
            )
            for i in range(5)
        ]
        df = samples_to_dataframe(samples)
        study_df = pd.DataFrame({
            "ground_truth_color": [9] * 5,
            "v_mean": [10.0, 11.0, 12.0, 13.0, 14.0],
            "v_max": [0] * 5, "s_mean": [0] * 5, "s_min": [0] * 5,
            "specular_ratio": [0] * 5, "bright_ratio": [0] * 5,
        })
        table = compute_ojama_vs_smoke_separability(df, study_df)
        v_mean_row = table.loc[table["feature"] == "v_mean"].iloc[0]
        assert v_mean_row["auc"] == 1.0

    def test_no_smoke_samples_returns_nan_auc(self) -> None:
        samples = [_make_sample("burst", EFFECT_STATE_BURST)]
        df = samples_to_dataframe(samples)
        study_df = pd.DataFrame({
            "ground_truth_color": [9], "v_mean": [10.0], "v_max": [0],
            "s_mean": [0], "s_min": [0], "specular_ratio": [0], "bright_ratio": [0],
        })
        table = compute_ojama_vs_smoke_separability(df, study_df)
        assert table["auc"].isna().all()


# =============================================================================
# compute_window_level_flag_counts / compute_frame_level_auc
# =============================================================================


class TestComputeWindowLevelFlagCounts:
    def _band_sample(self, video_stem: str, row: int, v_mean: float, label: int, status: str) -> LabeledCellSample:
        return LabeledCellSample(
            video_stem=video_stem, side="1P", t_sec=1.0, layer_hint="burst", chain_bin="",
            frame_status=status, row=row, col=0, label=label,
            v_mean=v_mean, v_max=0, s_mean=0, s_min=0, specular_ratio=0, bright_ratio=0,
        )

    def test_counts_flagged_cells_within_burst_band(self) -> None:
        samples = [
            self._band_sample("c1", row=1, v_mean=200.0, label=EFFECT_STATE_BURST, status="marked"),
            self._band_sample("c1", row=2, v_mean=10.0, label=EFFECT_STATE_NONE, status="marked"),
            self._band_sample("c1", row=8, v_mean=200.0, label=EFFECT_STATE_NONE, status="marked"),  # 行帯外
        ]
        df = samples_to_dataframe(samples)
        result = compute_window_level_flag_counts(df, feature="v_mean", threshold=100.0)
        assert len(result) == 1
        assert result.iloc[0]["n_flagged_in_band"] == 1  # row8は行帯外なので数えない
        assert bool(result.iloc[0]["has_true_effect_in_band"]) is True

    def test_no_effect_frame_marked_false(self) -> None:
        samples = [
            self._band_sample("c2", row=1, v_mean=5.0, label=EFFECT_STATE_NONE, status="no_effect"),
        ]
        df = samples_to_dataframe(samples)
        result = compute_window_level_flag_counts(df, feature="v_mean", threshold=100.0)
        assert bool(result.iloc[0]["has_true_effect_in_band"]) is False
        assert result.iloc[0]["n_flagged_in_band"] == 0


class TestComputeFrameLevelAuc:
    def test_perfect_separation_gives_auc_one(self) -> None:
        counts = pd.DataFrame({
            "n_flagged_in_band": [10, 12, 1, 2],
            "has_true_effect_in_band": [True, True, False, False],
        })
        result = compute_frame_level_auc(counts)
        assert result["auc"] == 1.0
        assert result["zero_fp_min_flagged_cells"] == 3  # neg最大2+1

    def test_single_class_returns_nan(self) -> None:
        counts = pd.DataFrame({
            "n_flagged_in_band": [1, 2, 3],
            "has_true_effect_in_band": [False, False, False],
        })
        result = compute_frame_level_auc(counts)
        assert np.isnan(result["auc"])


# =============================================================================
# build_labeled_cell_samples (画像との突合、absolute row マッピング確認)
# =============================================================================


class TestBuildLabeledCellSamples:
    def _write_frame(self, tmp_path: Path, name: str) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / name), frame)

    def test_usable_row_produces_72_cells(self, tmp_path: Path) -> None:
        self._write_frame(tmp_path, "c1_t1.00_1P_burst_full.png")
        rows = [{
            "video_id": "video_c1", "t_sec": "1.00", "side": "1P", "layer": "burst",
            "chain_bin": "2-3", "status": "no_effect",
            "effect_grid": "/".join(["000000"] * 12),
        }]
        samples, status_counts, warnings = build_labeled_cell_samples(rows, tmp_path)
        assert len(samples) == 12 * 6
        assert status_counts[STATUS_NO_EFFECT] == 1
        assert warnings == []

    def test_absolute_row_offset_by_hidden_rows(self, tmp_path: Path) -> None:
        self._write_frame(tmp_path, "c1_t1.00_1P_burst_full.png")
        rows = [{
            "video_id": "video_c1", "t_sec": "1.00", "side": "1P", "layer": "burst",
            "chain_bin": "", "status": "marked",
            "effect_grid": "/".join(["000000"] * 12),
        }]
        samples, _counts, _warn = build_labeled_cell_samples(rows, tmp_path)
        rows_seen = sorted({s.row for s in samples})
        # 可視12行 (vis_row 0-11) は 絶対行 HIDDEN_ROWS..HIDDEN_ROWS+11 になるはず
        assert rows_seen == list(range(HIDDEN_ROWS, HIDDEN_ROWS + 12))

    def test_skip_and_unlabeled_rows_are_excluded(self, tmp_path: Path) -> None:
        self._write_frame(tmp_path, "c1_t1.00_1P_burst_full.png")
        rows = [
            {"video_id": "video_c1", "t_sec": "1.00", "side": "1P", "layer": "burst",
             "chain_bin": "", "status": "skip", "effect_grid": ""},
            {"video_id": "video_c1", "t_sec": "2.00", "side": "1P", "layer": "burst",
             "chain_bin": "", "status": "", "effect_grid": "/".join(["000000"] * 12)},
        ]
        samples, status_counts, _warn = build_labeled_cell_samples(rows, tmp_path)
        assert samples == []
        assert status_counts[STATUS_SKIP] == 1
        assert status_counts[STATUS_UNLABELED] == 1

    def test_missing_image_warns_and_skips(self, tmp_path: Path) -> None:
        rows = [{
            "video_id": "video_missing", "t_sec": "1.00", "side": "1P", "layer": "burst",
            "chain_bin": "", "status": "no_effect", "effect_grid": "/".join(["000000"] * 12),
        }]
        samples, _counts, warnings = build_labeled_cell_samples(rows, tmp_path)
        assert samples == []
        assert len(warnings) == 1

    def test_label_values_round_trip_into_samples(self, tmp_path: Path) -> None:
        self._write_frame(tmp_path, "c1_t1.00_1P_smoke_full.png")
        grid_rows = ["100000"] + ["000000"] * 10 + ["000002"]
        rows = [{
            "video_id": "video_c1", "t_sec": "1.00", "side": "1P", "layer": "smoke",
            "chain_bin": "", "status": "marked", "effect_grid": "/".join(grid_rows),
        }]
        samples, _counts, _warn = build_labeled_cell_samples(rows, tmp_path)
        labels_present = {s.label for s in samples}
        assert labels_present == {EFFECT_STATE_NONE, EFFECT_STATE_BURST, EFFECT_STATE_SMOKE}
