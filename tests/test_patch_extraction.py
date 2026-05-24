"""
patch_extraction.py のテスト

PatchDataset / PatchExtractor / balance_dataset について:
- side メタ (1P/2P) が extract_from_frame_with_sides で正しく付くか
- npz save/load の後方互換 (sides キー有無)
- balance_dataset(stratify_by_side=True) の層別化動作
を検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_PURPLE,
    COLOR_RED,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
)
from src.calibration import CalibratedConfig
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.patch_extraction import (
    PATCH_DATASET_FORMAT,
    SIDE_1P,
    SIDE_2P,
    SIDE_UNKNOWN,
    ExtractionStats,
    PatchDataset,
    PatchExtractor,
    balance_dataset,
)

from tests.fixtures import make_synthetic_frame, sample_all_colors_board


# ============================
# PatchDataset: sides メタ
# ============================


class TestPatchDatasetSides:
    def test_sides_defaults_to_zeros_when_omitted(self):
        """sides 省略時は全 unknown (0) で埋まる後方互換。"""
        patches = np.zeros((5, 4, 4, 3), dtype=np.uint8)
        labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        ds = PatchDataset(patches=patches, labels=labels)
        assert ds.sides is not None
        assert ds.sides.shape == (5,)
        assert ds.sides.dtype == np.int8
        assert np.all(ds.sides == SIDE_UNKNOWN)

    def test_sides_length_mismatch_raises(self):
        patches = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        labels = np.array([0, 1, 2], dtype=np.int64)
        bad_sides = np.array([1, 2], dtype=np.int8)  # 長さ不一致
        with pytest.raises(ValueError):
            PatchDataset(patches=patches, labels=labels, sides=bad_sides)

    def test_sides_dtype_normalized_to_int8(self):
        patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        labels = np.array([1, 2], dtype=np.int64)
        sides_in = np.array([SIDE_1P, SIDE_2P], dtype=np.int64)
        ds = PatchDataset(patches=patches, labels=labels, sides=sides_in)
        assert ds.sides.dtype == np.int8


# ============================
# npz 永続化: 後方互換
# ============================


class TestPatchDatasetIO:
    def test_save_and_load_with_sides(self, tmp_path: Path):
        patches = (np.random.rand(4, 8, 8, 3) * 255).astype(np.uint8)
        labels = np.array([0, 1, COLOR_PURPLE, 2], dtype=np.int64)
        sides = np.array([SIDE_1P, SIDE_1P, SIDE_2P, SIDE_2P], dtype=np.int8)
        ds = PatchDataset(patches=patches, labels=labels, sides=sides)
        path = tmp_path / "ds.npz"
        ds.save(path)

        loaded = PatchDataset.load(path)
        assert loaded.patches.shape == patches.shape
        np.testing.assert_array_equal(loaded.labels, labels)
        np.testing.assert_array_equal(loaded.sides, sides)

    def test_load_legacy_npz_without_sides_key(self, tmp_path: Path):
        """
        旧 npz (sides キー無し) をロードすると、sides は全 unknown
        で埋まり例外は発生しない。
        """
        path = tmp_path / "legacy.npz"
        patches = (np.random.rand(3, 8, 8, 3) * 255).astype(np.uint8)
        labels = np.array([COLOR_EMPTY, COLOR_RED, COLOR_BLUE], dtype=np.int64)
        # 旧フォーマット: sides キーを敢えて入れない
        np.savez_compressed(
            path,
            format=np.array([PATCH_DATASET_FORMAT]),
            patches=patches,
            labels=labels,
            frames_sampled=np.array([0]),
        )

        loaded = PatchDataset.load(path)
        assert loaded.sides is not None
        assert loaded.sides.shape == (3,)
        assert np.all(loaded.sides == SIDE_UNKNOWN)
        np.testing.assert_array_equal(loaded.labels, labels)


# ============================
# PatchExtractor: side 付与
# ============================


def _make_extractor() -> PatchExtractor:
    config = CalibratedConfig(
        p1_region=DEFAULT_P1_REGION,
        p2_region=DEFAULT_P2_REGION,
    )
    return PatchExtractor(config=config)


class TestPatchExtractorSides:
    def test_extract_from_frame_with_sides_shape(self):
        extractor = _make_extractor()
        board = sample_all_colors_board()
        frame = make_synthetic_frame(board_1p=board, board_2p=board)

        patches, labels, sides = extractor.extract_from_frame_with_sides(
            frame,
        )
        expected = 2 * VISIBLE_ROWS * BOARD_COLS
        assert len(patches) == expected
        assert len(labels) == expected
        assert len(sides) == expected

    def test_extract_from_frame_with_sides_first_half_is_1p(self):
        """前半 72 個が 1P、後半 72 個が 2P の順序保証。"""
        extractor = _make_extractor()
        board = sample_all_colors_board()
        frame = make_synthetic_frame(board_1p=board, board_2p=board)

        _, _, sides = extractor.extract_from_frame_with_sides(frame)
        half = VISIBLE_ROWS * BOARD_COLS
        assert all(s == SIDE_1P for s in sides[:half])
        assert all(s == SIDE_2P for s in sides[half:])

    def test_extract_from_frame_backward_compat(self):
        """既存 extract_from_frame は (patches, labels) 2-tuple のまま。"""
        extractor = _make_extractor()
        board = sample_all_colors_board()
        frame = make_synthetic_frame(board_1p=board, board_2p=board)

        result = extractor.extract_from_frame(frame)
        assert isinstance(result, tuple)
        assert len(result) == 2  # 後方互換: 3 要素に膨らんでないこと
        patches, labels = result
        assert len(patches) == len(labels)


# ============================
# balance_dataset: デフォルト挙動の不変
# ============================


def _synth_dataset(
    n_per_side_per_label: dict[int, dict[int, int]],
) -> PatchDataset:
    """
    side→label→件数 の指定から合成データセットを作る。
    n_per_side_per_label[side][label] = 件数
    """
    all_patches: list[np.ndarray] = []
    all_labels: list[int] = []
    all_sides: list[int] = []
    for side_val, label_counts in n_per_side_per_label.items():
        for lbl, cnt in label_counts.items():
            for _ in range(cnt):
                all_patches.append(np.zeros((4, 4, 3), dtype=np.uint8))
                all_labels.append(lbl)
                all_sides.append(side_val)
    patches = (
        np.stack(all_patches) if all_patches
        else np.zeros((0, 4, 4, 3), dtype=np.uint8)
    )
    labels = np.array(all_labels, dtype=np.int64)
    sides = np.array(all_sides, dtype=np.int8)
    return PatchDataset(patches=patches, labels=labels, sides=sides)


class TestBalanceDatasetDefault:
    def test_default_behavior_unchanged_with_new_flag_false(self):
        """stratify_by_side 未指定 = False でも従来挙動。"""
        ds = _synth_dataset({
            SIDE_1P: {COLOR_EMPTY: 100, COLOR_RED: 20, COLOR_BLUE: 20},
            SIDE_2P: {COLOR_EMPTY: 100, COLOR_RED: 20, COLOR_BLUE: 20},
        })
        balanced = balance_dataset(ds, empty_ratio_cap=0.35, seed=0)
        # 空比率が 0.35 を超えない
        empty_ratio = float(
            (balanced.labels == COLOR_EMPTY).mean()
        ) if len(balanced.labels) > 0 else 0.0
        assert empty_ratio <= 0.36  # 整数丸め誤差を許容

    def test_explicit_false_equals_default(self):
        ds = _synth_dataset({
            SIDE_1P: {COLOR_EMPTY: 30, COLOR_RED: 10},
            SIDE_2P: {COLOR_EMPTY: 30, COLOR_RED: 10},
        })
        b1 = balance_dataset(ds, seed=42, stratify_by_side=False)
        b2 = balance_dataset(ds, seed=42)
        # ラベル分布が同一 (順序はシャッフルされるため set で比較)
        l1 = sorted(b1.labels.tolist())
        l2 = sorted(b2.labels.tolist())
        assert l1 == l2


# ============================
# balance_dataset: 層別化モード
# ============================


class TestBalanceDatasetStratified:
    def test_stratify_balances_each_side_independently(self):
        """
        1P に紫 100、2P に紫 10 の偏ったデータで、stratify=True なら
        各 side が独立にクラス均等化される (紫は各 side 内で赤/青と同程度)。
        """
        ds = _synth_dataset({
            SIDE_1P: {
                COLOR_EMPTY: 50,
                COLOR_RED: 30, COLOR_BLUE: 30, COLOR_PURPLE: 100,
            },
            SIDE_2P: {
                COLOR_EMPTY: 50,
                COLOR_RED: 30, COLOR_BLUE: 30, COLOR_PURPLE: 10,
            },
        })
        balanced = balance_dataset(
            ds, empty_ratio_cap=0.35, color_balance_factor=2.0,
            seed=0, stratify_by_side=True,
        )
        # 各 side 内で紫の件数が min*2.0 以内に収まっている
        for side_val in (SIDE_1P, SIDE_2P):
            mask = balanced.sides == side_val
            sub_labels = balanced.labels[mask]
            if len(sub_labels) == 0:
                continue
            non_empty = sub_labels[sub_labels != COLOR_EMPTY]
            if len(non_empty) == 0:
                continue
            unique, counts = np.unique(non_empty, return_counts=True)
            if len(counts) >= 2:
                assert counts.max() <= counts.min() * 2.0 + 1

    def test_stratify_preserves_sides_metadata(self):
        """stratify=True でも sides メタが正しく引き継がれる。"""
        ds = _synth_dataset({
            SIDE_1P: {COLOR_EMPTY: 20, COLOR_RED: 10, COLOR_BLUE: 10},
            SIDE_2P: {COLOR_EMPTY: 20, COLOR_RED: 10, COLOR_BLUE: 10},
        })
        balanced = balance_dataset(ds, seed=0, stratify_by_side=True)
        assert balanced.sides is not None
        assert len(balanced.sides) == len(balanced.labels)
        # 1P, 2P 両方が存在
        unique_sides = set(balanced.sides.tolist())
        assert SIDE_1P in unique_sides
        assert SIDE_2P in unique_sides

    def test_stratify_handles_unknown_side_only(self):
        """
        side が全 unknown な旧 npz 相当データで stratify=True を指定しても
        クラッシュせず、単一バケットとして処理される。
        """
        patches = np.zeros((60, 4, 4, 3), dtype=np.uint8)
        labels = np.concatenate([
            np.full(30, COLOR_EMPTY, dtype=np.int64),
            np.full(15, COLOR_RED, dtype=np.int64),
            np.full(15, COLOR_BLUE, dtype=np.int64),
        ])
        ds = PatchDataset(patches=patches, labels=labels)
        balanced = balance_dataset(ds, seed=0, stratify_by_side=True)
        assert len(balanced.labels) > 0
        assert np.all(balanced.sides == SIDE_UNKNOWN)

    def test_stratify_mixed_known_and_unknown_sides(self):
        """旧 npz (unknown) と新 npz (1P/2P) 混在でも壊れない。"""
        patches = np.zeros((90, 4, 4, 3), dtype=np.uint8)
        labels = np.concatenate([
            # unknown バケット
            np.full(10, COLOR_EMPTY, dtype=np.int64),
            np.full(10, COLOR_RED, dtype=np.int64),
            np.full(10, COLOR_BLUE, dtype=np.int64),
            # 1P
            np.full(10, COLOR_EMPTY, dtype=np.int64),
            np.full(10, COLOR_RED, dtype=np.int64),
            np.full(10, COLOR_BLUE, dtype=np.int64),
            # 2P
            np.full(10, COLOR_EMPTY, dtype=np.int64),
            np.full(10, COLOR_RED, dtype=np.int64),
            np.full(10, COLOR_BLUE, dtype=np.int64),
        ])
        sides = np.array(
            [SIDE_UNKNOWN] * 30 + [SIDE_1P] * 30 + [SIDE_2P] * 30,
            dtype=np.int8,
        )
        ds = PatchDataset(patches=patches, labels=labels, sides=sides)
        balanced = balance_dataset(ds, seed=0, stratify_by_side=True)
        # 3 グループ全てが結果に含まれている
        got_sides = set(balanced.sides.tolist())
        assert got_sides == {SIDE_UNKNOWN, SIDE_1P, SIDE_2P}
