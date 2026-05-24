"""src/cell_augment.py のテスト.

4 色 permutation augmentation のヘルパー関数を検証する。
- identity / swap / rotation 等の color_map で patch + label が正しく変わる
- HSV hue shift が想定範囲内の hue へ移動する
- 24 通りの permutation すべてが label-consistent
- EMPTY/OJAMA/UNKNOWN は固定 (不変)
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
)
from src.cell_augment import (
    DEFAULT_PERMUTABLE_COLORS,
    HUE_PERIOD,
    NON_PERMUTABLE_COLORS,
    REPRESENTATIVE_HUE,
    apply_color_permutation_to_label,
    generate_color_permutations,
    permute_colors_in_patch,
    random_color_permutation,
)


# ============================
# 合成 patch ヘルパー
# ============================


def _make_solid_hsv_patch(h: int, s: int = 200, v: int = 200,
                          size: int = 16) -> np.ndarray:
    """指定 HSV (uint8 0-179/0-255/0-255) の単色 BGR patch を生成."""
    hsv = np.full((size, size, 3), [h, s, v], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _median_hue(bgr_patch: np.ndarray) -> int:
    """patch の中央値 hue (uint8 0-179)."""
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    return int(np.median(hsv[:, :, 0]))


def _hue_dist(a: int, b: int) -> int:
    """uint8 hue 環状距離 (0-90)."""
    d = abs(int(a) - int(b)) % HUE_PERIOD
    return min(d, HUE_PERIOD - d)


# ============================
# permute_colors_in_patch
# ============================


def test_identity_permutation_preserves_patch():
    """identity color_map では patch が変わらない."""
    patch = _make_solid_hsv_patch(h=REPRESENTATIVE_HUE[COLOR_BLUE])
    cmap = {c: c for c in DEFAULT_PERMUTABLE_COLORS}
    out = permute_colors_in_patch(patch, COLOR_BLUE, cmap)
    assert out.shape == patch.shape
    np.testing.assert_array_equal(out, patch)


def test_swap_red_blue_shifts_hue():
    """RED → BLUE swap で patch の hue が BLUE 範囲へ shift."""
    patch = _make_solid_hsv_patch(h=REPRESENTATIVE_HUE[COLOR_RED])
    cmap = {COLOR_RED: COLOR_BLUE, COLOR_BLUE: COLOR_RED,
            COLOR_GREEN: COLOR_GREEN, COLOR_YELLOW: COLOR_YELLOW}
    out = permute_colors_in_patch(patch, COLOR_RED, cmap)
    new_h = _median_hue(out)
    target_h = REPRESENTATIVE_HUE[COLOR_BLUE]
    # uint8 量子化誤差込み ±2 以内
    assert _hue_dist(new_h, target_h) <= 2, (
        f"new_h={new_h} expected≈{target_h}"
    )


def test_swap_yellow_green_shifts_hue():
    """YELLOW → GREEN swap で hue が GREEN 範囲へ shift."""
    patch = _make_solid_hsv_patch(h=REPRESENTATIVE_HUE[COLOR_YELLOW])
    cmap = {COLOR_YELLOW: COLOR_GREEN, COLOR_GREEN: COLOR_YELLOW,
            COLOR_RED: COLOR_RED, COLOR_BLUE: COLOR_BLUE}
    out = permute_colors_in_patch(patch, COLOR_YELLOW, cmap)
    new_h = _median_hue(out)
    target_h = REPRESENTATIVE_HUE[COLOR_GREEN]
    assert _hue_dist(new_h, target_h) <= 2


def test_red_wraparound_handled():
    """RED の hue=0 から PURPLE の hue=147 への shift で wraparound 処理."""
    patch = _make_solid_hsv_patch(h=REPRESENTATIVE_HUE[COLOR_RED])
    cmap = {COLOR_RED: COLOR_PURPLE, COLOR_PURPLE: COLOR_RED,
            COLOR_BLUE: COLOR_BLUE, COLOR_GREEN: COLOR_GREEN,
            COLOR_YELLOW: COLOR_YELLOW}
    out = permute_colors_in_patch(patch, COLOR_RED, cmap)
    new_h = _median_hue(out)
    target_h = REPRESENTATIVE_HUE[COLOR_PURPLE]
    assert _hue_dist(new_h, target_h) <= 2


def test_empty_patch_unchanged():
    """src_color が EMPTY なら patch を変更しない."""
    patch = _make_solid_hsv_patch(h=10, s=10, v=10)
    cmap = {COLOR_RED: COLOR_BLUE, COLOR_BLUE: COLOR_RED,
            COLOR_GREEN: COLOR_GREEN, COLOR_YELLOW: COLOR_YELLOW}
    out = permute_colors_in_patch(patch, COLOR_EMPTY, cmap)
    np.testing.assert_array_equal(out, patch)


def test_ojama_patch_unchanged():
    """src_color が OJAMA なら patch を変更しない."""
    patch = _make_solid_hsv_patch(h=0, s=0, v=200)
    cmap = {COLOR_RED: COLOR_BLUE, COLOR_BLUE: COLOR_RED,
            COLOR_GREEN: COLOR_GREEN, COLOR_YELLOW: COLOR_YELLOW}
    out = permute_colors_in_patch(patch, COLOR_OJAMA, cmap)
    np.testing.assert_array_equal(out, patch)


def test_saturation_value_preserved():
    """hue だけ変わり、saturation/value が大幅変化しない."""
    s_in, v_in = 180, 220
    patch = _make_solid_hsv_patch(
        h=REPRESENTATIVE_HUE[COLOR_RED], s=s_in, v=v_in,
    )
    cmap = {COLOR_RED: COLOR_GREEN, COLOR_GREEN: COLOR_RED,
            COLOR_BLUE: COLOR_BLUE, COLOR_YELLOW: COLOR_YELLOW}
    out = permute_colors_in_patch(patch, COLOR_RED, cmap)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    s_out = int(np.median(hsv[:, :, 1]))
    v_out = int(np.median(hsv[:, :, 2]))
    # BGR ↔ HSV 往復の量子化誤差を許容
    assert abs(s_out - s_in) <= 5
    assert abs(v_out - v_in) <= 5


# ============================
# apply_color_permutation_to_label
# ============================


def test_label_swap():
    cmap = {COLOR_RED: COLOR_BLUE, COLOR_BLUE: COLOR_RED,
            COLOR_GREEN: COLOR_GREEN, COLOR_YELLOW: COLOR_YELLOW}
    assert apply_color_permutation_to_label(COLOR_RED, cmap) == COLOR_BLUE
    assert apply_color_permutation_to_label(COLOR_BLUE, cmap) == COLOR_RED
    assert apply_color_permutation_to_label(COLOR_GREEN, cmap) == COLOR_GREEN


@pytest.mark.parametrize("c", list(NON_PERMUTABLE_COLORS))
def test_label_non_permutable_colors_unchanged(c: int):
    """EMPTY/OJAMA/UNKNOWN は cmap に入っていても不変."""
    cmap = {COLOR_RED: COLOR_BLUE, COLOR_BLUE: COLOR_RED,
            COLOR_GREEN: COLOR_GREEN, COLOR_YELLOW: COLOR_YELLOW}
    assert apply_color_permutation_to_label(c, cmap) == c


# ============================
# generate_color_permutations
# ============================


def test_generate_24_permutations():
    """4 色 permutable なら 4! = 24 通り."""
    perms = generate_color_permutations()
    assert len(perms) == math.factorial(len(DEFAULT_PERMUTABLE_COLORS))


def test_generate_excludes_identity_when_requested():
    perms = generate_color_permutations(include_identity=False)
    assert len(perms) == math.factorial(
        len(DEFAULT_PERMUTABLE_COLORS),
    ) - 1
    for p in perms:
        assert any(s != d for s, d in p.items())


def test_all_permutations_are_label_consistent():
    """24 通りの cmap で apply_color_permutation_to_label が正しく動く.

    各 cmap の中で「元色集合 == 新色集合」であり、4 色全てが現れる。
    """
    perms = generate_color_permutations()
    base = set(DEFAULT_PERMUTABLE_COLORS)
    for cmap in perms:
        srcs = set(cmap.keys())
        dsts = set(cmap.values())
        assert srcs == base
        assert dsts == base
        # NON_PERMUTABLE_COLORS は不変
        for c in NON_PERMUTABLE_COLORS:
            assert apply_color_permutation_to_label(c, cmap) == c


def test_random_color_permutation_is_bijection():
    """random_color_permutation は bijection を返す."""
    rng = np.random.default_rng(seed=123)
    cmap = random_color_permutation(rng)
    base = set(DEFAULT_PERMUTABLE_COLORS)
    assert set(cmap.keys()) == base
    assert set(cmap.values()) == base


def test_full_pipeline_label_consistency():
    """全 24 通り × 4 元色について patch hue + label が同じ写像になる.

    permute_colors_in_patch(patch, src, cmap) の hue が
    REPRESENTATIVE_HUE[cmap[src]] に近いことを検証。
    """
    perms = generate_color_permutations()
    for cmap in perms:
        for src in DEFAULT_PERMUTABLE_COLORS:
            patch = _make_solid_hsv_patch(h=REPRESENTATIVE_HUE[src])
            out = permute_colors_in_patch(patch, src, cmap)
            new_h = _median_hue(out)
            target_h = REPRESENTATIVE_HUE[cmap[src]]
            assert _hue_dist(new_h, target_h) <= 2, (
                f"cmap={cmap} src={src} new_h={new_h} expected≈{target_h}"
            )
            new_label = apply_color_permutation_to_label(src, cmap)
            assert new_label == cmap[src]


def test_empty_input_patch():
    """空 patch でも crash しない."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    cmap = {COLOR_RED: COLOR_BLUE, COLOR_BLUE: COLOR_RED,
            COLOR_GREEN: COLOR_GREEN, COLOR_YELLOW: COLOR_YELLOW}
    out = permute_colors_in_patch(empty, COLOR_RED, cmap)
    assert out.shape == (0, 0, 3)
