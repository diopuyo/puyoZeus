"""src/cutmix.py のテスト。"""
from __future__ import annotations

import numpy as np

from src.board import COLOR_BLUE, COLOR_PURPLE, COLOR_RED, COLOR_YELLOW
from src.cutmix import cutmix_pair, generate_cutmix_arrays, generate_cutmix_samples
from src.patch_classifier import PatchSample


def _solid_patch(color_value: int, size: int = 44) -> np.ndarray:
    """指定値で塗りつぶした 44×44 BGR パッチ。"""
    return np.full((size, size, 3), color_value, dtype=np.uint8)


def test_cutmix_pair_returns_same_shape() -> None:
    a = _solid_patch(255)
    b = _solid_patch(0)
    mixed, _ = cutmix_pair(a, COLOR_RED, b, COLOR_BLUE)
    assert mixed.shape == a.shape


def test_cutmix_pair_mixes_pixels() -> None:
    """合成パッチには元 A と B の値が混在する。"""
    a = _solid_patch(255)  # 全白
    b = _solid_patch(0)    # 全黒
    mixed, _ = cutmix_pair(a, COLOR_RED, b, COLOR_BLUE)
    unique = np.unique(mixed)
    # 0 と 255 が両方入っているはず
    assert 0 in unique
    assert 255 in unique


def test_cutmix_pair_label_swap_threshold() -> None:
    """矩形が大きいと label が B に切替わる、小さいと A のまま。"""
    a = _solid_patch(255)
    b = _solid_patch(0)
    # ratio_range=(0.9, 0.95) なら必ず大矩形 → B label
    _, lbl = cutmix_pair(a, COLOR_RED, b, COLOR_BLUE, ratio_range=(0.9, 0.95))
    assert lbl == COLOR_BLUE
    # ratio_range=(0.05, 0.1) なら小矩形 → A label
    _, lbl = cutmix_pair(a, COLOR_RED, b, COLOR_BLUE, ratio_range=(0.05, 0.1))
    assert lbl == COLOR_RED


def test_cutmix_pair_shape_mismatch_raises() -> None:
    a = np.zeros((44, 44, 3), dtype=np.uint8)
    b = np.zeros((20, 20, 3), dtype=np.uint8)
    try:
        cutmix_pair(a, COLOR_RED, b, COLOR_BLUE)
    except ValueError as e:
        assert "shape" in str(e)
    else:
        assert False, "ValueError 期待"


def test_generate_cutmix_samples_count() -> None:
    samples = [
        PatchSample(patch=_solid_patch(200), color=COLOR_RED),
        PatchSample(patch=_solid_patch(100), color=COLOR_RED),
        PatchSample(patch=_solid_patch(50), color=COLOR_BLUE),
        PatchSample(patch=_solid_patch(150), color=COLOR_PURPLE),
    ]
    out = generate_cutmix_samples(samples, n_extra=10, seed=0)
    assert len(out) == 10
    assert all(isinstance(s, PatchSample) for s in out)
    assert all(s.patch.shape == (44, 44, 3) for s in out)


def test_generate_cutmix_samples_focus_pairs_only() -> None:
    """focus_pairs 指定時、生成サンプルのラベルは指定対のいずれかに限定。"""
    samples = [
        PatchSample(patch=_solid_patch(200), color=COLOR_RED),
        PatchSample(patch=_solid_patch(50), color=COLOR_BLUE),
        PatchSample(patch=_solid_patch(150), color=COLOR_PURPLE),
        PatchSample(patch=_solid_patch(180), color=COLOR_YELLOW),
    ]
    focus = {(COLOR_RED, COLOR_BLUE)}
    out = generate_cutmix_samples(samples, n_extra=20, focus_pairs=focus, seed=42)
    assert len(out) == 20
    labels = {s.color for s in out}
    # 合成ラベルは {R, B} のどちらか
    assert labels.issubset({COLOR_RED, COLOR_BLUE})


def test_generate_cutmix_samples_empty_input() -> None:
    assert generate_cutmix_samples([], n_extra=5) == []


def test_generate_cutmix_samples_single_color_no_pairs() -> None:
    samples = [
        PatchSample(patch=_solid_patch(200), color=COLOR_RED),
        PatchSample(patch=_solid_patch(180), color=COLOR_RED),
    ]
    out = generate_cutmix_samples(samples, n_extra=5)
    assert out == []


# ============================
# numpy 配列ベース API
# ============================


def test_generate_cutmix_arrays_shape() -> None:
    patches = np.stack([
        _solid_patch(200),
        _solid_patch(50),
        _solid_patch(150),
    ])
    labels = np.array([COLOR_RED, COLOR_BLUE, COLOR_PURPLE], dtype=np.int8)
    p_out, l_out = generate_cutmix_arrays(patches, labels, n_extra=10, seed=0)
    assert p_out.shape == (10, 44, 44, 3)
    assert l_out.shape == (10,)
    assert p_out.dtype == np.uint8


def test_generate_cutmix_arrays_focus_pairs_filter() -> None:
    patches = np.stack([
        _solid_patch(200),
        _solid_patch(50),
        _solid_patch(150),
        _solid_patch(120),
    ])
    labels = np.array([COLOR_RED, COLOR_BLUE, COLOR_PURPLE, COLOR_YELLOW], dtype=np.int8)
    focus = {(COLOR_RED, COLOR_BLUE)}
    _, l_out = generate_cutmix_arrays(patches, labels, n_extra=20, focus_pairs=focus, seed=42)
    unique = set(int(x) for x in l_out)
    assert unique.issubset({COLOR_RED, COLOR_BLUE})


def test_generate_cutmix_arrays_empty_input() -> None:
    patches = np.zeros((0, 44, 44, 3), dtype=np.uint8)
    labels = np.zeros((0,), dtype=np.int8)
    p_out, l_out = generate_cutmix_arrays(patches, labels, n_extra=5)
    assert len(p_out) == 0
    assert len(l_out) == 0


def test_generate_cutmix_arrays_single_color() -> None:
    patches = np.stack([_solid_patch(200), _solid_patch(180)])
    labels = np.array([COLOR_RED, COLOR_RED], dtype=np.int8)
    p_out, l_out = generate_cutmix_arrays(patches, labels, n_extra=5)
    assert len(l_out) == 0
