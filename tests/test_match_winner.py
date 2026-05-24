"""src/match_winner.py のテスト。"""
from __future__ import annotations

import numpy as np

from src.match_winner import (
    DIGIT_DIFF_HAMMING,
    DIGIT_SAME_HAMMING,
    SIGNATURE_SIZE,
    compare_digit_pairs,
    digit_signature,
    extract_digit_patches,
    hamming_distance,
)


def _solid_patch(value: int, size: int = 45) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _patch_with_pattern(seed: int, size: int = 45) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_digit_signature_shape() -> None:
    sig = digit_signature(_solid_patch(120))
    assert sig.shape == (SIGNATURE_SIZE * SIGNATURE_SIZE,)
    assert sig.dtype == np.uint8


def test_digit_signature_empty() -> None:
    sig = digit_signature(np.zeros((0, 0, 3), dtype=np.uint8))
    assert sig.shape == (SIGNATURE_SIZE * SIGNATURE_SIZE,)


def test_hamming_distance_identical() -> None:
    p = _patch_with_pattern(42)
    sig = digit_signature(p)
    assert hamming_distance(sig, sig) == 0


def test_hamming_distance_different() -> None:
    p1 = _patch_with_pattern(1)
    p2 = _patch_with_pattern(99)
    d = hamming_distance(digit_signature(p1), digit_signature(p2))
    assert d > DIGIT_DIFF_HAMMING


def test_compare_digit_pairs_left_won() -> None:
    """左 (1P) だけ変わった場合 → 1P 勝利。"""
    left_a = _patch_with_pattern(1)
    right_a = _patch_with_pattern(2)
    left_b = _patch_with_pattern(99)        # 左変化大
    right_b = right_a.copy()                # 右変化なし
    result = compare_digit_pairs(left_a, right_a, left_b, right_b)
    assert result.winner == "1P"
    assert result.left_changed is True
    assert result.right_changed is False


def test_compare_digit_pairs_right_won() -> None:
    """右 (2P) だけ変わった場合 → 2P 勝利。"""
    left_a = _patch_with_pattern(1)
    right_a = _patch_with_pattern(2)
    left_b = left_a.copy()
    right_b = _patch_with_pattern(99)
    result = compare_digit_pairs(left_a, right_a, left_b, right_b)
    assert result.winner == "2P"
    assert result.right_changed is True
    assert result.left_changed is False


def test_compare_digit_pairs_both_unchanged() -> None:
    """両方変化なし → 判定不能。"""
    left_a = _patch_with_pattern(1)
    right_a = _patch_with_pattern(2)
    result = compare_digit_pairs(left_a, right_a, left_a.copy(), right_a.copy())
    assert result.winner is None
    assert result.left_hamming == 0
    assert result.right_hamming == 0


def test_compare_digit_pairs_both_changed() -> None:
    """両方変化 → 判定不能（同時変化はあり得ないので）。"""
    left_a = _patch_with_pattern(1)
    right_a = _patch_with_pattern(2)
    left_b = _patch_with_pattern(99)
    right_b = _patch_with_pattern(77)
    result = compare_digit_pairs(left_a, right_a, left_b, right_b)
    assert result.winner is None


def test_compare_digit_pairs_none_inputs() -> None:
    result = compare_digit_pairs(None, None, None, None)
    assert result.winner is None


def test_extract_digit_patches_correct_size() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    left, right = extract_digit_patches(frame)
    assert left is not None and right is not None
    # NUMBER_Y=(965, 1010), heights=45; NUMBER_LEFT_X width=60, NUMBER_RIGHT_X width=60
    assert left.shape == (45, 60, 3)
    assert right.shape == (45, 60, 3)


def test_extract_digit_patches_wrong_resolution() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    left, right = extract_digit_patches(frame)
    assert left is None and right is None
