"""`_median_fast` と HSV median 最適化が旧実装と完全同値であることを検証する。

2026-07-30 の高速化 (np.median → np.partition、astype 遅延、boolean index 遅延) は
**bit-identical を意図した変更**なので、フラグで守らず直接置き換えている。
そのため旧実装をこのテスト内に保存し、両者の返り値が完全一致することを
乱数コーパスとエッジケースで恒久的に検証する (回帰防止)。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.image_reader import (
    RED_HUE_WRAP_CORRECTED_MAX,
    RED_HUE_WRAP_THRESHOLD,
    SPECULAR_FALLBACK_MIN_RATIO,
    SPECULAR_S_MAX,
    SPECULAR_V_MIN,
    ColorClassifier,
    _median_fast,
)

# 実運用のセルパッチサイズ帯 (ぷよ 1 セル ~60x60、サブ region はより小さい)
PATCH_SIZES: tuple[int, ...] = (1, 2, 3, 4, 15, 16, 17, 24, 32, 48, 60)
# 乱数コーパスの試行数
N_TRIALS: int = 60


# ---------------------------------------------------------------------------
# 旧実装 (2026-07-30 の高速化前。比較の基準として保存)
# ---------------------------------------------------------------------------
def _old_stable_h_median(h_channel: np.ndarray, enable_fix: bool) -> int:
    """旧 `_compute_stable_h_median` (高速化前の実装をそのまま保存)。"""
    h_flat = h_channel.ravel().astype(np.int16)
    if not enable_fix:
        return int(np.median(h_flat))
    RED_HUE_LOW_MAX: int = 30
    n_total = max(1, len(h_flat))
    low_ratio = float(np.sum(h_flat <= RED_HUE_LOW_MAX)) / n_total
    high_ratio = float(np.sum(h_flat >= RED_HUE_WRAP_THRESHOLD)) / n_total
    if low_ratio >= 0.15 and high_ratio >= 0.15:
        h_wrapped = np.where(
            h_flat >= RED_HUE_WRAP_THRESHOLD, h_flat - 180, h_flat,
        )
        med_wrapped = float(np.median(h_wrapped))
        if med_wrapped <= RED_HUE_WRAP_CORRECTED_MAX:
            return int(max(0, med_wrapped))
    return int(np.median(h_flat))


def _old_specular_robust_s(
    s_channel: np.ndarray, v_channel: np.ndarray, enable_fix: bool,
) -> int:
    """旧 `_compute_specular_robust_s` (高速化前の実装をそのまま保存)。"""
    s_flat = s_channel.ravel().astype(np.int32)
    if not enable_fix:
        return int(np.median(s_flat))
    v_flat = v_channel.ravel().astype(np.int32)
    n_total = max(1, len(s_flat))
    specular_mask = (v_flat >= SPECULAR_V_MIN) & (s_flat <= SPECULAR_S_MAX)
    valid_s = s_flat[~specular_mask]
    if len(valid_s) < int(n_total * SPECULAR_FALLBACK_MIN_RATIO):
        return int(np.median(s_flat))
    return int(np.median(valid_s))


def _make_classifier(enable_red: bool, enable_spec: bool) -> ColorClassifier:
    """フラグを指定した ColorClassifier を作る。"""
    return ColorClassifier(
        enable_red_hue_wrap_fix=enable_red,
        enable_specular_robust_saturation=enable_spec,
    )


# ---------------------------------------------------------------------------
# _median_fast 単体
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 100, 255, 256, 999, 1000])
def test_median_fast_matches_np_median(size: int) -> None:
    """任意長で np.median と完全一致する (偶数長の中央 2 値平均も含む)。"""
    rng = np.random.default_rng(size)
    for _ in range(20):
        a = rng.integers(0, 256, size=size, dtype=np.uint8)
        assert _median_fast(a) == float(np.median(a))


def test_median_fast_negative_and_int16() -> None:
    """折り返し補正後の負値を含む int16 でも一致する。"""
    rng = np.random.default_rng(7)
    for _ in range(50):
        a = rng.integers(-180, 180, size=rng.integers(1, 200), dtype=np.int16)
        assert _median_fast(a) == float(np.median(a))


def test_median_fast_empty_is_nan() -> None:
    """空配列は np.median と同じく nan を返す。"""
    assert np.isnan(_median_fast(np.array([], dtype=np.uint8)))


# ---------------------------------------------------------------------------
# H median (赤色相折り返し補正)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable", [True, False])
@pytest.mark.parametrize("size", PATCH_SIZES)
def test_h_median_equivalence_random(size: int, enable: bool) -> None:
    """H median が旧実装と完全一致する (H は 0-179 の全域乱数)。"""
    clf = _make_classifier(enable_red=enable, enable_spec=False)
    rng = np.random.default_rng(size * 31 + int(enable))
    for _ in range(N_TRIALS):
        # hsv[:, :, 0] と同じ非連続 view を再現する
        hsv = rng.integers(0, 180, size=(size, size, 3), dtype=np.uint8)
        h_view = hsv[:, :, 0]
        assert clf._compute_stable_h_median(h_view) == _old_stable_h_median(
            h_view, enable,
        )


@pytest.mark.parametrize("low_frac", [0.0, 0.10, 0.15, 0.16, 0.5, 0.85, 1.0])
@pytest.mark.parametrize("size", [16, 24, 32])
def test_h_median_equivalence_bimodal_red(size: int, low_frac: float) -> None:
    """赤 2 峰 (LOW/HIGH 共存) の境界比率で一致する = 早期打ち切りの正当性。

    LOW 比率 0.15 前後は「補正を適用するか」の分岐点であり、
    高速化で入れた早期打ち切りがここを取り違えていないことを確認する。
    """
    clf = _make_classifier(enable_red=True, enable_spec=False)
    rng = np.random.default_rng(int(low_frac * 1000) + size)
    n = size * size
    n_low = int(n * low_frac)
    for _ in range(20):
        low = rng.integers(0, 15, size=n_low, dtype=np.uint8)
        high = rng.integers(RED_HUE_WRAP_THRESHOLD, 180, size=n - n_low, dtype=np.uint8)
        flat = np.concatenate([low, high])
        rng.shuffle(flat)
        h_view = flat.reshape(size, size)
        assert clf._compute_stable_h_median(h_view) == _old_stable_h_median(
            h_view, True,
        )


def test_h_median_equivalence_purple_single_peak() -> None:
    """紫 (H=130-165、HIGH 側のみ) で補正が発動しないことが一致する。"""
    clf = _make_classifier(enable_red=True, enable_spec=False)
    rng = np.random.default_rng(99)
    for _ in range(30):
        h_view = rng.integers(130, 166, size=(24, 24), dtype=np.uint8)
        assert clf._compute_stable_h_median(h_view) == _old_stable_h_median(
            h_view, True,
        )


# ---------------------------------------------------------------------------
# S median (光沢ハイライト除外)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enable", [True, False])
@pytest.mark.parametrize("size", PATCH_SIZES)
def test_s_median_equivalence_random(size: int, enable: bool) -> None:
    """S median が旧実装と完全一致する。"""
    clf = _make_classifier(enable_red=False, enable_spec=enable)
    rng = np.random.default_rng(size * 17 + int(enable))
    for _ in range(N_TRIALS):
        hsv = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        s_view, v_view = hsv[:, :, 1], hsv[:, :, 2]
        try:
            expected = _old_specular_robust_s(s_view, v_view, enable)
        except ValueError:
            # 旧実装は「5 画素未満のパッチが全面ハイライト」で ValueError を投げる
            # 潜在バグを持っていた (test_all_specular_tiny_patch_no_crash 参照)。
            # 新実装はここをガードしているので、一致比較の対象外とする。
            continue
        assert clf._compute_specular_robust_s(s_view, v_view) == expected


@pytest.mark.parametrize("spec_frac", [0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0])
def test_s_median_equivalence_specular_fraction(spec_frac: float) -> None:
    """ハイライト画素比率を振って一致する = fallback 境界と除外なし分岐の検証。

    spec_frac=0.0 は「除外画素なしなら複製を作らず全画素 median」の分岐、
    高比率側は SPECULAR_FALLBACK_MIN_RATIO による fallback の分岐を突く。
    """
    clf = _make_classifier(enable_red=False, enable_spec=True)
    rng = np.random.default_rng(int(spec_frac * 100))
    size = 24
    n = size * size
    n_spec = int(n * spec_frac)
    for _ in range(20):
        # ハイライト画素: V 高 & S 低 / 通常画素: V 低 & S 高
        s_spec = rng.integers(0, SPECULAR_S_MAX + 1, size=n_spec, dtype=np.uint8)
        v_spec = rng.integers(SPECULAR_V_MIN, 256, size=n_spec, dtype=np.uint8)
        s_norm = rng.integers(
            SPECULAR_S_MAX + 1, 256, size=n - n_spec, dtype=np.uint8,
        )
        v_norm = rng.integers(0, SPECULAR_V_MIN, size=n - n_spec, dtype=np.uint8)
        s_flat = np.concatenate([s_spec, s_norm])
        v_flat = np.concatenate([v_spec, v_norm])
        perm = rng.permutation(n)
        s_view = s_flat[perm].reshape(size, size)
        v_view = v_flat[perm].reshape(size, size)
        assert clf._compute_specular_robust_s(s_view, v_view) == (
            _old_specular_robust_s(s_view, v_view, True)
        )


def test_all_specular_tiny_patch_no_crash() -> None:
    """5 画素未満の全面ハイライトパッチで落ちない (旧実装の潜在クラッシュ修正)。

    旧実装は `int(n_total * SPECULAR_FALLBACK_MIN_RATIO) == 0` になる極小パッチで
    fallback 判定 (0 < 0) をすり抜け、空配列の median = nan を int() して
    ValueError を投げていた。新実装は全画素 median にガードする。
    """
    clf = _make_classifier(enable_red=False, enable_spec=True)
    for n_px in (1, 2, 3, 4):
        # 全画素が「V 高 & S 低」= ハイライト条件を満たす
        s_view = np.full((1, n_px), SPECULAR_S_MAX, dtype=np.uint8)
        v_view = np.full((1, n_px), 255, dtype=np.uint8)
        # 旧実装は落ちることを明示 (回帰時に気付けるように)
        with pytest.raises(ValueError):
            _old_specular_robust_s(s_view, v_view, True)
        # 新実装は全画素 median を返す
        assert clf._compute_specular_robust_s(s_view, v_view) == SPECULAR_S_MAX


# ---------------------------------------------------------------------------
# classify() 全体 (実パッチ相当の BGR で end-to-end 一致)
# ---------------------------------------------------------------------------
def test_classify_unchanged_on_random_bgr() -> None:
    """classify() の返り値が本番既定フラグで安定している (回帰検出用)。

    旧実装との一致は上の 2 関数で担保済み。ここは classify 全体が
    例外なく走ることと、同一入力で決定的であることを確認する。
    """
    clf = _make_classifier(enable_red=True, enable_spec=True)
    rng = np.random.default_rng(2026)
    for _ in range(40):
        patch = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        first = clf.classify(patch)
        assert first == clf.classify(patch)
