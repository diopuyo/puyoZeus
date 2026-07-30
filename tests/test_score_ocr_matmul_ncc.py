"""スコアOCRの行列積NCC (enable_matmul_ncc) が従来経路と一致することを検証する。

2026-07-30 の高速化。同サイズ TM_CCOEFF_NORMED は Pearson 相関に等しいので、
正規化済みテンプレ行列との積 1 回で全テンプレ分のスコアが同時に得られる。
実測で 1セル分 1777us → 12.1us (146倍速)、1フレーム換算 28.43ms → 0.19ms。

**bit-identical ではない** (cv2 は内部 float32、こちらは float64)。
そのため既定 OFF で、このテストは「スコア差が十分小さく、ラベル決定が一致する」
ことを保証する。判定が変わる境界ケースがあれば実装ではなくここで検出したい。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.score_ocr import DIGIT_HEIGHT, DIGIT_WIDTH, ScoreOcr

# cv2(float32) と numpy(float64) の差の許容幅。実測は 5.5e-07 程度。
SCORE_ABS_TOL: float = 1e-05


def _make_templates(rng: np.random.Generator, n: int = 10) -> dict[int, np.ndarray]:
    """0-9 のテンプレ相当 (50x40 グレースケール) を作る。"""
    return {
        i: rng.integers(0, 256, size=(DIGIT_HEIGHT, DIGIT_WIDTH), dtype=np.uint8)
        for i in range(n)
    }


def _make_digit_like_templates() -> dict[int, np.ndarray]:
    """数字らしい構造を持つテンプレ (背景暗・筆跡明) を作る。

    乱数テンプレだとスコア分布が実運用と乖離するため、
    閾値 (_min_confidence / _margin_min) 付近を踏む可能性のある
    「互いに似たテンプレ」を意図的に作る。
    """
    templates: dict[int, np.ndarray] = {}
    for i in range(10):
        tpl = np.full((DIGIT_HEIGHT, DIGIT_WIDTH), 20, dtype=np.uint8)
        # 縦棒 + i に応じた横棒 = 互いに似た形になる
        tpl[10:40, 18:24] = 230
        tpl[10 + i * 3: 14 + i * 3, 8:32] = 230
        templates[i] = tpl
    return templates


def _pair(templates: dict[int, np.ndarray]) -> tuple[ScoreOcr, ScoreOcr]:
    """従来経路と行列積経路の 2 つの OCR を同一テンプレで作る。"""
    loop = ScoreOcr(templates=templates, enable_matmul_ncc=False)
    mm = ScoreOcr(templates=templates, enable_matmul_ncc=True)
    return loop, mm


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_scores_match_within_tolerance(seed: int) -> None:
    """NCC スコアが従来経路と許容差内で一致する。"""
    rng = np.random.default_rng(seed)
    loop, mm = _pair(_make_templates(rng))
    for _ in range(20):
        cell = rng.integers(
            0, 256, size=(DIGIT_HEIGHT, DIGIT_WIDTH), dtype=np.uint8,
        )
        s_loop = loop._ncc_scores_loop(cell)
        s_mm = mm._ncc_scores_matmul(cell)
        assert set(s_loop.keys()) == set(s_mm.keys())
        for label in s_loop:
            assert s_loop[label] == pytest.approx(s_mm[label], abs=SCORE_ABS_TOL)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_label_decision_matches_on_random_cells(seed: int) -> None:
    """_classify_digit のラベル決定が一致する (乱数セル)。"""
    rng = np.random.default_rng(seed)
    loop, mm = _pair(_make_templates(rng))
    for _ in range(40):
        cell = rng.integers(
            0, 256, size=(DIGIT_HEIGHT, DIGIT_WIDTH), dtype=np.uint8,
        )
        assert loop._classify_digit(cell) [0] == mm._classify_digit(cell)[0]


def test_label_decision_matches_on_digit_like_cells() -> None:
    """互いに似たテンプレ + ノイズ付き実物風セルでもラベル決定が一致する。

    閾値 (_min_confidence 0.55 / _margin_min 0.04) 付近を踏みやすい条件。
    """
    templates = _make_digit_like_templates()
    loop, mm = _pair(templates)
    rng = np.random.default_rng(2026)
    mismatches = 0
    for label, tpl in templates.items():
        for noise in (0, 5, 15, 30, 60):
            cell = tpl.astype(np.int16) + rng.integers(
                -noise, noise + 1, size=tpl.shape,
            )
            cell = np.clip(cell, 0, 255).astype(np.uint8)
            got_loop, _ = loop._classify_digit(cell)
            got_mm, _ = mm._classify_digit(cell)
            if got_loop != got_mm:
                mismatches += 1
    assert mismatches == 0, f"ラベル決定が {mismatches} 件不一致"


def test_uniform_cell_falls_back_to_loop() -> None:
    """分散ゼロのセルは相関が定義できないので従来経路に fallback する。"""
    rng = np.random.default_rng(7)
    loop, mm = _pair(_make_templates(rng))
    cell = np.full((DIGIT_HEIGHT, DIGIT_WIDTH), 128, dtype=np.uint8)
    # 例外を出さず、従来経路と同じ結果になること
    assert mm._ncc_scores_matmul(cell) == loop._ncc_scores_loop(cell)


def test_uniform_template_falls_back_to_loop() -> None:
    """分散ゼロのテンプレが混ざると行列化できないので従来経路に落ちる。"""
    rng = np.random.default_rng(8)
    templates = _make_templates(rng)
    templates[0] = np.full((DIGIT_HEIGHT, DIGIT_WIDTH), 100, dtype=np.uint8)
    loop, mm = _pair(templates)
    assert mm._prepare_template_matrix() is None
    cell = rng.integers(0, 256, size=(DIGIT_HEIGHT, DIGIT_WIDTH), dtype=np.uint8)
    assert mm._ncc_scores_matmul(cell) == loop._ncc_scores_loop(cell)


def test_no_templates_returns_none() -> None:
    """テンプレ不在時に落ちない (従来と同じ None)。"""
    mm = ScoreOcr(templates=None, enable_matmul_ncc=True)
    cell = np.zeros((DIGIT_HEIGHT, DIGIT_WIDTH), dtype=np.uint8)
    assert mm._classify_digit(cell) == (None, 0.0)


def test_template_matrix_is_cached() -> None:
    """正規化行列が 2 回目以降は再計算されない (同一オブジェクトが返る)。"""
    rng = np.random.default_rng(9)
    _, mm = _pair(_make_templates(rng))
    first = mm._prepare_template_matrix()
    second = mm._prepare_template_matrix()
    assert first is not None and first is second


def test_default_is_loop_path() -> None:
    """既定は従来経路 (bit-identical 維持) であること。"""
    rng = np.random.default_rng(10)
    ocr = ScoreOcr(templates=_make_templates(rng))
    assert ocr._enable_matmul_ncc is False
    assert ScoreOcr.load_default.__defaults__ is not None
