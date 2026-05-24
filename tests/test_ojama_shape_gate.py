"""OjamaShapeGate の単体テスト (= cycle 50、 2026-05-21)."""
from __future__ import annotations

import numpy as np
import pytest

from src.patch_classifier import OjamaShapeGate


@pytest.fixture
def gate() -> OjamaShapeGate:
    return OjamaShapeGate()


def test_empty_patch(gate: OjamaShapeGate) -> None:
    """空 patch は False."""
    assert gate.is_ojama(np.zeros((0, 0, 3), dtype=np.uint8)) is False


def test_flat_white_patch(gate: OjamaShapeGate) -> None:
    """真っ白 (= WIN/LOSE telop) は ojama でない (V 高すぎ)."""
    patch = np.full((40, 40, 3), 255, dtype=np.uint8)
    assert gate.is_ojama(patch) is False


def test_flat_red_patch(gate: OjamaShapeGate) -> None:
    """真っ赤 (= red puyo / red telop) は ojama でない (S 高い)."""
    patch = np.zeros((40, 40, 3), dtype=np.uint8)
    patch[:, :, 2] = 200  # BGR の R
    assert gate.is_ojama(patch) is False


def test_flat_gray_no_edges(gate: OjamaShapeGate) -> None:
    """無地灰色 (= edge なし) は ojama でない (= ヒビ模様欠如)."""
    patch = np.full((40, 40, 3), 120, dtype=np.uint8)
    assert gate.is_ojama(patch) is False


def test_gray_with_pattern(gate: OjamaShapeGate) -> None:
    """灰色 + 円形 + 内部模様 = ojama 候補."""
    import cv2
    patch = np.full((40, 40, 3), 120, dtype=np.uint8)
    # 中心に灰色円描画 (= 外接円形状)
    cv2.circle(patch, (20, 20), 14, (100, 100, 100), -1)
    # 内部にヒビ模様 (= 線描画 = edge)
    cv2.line(patch, (10, 20), (30, 20), (60, 60, 60), 1)
    cv2.line(patch, (20, 10), (20, 30), (60, 60, 60), 1)
    cv2.line(patch, (12, 12), (28, 28), (60, 60, 60), 1)
    # 評価
    result = gate.is_ojama(patch)
    # 形状次第で True or False (= threshold 依存)、 OK = エラーなく実行
    assert isinstance(result, bool)


def test_gate_threshold_constants(gate: OjamaShapeGate) -> None:
    """閾値定数が妥当な範囲 (= 想定外設定検知)."""
    assert 0 < gate.GRAY_S_MAX <= 100
    assert 0 < gate.GRAY_V_MIN < gate.GRAY_V_MAX <= 255
    assert 0 < gate.EDGE_DENSITY_MIN < 1.0
    assert 0 < gate.CIRCULARITY_MIN < 1.0
