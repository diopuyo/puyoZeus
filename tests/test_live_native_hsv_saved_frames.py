"""指定された保存実コマで、配布 CNN を含む本体リーダーのビット一致を検証する。"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.verify_live_native_hsv import load_frames, reader_pair, verify_pair


def test_saved_frames_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """外部実コマ未指定の CI では省略し、検収時は実 Rust 呼出も必須とする。"""
    names = ('PUYO_HSV_TEST_MANIFEST', 'PUYO_HSV_TEST_ASSETS', 'PUYO_HSV_TEST_CONFIG')
    if not all(os.environ.get(name) for name in names):
        pytest.skip('保存実コマ・配布モデル・認識設定のパスが未指定')
    import cv2
    import torch
    import puyo_core
    assert callable(puyo_core.classify_cells_hsv)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    monkeypatch.setenv('PUYO_CNN_BACKEND', 'onnx')
    manifest, assets, config = (Path(os.environ[name]) for name in names)
    frames = load_frames(manifest)
    with reader_pair(assets, config) as readers:
        hsv = readers[1]._get_hsv_classifier()
        with patch.object(hsv, '_classify_native', wraps=hsv._classify_native) as native:
            result = verify_pair(readers, frames)
        assert native.call_count > 0, 'ON が実際の分類で使われていません'
    assert result['mismatches'] == 0
    assert result['frames'] == len(frames)
