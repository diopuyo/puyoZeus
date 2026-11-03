"""実行設定の切替で従来セッションを混用しないことを検証する。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src import cnn_onnx
from src.cnn_parallel_runtime import requested


def test_opt_in_session_is_separate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ort = pytest.importorskip('onnxruntime')
    created = []

    def create(path: str, options: Any, providers: list[str]) -> Any:
        created.append(options)
        return options

    monkeypatch.setattr(ort, 'InferenceSession', create)
    monkeypatch.setattr(cnn_onnx, '_SESSIONS', {})
    monkeypatch.delenv('PUYO_LIVE_ONNX_PARALLEL', raising=False)
    (tmp_path/'fixture.onnx').touch()
    assert not requested()
    old = cnn_onnx.session_for('fixture', tmp_path)
    assert old.intra_op_num_threads == old.inter_op_num_threads == 1
    monkeypatch.setenv('PUYO_LIVE_ONNX_PARALLEL', '1')
    new = cnn_onnx.session_for('fixture', tmp_path)
    assert new is not old
    assert new.intra_op_num_threads == 2 and new.inter_op_num_threads == 1
    assert new.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    for kind in ('intra', 'inter'):
        assert new.get_session_config_entry(f'session.{kind}_op.allow_spinning') == '0'
    assert cnn_onnx.session_for('fixture', tmp_path) is new
    monkeypatch.setenv('PUYO_LIVE_ONNX_PARALLEL', '0')
    assert cnn_onnx.session_for('fixture', tmp_path) is old
    assert len(created) == 2
