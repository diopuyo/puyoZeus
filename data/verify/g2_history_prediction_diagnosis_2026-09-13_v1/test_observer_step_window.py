"""原凍結stepの境界採録を再用する。readerは人工、画像の検収ではない。"""
import importlib.util
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'g2_hidden_probability_capture_2026-09-08_v1'
sys.path.insert(0, str(PRIOR))
SPEC = importlib.util.spec_from_file_location('_window_pb_fixture', PRIOR / 'test_runtime_capture.py')
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
frozen, prepared, guarded_hash_cache = T.frozen, T.prepared, T.guarded_hash_cache
FIRST, LAST = 29052, 36900


@pytest.mark.parametrize('frame', [36298, 36300, LAST])
def test_original_step_extended_capture(frozen: Any, prepared: Any, monkeypatch: Any,
                                        tmp_path: Path, frame: int) -> None:
    monkeypatch.setattr(T.T.W, 'WINDOWS', ((FIRST, LAST),))
    monkeypatch.setattr(T.H, 'WINDOWS', ((FIRST, LAST),))
    output = tmp_path / 'receipts'
    output.mkdir()
    monkeypatch.setenv('MANUFACTURING_OUTPUT', str(output))
    T.test_actual_step_preserves_full_distribution_and_old_traces(
        frozen, prepared, monkeypatch, tmp_path, frame)
    assert not T.T.W.selected(LAST + 2, '2P')
    assert not T.T.W.selected(LAST - 1, '2P')
    assert T.H.expected()[-1] == (LAST, '2P')
