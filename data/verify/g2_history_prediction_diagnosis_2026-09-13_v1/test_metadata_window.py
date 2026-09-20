"""原採録器の旧終端越え・全保存・非干渉を人工盤面で検査する。"""
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'g2_collector_metadata_bounded_2026-09-09_v1'
sys.path.insert(0, str(PRIOR))
SPEC = importlib.util.spec_from_file_location('_window_metadata_fixture', PRIOR / 'test_bounded.py')
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
collector = T.collector
FIRST, LAST, STRIDE = 36298, 36900, 2


def test_original_metadata_boundary_save(collector: Any, tmp_path: Path, monkeypatch: Any) -> None:
    # 初期履歴は人工。実構成のFIRST=29052は別の実configured検査で保証する。
    monkeypatch.setattr(T.O, 'FIRST', FIRST)
    monkeypatch.setattr(T.O, 'LAST', LAST)
    old, _ = T.run(collector, tmp_path / 'off', False)
    new, state = T.run(collector, tmp_path / 'on', True)
    assert old == new
    assert state[T.B.KEY].rows.count == ((LAST - FIRST) // STRIDE + 1) * 2
    T.B.finish(state)
    T.B.verify(tmp_path / 'on')
