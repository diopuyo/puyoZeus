"""元score helper→Admission→最外側出口を人工updateで検査する。CNN/GTではない。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any
import json

import pytest
from scripts import g3_current_exit as C
from tests.test_g3_admission import A, collector, pipeline, owned_replace, admission_root  # noqa: F401  admission_root は autouse fixture。採録先の根を一時領域へ差し替える (W45)


def result(frame: int, *, stable: bool = True, missing: bool = False,
           active: bool = True) -> Any:
    """凍結moduleの実結果型。隠し段は評価可視盤面に含めない。"""
    import numpy as np
    from src.recognition_pipeline import PipelineResult, SideResult, BoardState, Board
    board = Board.from_list(np.ones((13, 6), dtype=np.int8).tolist())
    values = [SideResult(side, BoardState.STABLE, board, None, board, None, 0, 0, None)
              for side in A.SIDES]
    if not stable:
        values[1] = replace(values[1], state=BoardState.MENU)
    if missing:
        values[1] = replace(values[1], confirmed_board=None)
    return PipelineResult(frame, frame / 60, active, *values)


@pytest.mark.parametrize('mode', ['unknown', 'nonstable', 'missing', 'inactive'])
def test_current_then_hold(collector: Any, monkeypatch: Any, tmp_path: Path, mode: str) -> None:
    """0点/片側OCR正常、次frame欠測、片側非STABLE等を分離する。"""
    c = collector
    pipe, ocr = pipeline(c, monkeypatch)
    state, before = dict(output=tmp_path), c.RecognitionPipeline.update
    with ExitStack() as stack:
        observer = A.install(stack, c, state, owned_replace(), source_id='artificial-current')
        output = C.install(stack, c, state, owned_replace())
        with pytest.raises(ValueError, match='duplicate'):
            C.install(stack, c, state, owned_replace())
        ocr.values['1P'] = 0
        first = result(0)
        assert pipe.update(0, 0.0, first) is first
        ocr.values['1P'] = None if mode == 'unknown' else 10
        second = result(2, stable=mode != 'nonstable', missing=mode == 'missing', active=mode != 'inactive')
        assert pipe.update(2, 2 / 60, second) is second
        assert output.rows == 2 and output.eligible == 1 and observer.decisions == 0
        with pytest.raises(ValueError, match='current_clock'):
            C.selection(observer, pipe, first, 0, 0.0)
    rows = [json.loads(line) for line in (tmp_path / 'G3_CURRENT_EXIT.jsonl').read_text().splitlines()]
    assert rows[0]['status'] == 'CURRENT_INPUT_ELIGIBLE'
    assert len(rows[0]['visible_confirmed_boards']['1P']) == 12
    assert rows[1]['status'] == 'HOLD' and rows[1]['visible_confirmed_boards'] is None
    assert all(not row['human_gt'] and not row['model_evaluation_completed'] for row in rows)
    assert first.p1.confirmed_board._grid.shape == (13, 6)
    assert c.RecognitionPipeline.update is before and output.stream.closed
    assert json.loads((tmp_path / 'G3_UI_ADMISSION_STATUS.json').read_text())['references_restored']


def test_original_update_error(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """内側更新の例外を出口で成功/HOLD行へ変換しない。"""
    pipe, _ = pipeline(collector, monkeypatch, failure=True)
    state = dict(output=tmp_path)
    with pytest.raises(LookupError, match='original_update_error'):
        with ExitStack() as stack:
            A.install(stack, collector, state, owned_replace(), source_id='artificial-error')
            output = C.install(stack, collector, state, owned_replace())
            pipe.update(0, 0.0, result(0))
    assert output.rows == 0 and output.stream.closed
    assert 'original_update_error' in output.error


def test_current_save_error(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """資格原票のwrite障害を保存し、成功行数を増やさない。"""
    pipe, ocr = pipeline(collector, monkeypatch)
    state = dict(output=tmp_path)
    with pytest.raises(OSError, match='current_write_failed'):
        with ExitStack() as stack:
            A.install(stack, collector, state, owned_replace(), source_id='artificial-write')
            output = C.install(stack, collector, state, owned_replace())
            ocr.values['1P'] = 0
            def broken(row: dict) -> bytes:
                raise OSError('current_write_failed')
            monkeypatch.setattr(C, 'encoded', broken)
            pipe.update(0, 0.0, result(0))
    assert output.rows == 0 and output.stream.closed and 'current_write_failed' in output.save_error
    saved = json.loads((tmp_path / 'G3_CURRENT_EXIT_STATUS.json').read_text())
    assert saved['references_restored'] and saved['closed']


def test_caught_save_error_cannot_resume(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """呼出側が保存例外を握っても同じrunの次frameを更新させない。"""
    pipe, ocr = pipeline(collector, monkeypatch)
    state = dict(output=tmp_path)
    with ExitStack() as stack:
        observer = A.install(stack, collector, state, owned_replace(), source_id='artificial-latch')
        output = C.install(stack, collector, state, owned_replace())
        ocr.values['1P'] = 0
        original = C.encoded
        def broken(row: dict) -> bytes:
            raise OSError('current_write_failed')
        monkeypatch.setattr(C, 'encoded', broken)
        with pytest.raises(OSError, match='current_write_failed'):
            pipe.update(0, 0.0, result(0))
        monkeypatch.setattr(C, 'encoded', original)
        with pytest.raises(ValueError, match='current_exit_failed'):
            pipe.update(2, 2 / 60, result(2))
        assert observer.frames == 1 and output.rows == 0
