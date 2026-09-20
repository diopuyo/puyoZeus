"""局所色vetoの境界と原画像反例。実動画全品質とは分離する。"""
from __future__ import annotations
import contextlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import pytest
import adapter as A
import probe as P

sys.path.insert(0, str(P.SNAPSHOT))
from src.board import Board
from src.image_reader import DEFAULT_P2_REGION
from src.recognition_pipeline import RecognitionPipeline


def board(color: int = 4, row: int = 12, col: int = 0) -> Any:
    value = Board()
    value.set(row, col, color)
    return value


def measured(original: int = 4) -> tuple[Any, ...]:
    return (18, 74, 247), {c: 24.0 if c == original else 32.0 for c in A.COLOR_CODES}, 60, 60.0


@pytest.mark.parametrize('reason,saturation,distance,tie', [
    ('cnn_and_full_hsv_support_original', 60, 60.0, False),
    ('low_saturation', 59, 24.0, False), ('distant_original', 74, 60.01, False),
    ('original_not_unique_nearest', 74, 24.0, True)])
def test_support_boundary(reason: str, saturation: int, distance: float, tie: bool) -> None:
    values = {c: distance + 8 for c in A.COLOR_CODES}
    values[4], values[1] = distance, distance if tie else distance + 8
    assert A.support(4, (18, saturation, 247), values, 60, 60.0) == reason


@pytest.mark.parametrize('missing', ['cnn', 'disagree', 'frame', 'region', 'patch', 'exception'])
def test_missing_evidence_preserves_original(monkeypatch: Any, missing: str) -> None:
    original, result, cnn, frame, region = board(), board(1), board(), object(), object()
    if missing == 'cnn':
        cnn = None
    elif missing == 'disagree':
        cnn = board(1)
    elif missing == 'frame':
        frame = None
    elif missing == 'region':
        region = None
    def measure(*args: Any) -> Any:
        if missing == 'exception':
            raise RuntimeError('measurement_marker')
        return None
    monkeypatch.setattr(A, 'measure', measure)
    out, rows = A.apply_veto(original, result, cnn, frame, region)
    assert out is result and rows[0]['vetoed'] is False


def test_real_partial_writer_reproduction_and_two_cell_veto() -> None:
    import cv2
    saved = P.ROOT / 'partial_cpu_v1/CAPTURED_ROW.json'
    row = json.loads(saved.read_bytes())
    report = P.reproduce(row)
    events = {e['stage']: e for e in row['events']}
    before, after = events['next_validation_before'], events['next_validation_after']
    original, result = Board.from_list(before['confirmed']['grid']), Board.from_list(after['boards']['published_confirmed']['grid'])
    cnn = Board.from_list(before['boards']['cnn_board']['grid'])
    frame = cv2.resize(cv2.imread(str(P.IMAGE)), P.RUNTIME_SIZE)
    inputs = [v._grid.tolist() for v in (original, result, cnn)]
    out, rows = A.apply_veto(original, result, cnn, frame, DEFAULT_P2_REGION)
    assert report['original_result_matches_saved'] and out._grid.tolist() == original._grid.tolist()
    assert [(v['row'], v['col'], v['vetoed']) for v in rows] == [(12, 0, True), (12, 1, True)]
    assert [v._grid.tolist() for v in (original, result, cnn)] == inputs
    assert out is not original and out is not result


@pytest.mark.parametrize('replacement', [0, 9, 10])
def test_noncolor_and_gravity_removals_not_restored(monkeypatch: Any, replacement: int) -> None:
    original, result = board(), board(replacement)
    monkeypatch.setattr(A, 'measure', lambda *args: pytest.fail('excluded_cell_measured'))
    out, rows = A.apply_veto(original, result, board(), object(), object())
    assert out is result and rows == []


def test_hidden_row_not_visually_certified(monkeypatch: Any) -> None:
    original, result = board(4, row=0), board(1, row=0)
    monkeypatch.setattr(A, 'measure', lambda *args: pytest.fail('hidden_measured'))
    assert A.apply_veto(original, result, original, object(), object()) == (result, [])


def test_known_correlated_fifth_color_failure_is_explicit(monkeypatch: Any) -> None:
    # 物理真値赤を仮定してもCNN/HSVが共に紫なら誤読を保持する限界。
    monkeypatch.setattr(A, 'measure', lambda *args: measured(5))
    out, rows = A.apply_veto(board(5), board(1), board(5), object(), object())
    assert out.get(12, 0) == 5 and rows[0]['vetoed'] is True


def test_true_wrong_color_still_removed(monkeypatch: Any) -> None:
    monkeypatch.setattr(A, 'measure', lambda *args: measured(1))
    result = board(1)
    out, rows = A.apply_veto(board(5), result, board(5), object(), object())
    assert out is result and rows[0]['reason'] == 'original_not_unique_nearest'


def test_default_off_has_no_patch() -> None:
    assert A.install(None, None, None, {}, enabled=False) is None


def test_original_exception_identity() -> None:
    failure = RuntimeError('original_validate_marker')
    def original(*args: Any, **kwargs: Any) -> Any:
        raise failure
    with pytest.raises(RuntimeError) as caught:
        A.wrapper(original, None)(board(), [])
    assert caught.value is failure


def test_unknown_caller_rejected_descriptor_restored(tmp_path: Path) -> None:
    original = inspect.getattr_static(RecognitionPipeline, '_validate_next_history')
    state = {'output': tmp_path, 'atomic_journal_observer': SimpleNamespace(codes=set())}
    with contextlib.ExitStack() as stack:
        rec = A.install(stack, SimpleNamespace(RecognitionPipeline=RecognitionPipeline), None, state, enabled=True)
        with pytest.raises(ValueError, match='palette_unknown_generated_caller'):
            RecognitionPipeline._validate_next_history(board(), [])
    assert rec.closed and inspect.getattr_static(RecognitionPipeline, '_validate_next_history') is original


def test_saved_permission_and_mask_boundary(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(A, 'measure', lambda *args: measured())
    original, result = board(), board(1)
    out, rows = A.apply_veto(original, result, original, object(), object())
    assert (out._grid != 0).tolist() == (result._grid != 0).tolist()
    rec = A.Recorder(tmp_path, None, None)
    rec.save({'frame_idx': 1, 'time_sec': 1 / 60, 'side': '2P'}, rows)
    rec.close()
    A.finish({A.KEY: rec})
    assert A.verify(tmp_path)['vetoed_cells'] == 1
    receipt = json.loads((tmp_path / A.RECEIPT).read_bytes())
    receipt['physical_palette_certified'] = True
    (tmp_path / A.RECEIPT).write_text(A.encoded(receipt))
    with pytest.raises(ValueError, match='palette_permission'):
        A.verify(tmp_path)


@pytest.mark.parametrize('change', ['winner', 'cnn', 'row_bool', 'hidden', 'threshold'])
def test_saved_numeric_support_rechecked(monkeypatch: Any, change: str) -> None:
    monkeypatch.setattr(A, 'measure', lambda *args: measured())
    _, rows = A.apply_veto(board(), board(1), board(), object(), object())
    cell = rows[0]
    if change == 'winner':
        cell['distances'][1] = 0.0
    elif change == 'cnn':
        cell['cnn_color'] = 1
    elif change == 'row_bool':
        cell['row'] = True
    elif change == 'hidden':
        cell['row'] = 0
    else:
        cell['minimum_saturation'] = 0
    with pytest.raises(ValueError):
        A.verify_cell(cell)
