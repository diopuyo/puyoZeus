"""実v59の2P MENU/盤面Noneを最小再現。確定盤面を人工ゼロへ補完しない。"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import collector_connector as C
import anchor as OLD

ROOT = Path(__file__).resolve().parent
SAVED = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v59_two_updates/collector_metadata.jsonl'


def actual_missing() -> tuple[dict, Any]:
    with SAVED.open(encoding='utf-8') as stream:
        rows = [json.loads(next(stream)) for _ in C.ANCHOR.SIDES]
    row = rows[1]
    args = row['arguments']
    assert row['side'] == '2P' and args['bstate']['value'] == 'menu' and args['board'] is None
    side = N(state=N(value=args['bstate']['value']), confirmed_board=None, score=args['score'],
             board_provenance=args['board_provenance'], chain_event=None)
    return row, side


def test_original_reproduces_actual_absence_error() -> None:
    row, side = actual_missing()
    with pytest.raises(ValueError, match='producer_capture:grid_shape'):
        OLD.checked_side(row, side, None)


def test_actual_none_is_recorded_not_qualified() -> None:
    row, side = actual_missing()
    result = C.ANCHOR.checked_side(row, side, None)
    assert result['grid'] is None and not result['checks']['board_present'] and not result['checks']['empty']
    assert result['arguments'] == row['arguments'] and result['checks']['stable'] is False


@pytest.mark.parametrize('case', ['metadata_only', 'result_only'])
def test_presence_mismatch_refused(case: str) -> None:
    row, side = actual_missing()
    if case == 'metadata_only': row['arguments']['board'] = [[0] * 6 for _ in range(13)]
    else: side.confirmed_board = object()
    with pytest.raises(ValueError, match='metadata_board_presence_mismatch'):
        C.ANCHOR.checked_side(row, side, None)


def test_absence_does_not_qualify_even_if_stable() -> None:
    row, side = actual_missing()
    row['arguments']['bstate']['value'] = side.state.value = 'stable'
    row['arguments']['score'] = side.score = 0
    result = C.ANCHOR.checked_side(row, side, None)
    assert result['checks']['stable'] and not all(result['checks'].values())
    assert not all(result['checks'][k] for k in C.ANCHOR.STABLE_CHECKS)
