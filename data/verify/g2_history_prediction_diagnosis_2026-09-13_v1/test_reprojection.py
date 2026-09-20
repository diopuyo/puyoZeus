"""原台帳で正例・空再予測・異常値・世代移行・保存失敗を分離する。"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
for name in ('g2_async_projected_evaluation_2026-09-13_v1', 'g2_belief_live_publication_2026-09-11_v1'):
    sys.path.insert(0, str(ROOT.parent / name))
from src import chain_prediction_ledger_v1 as L
from src.board import Board
from src.chain import ChainSimulator
import journal_origin_capture_candidate as C
import fixed_origin_reference as F

AVAILABLE, FIRST, EMPTY_FRAME = 35160, 31330, 31332


def row(*, positive: bool = False, frame: int = EMPTY_FRAME, revision: int = 38) -> dict:
    value = json.loads((ROOT / 'REPLAY_v1.json').read_bytes())['source_row']
    value['frame_idx'], value['time_sec'] = frame, frame / C.FPS
    value['generation']['action_revision'] = value['generation_after']['action_revision'] = revision
    raw = deepcopy(value['events'][0]['active_origin'])
    if positive:
        grid = [[0] * 6 for _ in range(13)]
        grid[-1][:4] = [1] * 4
        raw.update(object_id=frame, trigger_sec=frame / C.FPS, end_sec=frame / C.FPS,
                   mechanism='landing', chain_count=1, total_score=40)
        raw['before_board'] = dict(grid=grid, sha256=hashlib.sha256(C.encoded(grid).encode()).hexdigest())
    value['events'] = [dict(stage='origin_after', active_origin=raw)]
    value['returned']['active_origin'] = deepcopy(raw)
    return value


def capture(simulator: Any = None, stream: Any = None) -> Any:
    return C.OriginCapture(None, None, None, L, simulator or ChainSimulator(), Board.from_dict,
                           lambda raw: False, io.StringIO() if stream is None else stream)


def test_existing_root_unchanged_and_hold_saved_once() -> None:
    value = capture()
    value.consume(row(positive=True, frame=FIRST), available_frame=AVAILABLE)
    handle = value.handles['2P']
    before = value.ledger.snapshot(handle)
    held = value.consume(row(), available_frame=AVAILABLE)
    value.consume(row(), available_frame=AVAILABLE)
    assert held['reasons'] == ['origin_reprojection_zero_chain_hold']
    assert value.ledger.snapshot(handle) == before
    packets = [json.loads(line) for line in value.stream.getvalue().splitlines()]
    assert len(packets) == 1 and packets[0]['recorded_frame'] == EMPTY_FRAME
    assert packets[0]['available_frame'] == AVAILABLE and packets[0]['prediction'] is None
    assert not packets[0]['ledger_modified'] and not packets[0]['quality_gate_clear']
    assert F.qualification(value, row(), value.ledger.snapshot(handle), '2P') == 'returned_origin_not_consumed'
    value.close()
    assert value.reprojection_holds == {} and value.ledger is None


def test_first_zero_landing_does_not_use_generation_or_block_fresh_origin() -> None:
    value, zero = capture(), row()
    zero['events'][0]['active_origin']['mechanism'] = 'landing'
    value.consume(zero, available_frame=AVAILABLE)
    assert not value.handles and not value.seen['2P']
    result = value.consume(row(positive=True, frame=EMPTY_FRAME + 2), available_frame=AVAILABLE)
    assert result['origin']['chain_count'] == 1
    value.close()


@pytest.mark.parametrize('broken', ['steps', 'total', 'negative', 'bool'])
def test_malformed_result_still_rejected(broken: str) -> None:
    real = ChainSimulator()
    class Bad:
        def simulate(self, board: Any) -> Any:
            result = real.simulate(board)
            changes = {'steps': {'steps': [object()]}, 'total': {'total_erased': 1},
                       'negative': {'chain_count': -1}, 'bool': {'chain_count': False}}
            return replace(result, **changes[broken])
    value, zero = capture(Bad()), row()
    zero['events'][0]['active_origin']['mechanism'] = 'landing'
    with pytest.raises((ValueError, L.ChainPredictionLedgerError)):
        value.consume(zero, available_frame=AVAILABLE)
    assert not value.reprojection_holds['2P']
    value.close()


def test_retired_origin_is_not_retried_but_new_generation_landing_recovers() -> None:
    value = capture()
    value.consume(row(positive=True, frame=FIRST), available_frame=AVAILABLE)
    value.consume(row(), available_frame=AVAILABLE)
    retired = value.consume(row(frame=EMPTY_FRAME + 2, revision=39), available_frame=AVAILABLE)
    assert 'origin_from_retired_software_generation' in retired['reasons']
    fresh = value.consume(row(positive=True, frame=EMPTY_FRAME + 4, revision=39), available_frame=AVAILABLE)
    assert fresh['origin']['chain_count'] == 1
    assert value.reprojection_holds['2P'] == {}
    value.close()


def test_hold_write_failure_propagates_before_ledger_change() -> None:
    class BadWriter(io.StringIO):
        def write(self, value: str) -> int:
            raise OSError('planned_hold_save_failure')
    value, zero = capture(stream=BadWriter()), row()
    zero['events'][0]['active_origin']['mechanism'] = 'landing'
    with pytest.raises(OSError, match='planned_hold_save_failure'):
        value.consume(zero, available_frame=AVAILABLE)
    assert not value.handles and not value.reprojection_holds['2P']
    value.close()
