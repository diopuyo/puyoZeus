"""liveの採用開始通知で累積状態を区切り、一時欠測では区切らない。"""
from copy import deepcopy

import pytest

from src.b3_realtime import QueueObservation, RealtimeRows
from src.b3_timing import TimingTable
from tests.test_b3_start_residue import _assert_empty, _frame

START = 10.0


def _queue(t: float, game: int = 1) -> tuple[QueueObservation, ...]:
    return (QueueObservation(0, '1P', round(t * 60), t, (1, 2, 3, 4), game),)


def _placement(t: float) -> tuple[dict, ...]:
    return (dict(kind='placement_signal', side='1P', frame=round(t * 60),
                 observed_at_sec=t, game_idx=0),)


def test_start_blocks_residue_and_delayed_events() -> None:
    rows = RealtimeRows(TimingTable(), measure_timing=False)
    assert not rows.advance(_frame(1, count=18), _placement(1))
    assert not rows.advance(_frame(2, state='OJAMA_FALL'))
    assert rows.advance(_frame(START), _placement(1), _queue(START))
    _assert_empty(rows.state)
    assert rows.state.game_start_t == START
    assert rows.state.sides[0].queue == (1, 2, 3, 4)
    assert rows.last_t == START


@pytest.mark.parametrize('gap', ['MENU', 'ocr', 'emit'])
def test_temporary_gap_preserves_state(gap: str) -> None:
    rows = RealtimeRows(TimingTable(), measure_timing=False)
    rows.advance(_frame(START), _placement(START), _queue(START))
    ledger = rows.state.sl
    frame = _frame(START + 1, state='MENU' if gap == 'MENU' else 'STABLE')
    if gap == 'ocr':
        for side in frame['sides']:
            side['score'] = None
    rows.advance(frame, emit=gap != 'emit')
    rows.advance(_frame(START + 2))
    assert rows.state.sl is ledger
    assert rows.state.sides[0].first_place_t == START
    assert rows.state.game_start_t == START


def test_adopted_boundary_resets_same_observation_game() -> None:
    rows = RealtimeRows(TimingTable(), measure_timing=False)
    rows.advance(_frame(START), _placement(START), _queue(START))
    rows.prev_icon, rows.prev_land = [99, 99], [99, 99]
    assert rows.advance(_frame(START + 1), queues=_queue(START + 1, game=2))
    _assert_empty(rows.state)
    assert rows.state.game_start_t == START + 1
    assert rows.prev_land == [0, 0]


def test_start_waits_for_valid_frame_and_discards_old_next() -> None:
    rows = RealtimeRows(TimingTable(), measure_timing=False)
    frame = deepcopy(_frame(START))
    frame['sides'][0]['state'] = 'MENU'
    assert not rows.advance(frame, queues=_queue(START))
    assert rows.advance(_frame(START + 1))
    assert rows.state.sides[0].queue == (0, 0, 0, 0)
    _assert_empty(rows.state)


def test_rounded_start_waits_and_discards_early_placement() -> None:
    """保存時刻の丸めがフレーム時刻より後なら、正本と同じ次フレームで開始する。"""
    rows = RealtimeRows(TimingTable(), measure_timing=False)
    stamp = START + 1 / 30
    queue = (QueueObservation(0, '1P', round(stamp * 60), stamp + 0.0003,
                              (1, 2, 3, 4), 1),)
    assert not rows.advance(_frame(stamp), _placement(stamp), queue)
    assert rows.advance(_frame(stamp + 1 / 30))
    assert rows.state.game_start_t == stamp + 1 / 30
    assert rows.state.sides[0].first_place_t is None
