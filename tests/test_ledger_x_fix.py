"""部分台帳と盤面火力を混同しないための境界試験。"""
from types import SimpleNamespace
import math

import pytest

from src.b3_score_ledger import FrameObs, SideObs, LedgerEvent, ScoreLedger, ScoreLedgerV2, snapshot_to_dict
from src.ledger_partial_sync import PartialSync, has_landing_space
from src.ledger_x_board import generated_range, board_record
from src.ledger_x_options import boundary_game


def observation(state: str = 'STABLE', space: bool = True) -> FrameObs:
    return FrameObs(1., 1, 1, True, tuple(SideObs(200, state=state, ojama_cells=0,
                                               landing_space=space) for _ in range(2)))


def states() -> list:
    return [SimpleNamespace(chain=None, queue=[]) for _ in range(2)]


def landing(side: int, amount: int) -> LedgerEvent:
    return LedgerEvent('observed_landing', side, 1., 1., (('board_increase', amount),))


@pytest.mark.parametrize('amount,space,expected', [(1, True, True), (29, True, True),
                                                 (30, True, False), (0, True, False),
                                                 (6, False, False)])
def test_sync_requires_under_cap_landing_on_both_sides(amount: int, space: bool, expected: bool) -> None:
    sync = PartialSync()
    assert not sync.observe(observation(), states(), [])
    assert not sync.observe(observation('OJAMA_FALL'), states(), [])
    assert not sync.observe(observation(space=space), states(), [landing(0, amount)])
    assert sync.observe(observation(space=space), states(), [landing(1, amount)]) == expected


def test_chain_invalidates_zero_evidence() -> None:
    sync = PartialSync(clear=[True, True])
    assert not sync.observe(observation('CHAIN'), states(), [])
    assert not sync.observe(observation(), states(), [])


def test_icon_zero_is_not_sync_evidence() -> None:
    sync = PartialSync()
    assert not sync.observe(observation(), states(), [])
    assert sync.clear == [False, False]


def test_landing_space_rejects_unknown_or_full_columns() -> None:
    board = [[0] * 6 for _ in range(13)]
    assert has_landing_space(board)
    board[0][0] = 10
    assert not has_landing_space(board)
    assert not has_landing_space(None)


def test_partial_ledger_is_finite_but_still_marked_incomplete() -> None:
    core = ScoreLedgerV2(partial_sync=True)
    for i in range(10):
        snap = core.update(FrameObs(i / 30, i, 1, True, (SideObs(200), SideObs(200))))
    off = snapshot_to_dict(snap)
    on = snapshot_to_dict(snap, partial_sync=True)
    assert math.isnan(off['pending_p1'])
    assert math.isfinite(on['pending_p1'])
    assert on['pending_partial'] and on['history_incomplete']
    assert 'pending_partial' not in off


@pytest.mark.parametrize('ledger,low,high', [({}, 10, 769),
                                           ({'carry': 0}, 10, 700),
                                           ({'first_move': 0}, 10, 10),
                                           ({'carry': 1, 'first_move': 0}, 10, 10)])
def test_unknown_rate_and_carry_are_bounded_not_zero_filled(ledger: dict, low: int, high: int) -> None:
    result = generated_range([700], 1., ledger)
    assert (result['low'], result['high']) == (low, high)
    assert result['exact'] == (low if low == high else None)


def test_unknown_carry_can_straddle_one_cell() -> None:
    assert generated_range([701], 1., {'first_move': 0}) == dict(low=10, high=11, exact=None)


def test_boundary_uses_adopted_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = dict(game_idx=0, adopted_game=6)
    monkeypatch.delenv('PUYO_X_B3_BOUNDARY', raising=False)
    assert boundary_game(frame) == 0
    monkeypatch.setenv('PUYO_X_B3_BOUNDARY', '1')
    assert boundary_game(frame) == 6
    assert boundary_game(dict(game_idx=0, adopted_game=None)) == 0


def test_board_calculation_without_formula_carry_or_start() -> None:
    native = SimpleNamespace(simulate_chain_summary_py=lambda board, hidden: ([4], [40], [0] * 78))
    event = dict(trigger_sec=1., before_board=dict(board=dict(array=[[0] * 6 for _ in range(13)])))
    source = dict(chain_event=dict(namespace=event))
    value = board_record(2., 6, [source, {}], [None, None], None, native)
    side = value['board_firepower'][0]
    assert side['chain_count'] == 1 and side['score_points'] == 40
    assert side['carry'] is None and side['first_move'] is None and side['net_cells'] is None
    assert value['board_only'] and value['net_cells'] is None


def board_engine() -> object:
    """モデル推論を使わず、公開口から盤面票までの経路を試す。"""
    from src.phase_j.live_v49f_engine import LiveV49FEngine
    engine = LiveV49FEngine.__new__(LiveV49FEngine)
    native = SimpleNamespace(simulate_chain_summary_py=lambda board, hidden: ([4], [40], [0] * 78))
    engine.base = SimpleNamespace(observe=lambda *args: None, chains=[None, None], native=native,
                                  state=SimpleNamespace(sl_snap=None))
    engine.engine = SimpleNamespace(_boundary=lambda *args: None, _current=lambda *args: None)
    engine.latest, engine.previous, engine.game = None, None, None
    engine.board_game, engine.board_start = None, None
    return engine


@pytest.mark.parametrize('stamp,start,trigger', [
    (771.751832, 771.717967, 762.278855),
    (816.451816, 816.417983, 813.743783),
    (818.136393, 816.417983, 813.743783),
    (1176.192239, 1176.158431, 1170.455386),
    (1341.500993, 1341.467559, 1337.537559),
])
def test_board_rejects_five_previous_game_events(
    monkeypatch: pytest.MonkeyPatch, stamp: float, start: float, trigger: float,
) -> None:
    """24dの待機5行では、前区間の発火から公開票を作らない。"""
    monkeypatch.setenv('PUYO_X_BOARD_ONLY', '1')
    monkeypatch.setenv('PUYO_X_B3_BOUNDARY', '0')
    engine = board_engine()
    engine.observe(start, 1, [{}, {}], [])
    event = dict(trigger_sec=trigger, before_board=dict(board=dict(array=[[0] * 6 for _ in range(13)])))
    raw = [dict(chain_event=dict(namespace=event)), {}]
    assert engine.observe(stamp, 1, raw, []) == (None, None)


@pytest.mark.parametrize('boundary_flag', ['0', '1'])
def test_board_boundary_is_adopted_even_when_local_game_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, boundary_flag: str,
) -> None:
    monkeypatch.setenv('PUYO_X_BOARD_ONLY', '1')
    monkeypatch.setenv('PUYO_X_B3_BOUNDARY', boundary_flag)
    engine = board_engine()
    engine.observe(1., 0, [{}, {}], [], adopted_game=14)
    event = dict(trigger_sec=2., before_board=dict(board=dict(array=[[0] * 6 for _ in range(13)])))
    raw = [dict(chain_event=dict(namespace=event)), {}]
    assert engine.observe(2., 0, raw, [], adopted_game=14)[1] is not None
    assert engine.observe(3., 0, raw, [], adopted_game=15)[1] is None
    event['trigger_sec'] = 3.
    assert engine.observe(4., 0, raw, [], adopted_game=15)[1] is not None


def test_board_boundary_filters_each_side_before_using_saved_candidate() -> None:
    event = lambda stamp: dict(chain_event=dict(namespace=dict(trigger_sec=stamp)))
    chains = [dict(signature=(1.,), best=dict(scores=[40])),
              dict(signature=(2.,), best=dict(scores=[100]))]
    value = board_record(3., 1, [event(1.), event(2.)], chains, None, None, game_start=2.)
    assert value['board_firepower'][0] is None
    assert value['board_firepower'][1]['score_points'] == 100


def test_unknown_board_is_missing_not_zero() -> None:
    event = dict(trigger_sec=1., before_board=dict(board=dict(array=[[10] * 6 for _ in range(13)])))
    assert board_record(2., 1, [dict(chain_event=dict(namespace=event)), {}], [None, None], None, None) is None
