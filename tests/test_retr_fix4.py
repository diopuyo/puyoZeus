"""落下開始の台帳差分と過去NEXT履歴だけを使う経路の試験。"""
from types import SimpleNamespace

import pytest

from src.phase_j._live_ex_frozen.src import dl1k_retr_fall as fall
from tests.test_retr_survivor_followup import small_case
from tests.test_retr_survivor_followup import sync_case
from src.phase_j._live_ex_frozen.src import dl1k_retr_history as history
from src.phase_j._live_ex_frozen.src import dl1k_retr_tsumo as sync


def fall_engine() -> SimpleNamespace:
    sides = [SimpleNamespace(state='STABLE', board_t=1.) for _ in range(2)]
    snap = dict(pending_confirmed_p1=40, pending_confirmed_p2=0, offset_p1=0, offset_p2=0)
    return SimpleNamespace(state=SimpleNamespace(game_idx=1, t=1., sides=sides, sl_snap=snap))


def start_fall(engine: object, pending: int = 10) -> None:
    fall.observe(engine)
    engine.state.t = 2.
    engine.state.sides[0].state = 'OJAMA_FALL'
    engine.state.sl_snap['pending_confirmed_p1'] = pending
    fall.observe(engine)


def test_fall_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fall.ENV, raising=False)
    engine = fall_engine()
    start_fall(engine)
    assert not hasattr(engine, 'retr_fall_rows')
    assert fall.amount(engine, 0) is None


def test_fall_delta_only_once_and_new_board_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fall.ENV, '1')
    engine = fall_engine()
    start_fall(engine)
    fall.observe(engine)
    assert fall.amount(engine, 0) == 30
    engine.state.sides[0].board_t = 3.
    fall.observe(engine)
    assert fall.amount(engine, 0) is None


@pytest.mark.parametrize('offset,pending', [(30, 10), (0, 40)])
def test_fall_no_evidence_or_offset_is_missing(monkeypatch: pytest.MonkeyPatch, offset: int, pending: int) -> None:
    monkeypatch.setenv(fall.ENV, '1')
    engine = fall_engine()
    fall.observe(engine)
    engine.state.sl_snap['offset_p1'] = offset
    engine.state.sl_snap['pending_confirmed_p1'] = pending
    engine.state.t = 2.
    engine.state.sides[0].state = 'OJAMA_FALL'
    fall.observe(engine)
    assert fall.amount(engine, 0) is None


def test_fall_game_change_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fall.ENV, '1')
    engine = fall_engine()
    start_fall(engine)
    engine.state.game_idx = 2
    fall.observe(engine)
    assert fall.amount(engine, 0) is None


def test_fall_missing_ledger_at_menu_is_not_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fall.ENV, '1')
    engine = fall_engine()
    engine.state.sl_snap = None
    fall.observe(engine)
    assert fall.amount(engine, 0) is None


def test_small_fall_restores_pending_for_landing(monkeypatch: pytest.MonkeyPatch) -> None:
    run, board, calls = small_case(monkeypatch)
    monkeypatch.setenv(fall.ENV, '1')
    run.engine.state.sides[0].state = 'OJAMA_FALL'
    run.engine.retr_fall_game = run.engine.state.game_idx
    run.engine.retr_fall_rows = [dict(started=2., board_t=1., amount=30), None]
    assert fall.apply(run, 0, board, 100, 14)
    assert calls[0][2]['pending'] == 40  # 12 + 落下開始30 - 小連鎖2。
    assert run.diag[0]['fall_board']['restored'] == 30


def observe_window(engine: object, first: list, second: list) -> None:
    raw = [dict(retr_next_pair=first, retr_dnext_pair=second)]
    engine.state.t += 1.
    sync.observe(engine, raw)
    engine.state.t += .2
    sync.observe(engine, raw)


def test_history_disambiguates_using_previous_window(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = sync_case(monkeypatch)
    monkeypatch.setenv(history.ENV, '1')
    colors = engine.knowledge.lookup.table.abs_colors
    colors[0,4], colors[0,7:9] = (1,2), [[2,3],[4,1]]
    observe_window(engine, [1,2], [2,3])
    observe_window(engine, [2,3], [4,1])
    assert sync.synchronize(engine, 0, False, 0, 2, 16) == 4
    assert engine.retr_sync_diag[0]['candidates'] == [5,7]
    assert engine.retr_sync_diag[0]['history_candidates'] == [5]


def test_history_does_not_break_remaining_ambiguity(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = sync_case(monkeypatch)
    monkeypatch.setenv(history.ENV, '1')
    engine.knowledge.lookup.table.abs_colors[0,7:9] = [[2,3],[4,1]]
    observe_window(engine, [2,3], [4,1])
    assert sync.synchronize(engine, 0, False, 0, 2, 16) == 2
    assert engine.retr_sync_diag[0]['reason'] == 'ambiguous'


def test_history_missing_current_never_reuses_past(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = sync_case(monkeypatch)
    monkeypatch.setenv(history.ENV, '1')
    observe_window(engine, [2,3], [4,1])
    observe_window(engine, None, None)
    assert sync.synchronize(engine, 0, False, 0, -1, 16) == -1


def test_history_new_game_clears_previous_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = sync_case(monkeypatch)
    monkeypatch.setenv(history.ENV, '1')
    observe_window(engine, [2,3], [4,1])
    engine.state.game_idx += 1
    sync.observe(engine, [dict(retr_next_pair=[1,2], retr_dnext_pair=[2,3])])
    assert engine.retr_history_tape.sides[0].own == []


def test_history_alone_saves_raw_without_changing_lean(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j._live_ex_frozen.src.dl1k_records import side_raw
    monkeypatch.setenv(history.ENV, '1')
    monkeypatch.setenv(sync.ENV, '0')
    side = dict(state='STABLE', confirmed_board=None, score=0, next_slide_motion=False,
                next_pair=[2,3], dnext_pair=[4,1])
    raw = side_raw(side, (1,1,2,2))
    assert raw['next_pair'] == dict(tuple=(1,1))
    assert raw['retr_next_pair'] == [2,3]


def test_history_identical_windows_do_not_fake_a_placement_count() -> None:
    a, b, c = (1,2), (3,4), (1,5)
    colors = (a,b,b,b,c)
    windows = [(a,b),(b,b),(b,c)]
    assert history.matches_windows(colors, windows, 3)
    assert not history.matches_windows((a,b,c,b,c), windows, 3)


def test_history_does_not_skip_different_missing_windows() -> None:
    a, b, c = (1,2), (3,4), (1,5)
    assert not history.matches_windows((a,b,c,b,c), [(a,b),(b,c)], 3)


def test_history_recovers_short_lean_without_importing_a_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
    engine = sync_case(monkeypatch)
    monkeypatch.setenv(history.ENV, '1')
    table = engine.knowledge.lookup.table
    table.predict_absolute = lambda seed, start, depth: table.abs_colors[seed,start:start+depth]
    engine.retr_seed = {1:0}
    engine.knowledge.tape = SimpleNamespace(sides=[SimpleNamespace(queue=None, own=[])])
    observe_window(engine, [2,3], [4,1])
    pairs, reason = retr.seed_pairs(engine, 0, False)
    assert reason == 'ok' and len(pairs) == retr.DEPTH
    assert pairs[1:3] == [(2,3),(4,1)]
    engine.retr_seed.clear()
    assert retr.seed_pairs(engine, 0, False) == (None, 'no_seed')


def test_new_flags_are_part_of_the_request_key(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimpleNamespace()
    key = {}
    sync.add_input_key(engine, key)
    assert key == {}
    monkeypatch.setenv(fall.ENV, '1')
    fall_key = {}
    sync.add_input_key(engine, fall_key)
    assert 'retr_fall_board' in fall_key
    monkeypatch.setenv(history.ENV, '1')
    history_key = {}
    sync.add_input_key(engine, history_key)
    assert history_key != fall_key and 'retr_sync_history' in history_key


def test_fall_on_does_not_enable_small_chain_for_stable_board(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j._live_ex_frozen.src import dl1k_retr_continue as small
    run, board, calls = small_case(monkeypatch)
    monkeypatch.setenv(small.ENV, '0')
    monkeypatch.setenv(fall.ENV, '1')
    assert not small.apply(run, 0, board, 100, 14)
    assert calls == []
