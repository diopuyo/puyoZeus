"""小連鎖・発火基点・ツモ同期の既定OFF修正を検証する。"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_retr_estimate_fix import case
from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
from src.phase_j._live_ex_frozen.src import dl1k_retr_continue as continuation
from src.phase_j._live_ex_frozen.src import dl1k_retr_origin as origin
from src.phase_j._live_ex_frozen.src import dl1k_retr_tsumo as sync
from src.phase_j._live_ex_frozen.src.prefire_post_v47 import remaining_send
from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import project, attack
from src.fire_board import Forecast


def small_case(monkeypatch: pytest.MonkeyPatch) -> tuple:
    engine, value = case(monkeypatch)
    engine._resolving = lambda side: side == 1
    engine.state.sides[0].state, engine.state.sides[0].board_t = 'STABLE', 1.
    original = np.ones((13, 6), dtype=int)
    final = np.zeros((13, 6), dtype=int)
    engine.native = SimpleNamespace(simulate_chain_trace_py=lambda *args: [(list(original.ravel()), list(final.ravel()), [], 0, 140)])
    engine.state.sl.core = SimpleNamespace(_pending=lambda side, confirmed_only: 12)
    calls = []
    engine.retr_ama.profile = lambda board, pairs, **kw: calls.append((board, pairs, kw)) or [70*(i+1) for i in range(16)]
    monkeypatch.setenv(continuation.ENV, '1')
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    run = retr._RetrRun(engine, value)
    run.ends = [None, 10.]
    run.diag[0] = {}
    return run, original, calls


def test_small_chain_uses_final_board_and_reserves_time(monkeypatch: pytest.MonkeyPatch) -> None:
    run, board, calls = small_case(monkeypatch)
    assert continuation.apply(run, 0, board, 100, 14)
    h = int(run.extra['retR_h_p1'])
    assert 0 < h < 14
    assert run.extra['retR_c_p1'] == 2+h
    assert run.extra['retR_cvis_p1'] == 4
    assert run.extra['retR_r_p1'] == 98-h
    assert calls[0][0] == [0]*78
    assert calls[0][2] == dict(pending=10, rate=70)
    assert np.array_equal(board, np.ones((13,6)))


def test_small_chain_falling_board_is_not_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    run, board, calls = small_case(monkeypatch)
    run.engine.state.sides[0].state = 'OJAMA_FALL'
    assert continuation.apply(run, 0, board, 100, 14)
    assert not calls
    assert run.extra['retR_c_p1'] == 2
    assert math.isnan(run.extra['retR_lethal_p1'])
    assert math.isnan(run.engine.retr_pre[0][1][-1])
    assert run.diag[0]['small_chain']['after_why'] == 'landing_board_unresolved'


@pytest.mark.parametrize('flag,h', [('0',14),('1',0)])
def test_small_chain_disabled_or_no_h_keeps_old_route(monkeypatch: pytest.MonkeyPatch, flag: str, h: int) -> None:
    run, board, calls = small_case(monkeypatch)
    monkeypatch.setenv(continuation.ENV, flag)
    assert not continuation.apply(run, 0, board, 100, h)
    assert not calls


def test_small_chain_missing_seed_is_explicit_lower_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    run, board, calls = small_case(monkeypatch)
    monkeypatch.setattr(retr, 'seed_pairs', lambda *args: (None,'no_seed'))
    assert continuation.apply(run, 0, board, 100, 14)
    assert run.extra['retR_c_p1'] == 2 and not calls
    assert run.diag[0]['small_chain']['lower_bound']


def test_second_after_small_uses_after_start() -> None:
    after = dict(kind='ama',cummax=[0,700],h=2)
    assert retr.return_end(dict(kind='after_small',start=10.,end=9.,after=after),0.,None) == retr.return_end(after,10.,None)


def origin_case(monkeypatch: pytest.MonkeyPatch) -> object:
    engine, value = case(monkeypatch)
    engine.state.t = 110.
    engine.ledger_history = [(90., 5, 0, 69, 0), (100., 8, 0, 1, 0)]
    engine.chains[0].update(started=109., generated_at_fire=8, carry_at_fire=1,
                             signature=(109., 1, 700), best=dict(scores=[700]))
    fc = Forecast(94., (700,), (700,), 'fire_board', value['boards'][0], 110.)
    engine.retr_fire = SimpleNamespace(current=lambda side, now: fc if side == 0 else None)
    monkeypatch.setenv(origin.ENV, '1')
    return engine


def test_origin_preserves_margin_and_fraction_across_renotification(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = origin_case(monkeypatch)
    origin.observe(engine)
    held = engine.chains[0]
    assert (held['started'],held['generated_at_fire'],held['carry_at_fire']) == (94.,5,69)
    assert remaining_send(held, 8, 0.) == 7  # 95.5秒より前の700+69点は10個、反映済み3個を引く。
    engine.chains[0] = dict(held, started=110., generated_at_fire=9, carry_at_fire=20)
    origin.observe(engine)
    assert remaining_send(engine.chains[0], 8, 0.) == 7


def test_origin_keeps_real_ledger_and_own_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    import pickle
    engine = origin_case(monkeypatch)
    attack(engine.state.sl.core, 1, 10, 90.)
    before = pickle.dumps(engine.state.sl.core)
    origin.observe(engine)
    assert pickle.dumps(engine.state.sl.core) == before
    amount = remaining_send(engine.chains[0], 8, 0.)
    result = project(engine.state.sl.core, [amount,0], 110., [94.,None], [111.,None])
    assert result['pending'] == [3,0]
    assert pickle.dumps(engine.state.sl.core) == before


def test_origin_does_not_cross_games(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = origin_case(monkeypatch)
    origin.observe(engine)
    engine.state.game_idx = 2
    engine.ledger_history = [(90.,20,0,10,0)]
    origin.observe(engine)
    assert engine.chains[0]['generated_at_fire'] == 20


def test_origin_off_has_no_state_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    import pickle
    engine = origin_case(monkeypatch)
    before = pickle.dumps(engine.chains)
    monkeypatch.setenv(origin.ENV, '0')
    origin.observe(engine)
    assert pickle.dumps(engine.chains) == before
    assert not hasattr(engine, 'retr_origins')


@pytest.mark.parametrize('stable_key', [False, True])
def test_origin_hook_runs_for_research_and_live(monkeypatch: pytest.MonkeyPatch, stable_key: bool) -> None:
    from src.phase_j._live_ex_frozen.src.dl1k_engine import ExEngine
    from src.phase_j.live_ex_stable_key import StableKeyExEngine
    cls = StableKeyExEngine if stable_key else ExEngine
    engine = cls.__new__(cls)
    engine.observe = lambda *args: None
    engine.retr_fire = SimpleNamespace(observe=lambda *args: None)
    called = []
    monkeypatch.setenv('PUYO_EX_RETR', '1')
    monkeypatch.setattr(origin, 'observe', lambda value: called.append(value))
    engine.step(1., 1, [])
    assert called == [engine]


def sync_case(monkeypatch: pytest.MonkeyPatch) -> object:
    colors = np.full((1,24,2), 5)
    colors[0,5], colors[0,6] = (2,3), (4,1)
    engine = SimpleNamespace(state=SimpleNamespace(t=10.,game_idx=1),
        knowledge=SimpleNamespace(lookup=SimpleNamespace(table=SimpleNamespace(abs_colors=colors))))
    monkeypatch.setenv(sync.ENV, '1')
    sync.observe(engine,[dict(retr_next_pair=[3,2],retr_dnext_pair=[1,4])])
    return engine


@pytest.mark.parametrize('busy,expected', [(False,4),(True,5)])
def test_sync_unique_current_next_pair_position(monkeypatch: pytest.MonkeyPatch, busy: bool, expected: int) -> None:
    engine = sync_case(monkeypatch)
    assert sync.synchronize(engine,0,busy,0,3,16) == expected
    assert engine.retr_sync_diag[0]['changed']


def test_sync_ambiguous_pair_does_not_jump(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = sync_case(monkeypatch)
    engine.knowledge.lookup.table.abs_colors[0,7:9] = [[2,3],[4,1]]
    assert sync.synchronize(engine,0,False,0,3,16) == 3
    assert engine.retr_sync_diag[0]['reason'] == 'ambiguous'


@pytest.mark.parametrize('mode', ['missing','stale','game','off'])
def test_sync_never_uses_old_or_missing_observation(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    engine = sync_case(monkeypatch)
    if mode == 'missing':
        sync.observe(engine,[dict(retr_next_pair=None,retr_dnext_pair=[1,4])])
    elif mode == 'stale':
        engine.state.t += 1
    elif mode == 'game':
        engine.state.game_idx += 1
    else:
        monkeypatch.setenv(sync.ENV,'0')
    assert sync.synchronize(engine,0,False,0,3,16) == 3


def test_sync_research_keeps_lean_and_raw_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j._live_ex_frozen.src.dl1k_records import side_raw
    side=dict(state='STABLE',confirmed_board=[[0]*6]*13,score=0,next_slide_motion=False,
              next_pair=[2,3],dnext_pair=[4,1])
    monkeypatch.delenv(sync.ENV,raising=False)
    old=side_raw(side,(1,1,2,2))
    monkeypatch.setenv(sync.ENV,'1')
    new=side_raw(side,(1,1,2,2))
    assert new['next_pair'] == old['next_pair'] == dict(tuple=(1,1))
    assert new['retr_next_pair'] == [2,3]
    assert {k:v for k,v in new.items() if k not in sync.FIELDS} == old


def test_sync_live_saves_next_before_stable_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    from copy import deepcopy
    from src.phase_j.live_ex_capture import CaptureEx
    monkeypatch.setenv(sync.ENV,'1')
    capture=CaptureEx()
    side=dict(state=dict(state='STABLE'),confirmed_board=dict(board=dict(array=[[0]*6]*13)),
              next_pair=dict(tuple=(1,1)),dnext_pair=dict(tuple=(2,2)))
    row=dict(game_idx=1,_v49f=dict(raw=[side,side]))
    capture.capture(None,SimpleNamespace(media_sec=1.),row)
    row=deepcopy(row)
    row['_v49f']['raw'][0]['next_pair']=dict(tuple=(3,4))
    row['_v49f']['raw'][0]['dnext_pair']=dict(tuple=(5,1))
    current=capture.capture(None,SimpleNamespace(media_sec=2.),row)['raw'][0]
    assert current['next_pair'] == dict(tuple=(1,1))
    assert current['retr_next_pair'] == dict(tuple=(3,4))
    assert current['retr_dnext_pair'] == dict(tuple=(5,1))


def test_sync_input_key_covers_observations_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j._live_ex_frozen.src.dl1k_regen_state import input_keys
    engine=SimpleNamespace(epoch=0,policy={},state={},chains=[],histories=[],ledger_history=[],knowledge={},placements={},
        tracker=SimpleNamespace(verdicts={},physics={}),landing=SimpleNamespace(certs={},game=1))
    for flag in sync.FOLLOWUP_FLAGS:
        monkeypatch.setenv(flag,'0')
    base=input_keys(engine,{},[{},{}])
    monkeypatch.setenv(continuation.ENV,'1')
    assert input_keys(engine,{},[{},{}]) != base
    monkeypatch.setenv(continuation.ENV,'0')
    assert input_keys(engine,{},[{},{}]) == base
    monkeypatch.setenv(sync.ENV,'1')
    engine.retr_observed_next=[dict(t=1.,game=1,pairs=[(1,2),(3,4)])]
    before=input_keys(engine,{},[{},{}])
    engine.retr_observed_next[0]['pairs'][1]=(1,5)
    assert input_keys(engine,{},[{},{}]) != before
