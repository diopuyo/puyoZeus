"""既発火の返し、最新の締切、攻め側自身の相殺を実台帳で検査する。"""
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.fire_board import Forecast
from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
from src.phase_j._live_ex_frozen.src import dl1k_retr_fix as fix
from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import ConnectedLedger, project, attack


@pytest.fixture(autouse=True)
def flags_off(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (fix.OWN_CHAIN_ENV, fix.LONG_ATTACK_ENV, fix.RECEIVER_FORECAST_ENV, retr.VISIBLE_ENV, retr.LANDING_ENV,
                 retr.H_FORECAST_ENV, retr.DEADLINE_EXTEND_ENV, retr.SECOND_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize('enabled,own_flag', [(False, False), (True, False), (True, True)])
def test_longer_receiver_forecast_offsets_once(monkeypatch: pytest.MonkeyPatch,
                                             enabled: bool, own_flag: bool) -> None:
    engine, value = case(monkeypatch, own=1, attack_amount=20)
    fc = Forecast(.9, (70,), (70, 2030), 'fire_board', value['boards'][0], .9)
    engine.retr_fire = SimpleNamespace(current=lambda side, now: fc if side == 0 else None)
    monkeypatch.setenv(fix.RECEIVER_FORECAST_ENV, str(int(enabled)))
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, str(int(own_flag)))
    if not own_flag:
        actual = retr.pending_with_fire(engine, 1, (1400,))
        assert actual == ([0, 10] if enabled else [19, 0])
    else:
        result, _ = retr.compute_retr(engine, value)
        assert result['retR_c_p1'] == 30
        assert result['retR_r_p1'] == -10
        assert engine.retr_diag[0]['I'] == 20
        assert engine.retr_diag[0]['own_source'] == 'receiver_forecast'


@pytest.mark.parametrize('steps,resolving', [((70,), True), ((70, 140), False)])
def test_receiver_forecast_requires_longer_active_chain(monkeypatch: pytest.MonkeyPatch,
                                                       steps: tuple[int, ...], resolving: bool) -> None:
    engine, value = case(monkeypatch)
    fc = Forecast(.9, (70,), steps, 'fire_board', value['boards'][0], .9)
    engine.retr_fire = SimpleNamespace(current=lambda side, now: fc)
    engine._resolving = lambda side: resolving
    monkeypatch.setenv(fix.RECEIVER_FORECAST_ENV, '1')
    assert fix.receiver_forecast(engine, 0) is None


def case(monkeypatch: pytest.MonkeyPatch, own: int = 10, attack_amount: int = 20) -> tuple:
    board = np.zeros((retr.ROWS, retr.COLS), dtype=np.int64)
    chains = [dict(started=0., finished=False, best=dict(scores=[n*70]), observed=[],
                   carry_at_fire=0, generated_at_fire=0) for n in (own, attack_amount)]
    core = ConnectedLedger()
    engine = SimpleNamespace(native=None, chains=chains, _resolving=lambda side: True,
        state=SimpleNamespace(t=1., game_idx=1, game_start_t=0., first_move_t=lambda: 0.,
            sides=[SimpleNamespace(board=board.copy()) for _ in retr.SIDES], sl=core,
            sl_snap=dict(margin_origin_sec=0., generated_p1=0, generated_p2=0)),
        retr_ama=SimpleNamespace(profile=lambda *a, **kw: [70*(i+1) for i in range(retr.DEPTH)]),
        retr_pre={}, retr_seed={})
    value = dict(project(core.core, [own, attack_amount], 1., [0., 0.], [2., 2.]),
                 rate=70, boards=[board.copy(), board.copy()], board_prediction_active=[True, True],
                 remaining=[own, attack_amount], reconciliation=[dict(times=[2.]), dict(times=[2.])])
    monkeypatch.setattr(retr, 'chain_end', lambda e, v, side, now: [10., 5.][side])
    monkeypatch.setattr(retr, 'seed_pairs', lambda *a: ([[1, 2]]*retr.DEPTH, 'ok'))
    return engine, value


@pytest.mark.parametrize('own,attacker', [(10, 20), (30, 20), (0, 20)])
def test_own_h0_and_no_double_offset(monkeypatch: pytest.MonkeyPatch, own: int, attacker: int) -> None:
    engine, value = case(monkeypatch, own, attacker)
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, '1')
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    result, why = retr.compute_retr(engine, value)
    assert result['retR_h_p1'] == 0
    assert result['retR_c_p1'] == result['retR_cvis_p1'] == own
    assert engine.retr_diag[0]['I'] == attacker
    assert result['retR_r_p1'] == result['retR_rvis_p1'] == attacker-own
    assert why[0] == 'own_chain'


def test_add_after_chain_and_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    monkeypatch.setattr(retr, 'chain_end', lambda e, v, side, now: [2., 5.][side])
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, '1')
    monkeypatch.setenv(retr.VISIBLE_ENV, '1')
    result, _ = retr.compute_retr(engine, value)
    h = int(result['retR_h_p1'])
    assert h > retr.BUSY_VISIBLE_HANDS
    assert result['retR_c_p1'] == 10+h
    assert result['retR_cvis_p1'] == 10+retr.BUSY_VISIBLE_HANDS
    assert result['retR_unseen_p1'] == h-retr.BUSY_VISIBLE_HANDS


def test_observed_without_prediction_never_carries_old_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    value['board_prediction_active'][0] = False
    engine.chains[0]['observed'] = [dict(count=1, score=700, t=.9)]
    engine.state.sl_snap['generated_p1'] = 4
    engine.retr_pre[0] = (1, (999., 999., 999., 1.))
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, '1')
    result, _ = retr.compute_retr(engine, value)
    assert result['retR_c_p1'] == 6
    assert result['retR_r_p1'] == 14
    assert math.isnan(result['retR_lethal_p1'])
    assert engine.retr_diag[0]['own_lower_bound']


@pytest.mark.parametrize('observed,expected', [([], []), ([dict(count=2)], [40, 320]),
    ([dict(count=1, score=700), dict(count=2)], [700, 320])])
def test_observed_score_lower_bound(observed: list, expected: list) -> None:
    assert fix.observed_scores(dict(observed=observed)) == expected


@pytest.mark.parametrize('reflected,expected', [(0, 10), (4, 6), (20, 0)])
def test_reflected_and_old_external_total(monkeypatch: pytest.MonkeyPatch, reflected: int, expected: int) -> None:
    engine, value = case(monkeypatch)
    engine.state.sl_snap['generated_p1'] = reflected
    engine.chains[0]['external_sent'] = 999
    assert fix.unreflected(engine, 0, [700]) == expected


def test_forecast_fallback_has_final_board(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    fc = Forecast(.9, (700,), (700, 1400), 'fire_board', value['boards'][0], .9)
    engine.retr_fire = SimpleNamespace(current=lambda side, now: fc if side == 0 else None)
    value['board_prediction_active'][0] = False
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, '1')
    result, _ = retr.compute_retr(engine, value)
    assert result['retR_c_p1'] == 30
    assert engine.retr_diag[0]['own_source'] == 'forecast'


@pytest.mark.parametrize('attacker_side', [0, 1])
@pytest.mark.parametrize('own_pending,sent,expected', [(100, 80, 0), (50, 80, 30), (0, 80, 80)])
def test_attacker_offsets_own_pending(attacker_side: int, own_pending: int, sent: int, expected: int) -> None:
    core = ConnectedLedger().core
    attack(core, 1-attacker_side, own_pending, 0.)
    before = pickle.dumps(core)
    amounts = [0, 0]
    amounts[attacker_side] = sent
    result = project(core, amounts, 1., [None, None], [None, None])
    assert result['pending'][1-attacker_side] == expected
    assert result['pending'][attacker_side] == max(0, own_pending-sent)
    assert pickle.dumps(core) == before


def test_incoming_fire_preserves_attacker_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    attack(engine.state.sl.core, 0, 15, 0.)
    fc = Forecast(.9, (1400,), (1400,), 'fire_board', value['boards'][1], .9)
    engine.retr_fire = SimpleNamespace(current=lambda side, now: fc if side == 1 else None)
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, '1')
    result, _ = retr.compute_retr(engine, value)
    assert engine.retr_diag[0]['I'] == 5
    assert result['retR_r_p1'] == -5


@pytest.mark.parametrize('flag', [None, '0', 'false'])
def test_flags_off_exact_bits(monkeypatch: pytest.MonkeyPatch, flag: str | None) -> None:
    engine, value = case(monkeypatch)
    baseline = pickle.dumps(retr.compute_retr(engine, value))
    diag = pickle.dumps(engine.retr_diag)
    if flag is not None:
        for name in (fix.OWN_CHAIN_ENV, fix.LONG_ATTACK_ENV):
            monkeypatch.setenv(name, flag)
    assert pickle.dumps(retr.compute_retr(engine, value)) == baseline
    assert pickle.dumps(engine.retr_diag) == diag


def test_long_attack_uses_new_ex_over_short_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    engine.chains[1]['best']['scores'] = [40]*10
    engine.chains[1]['observed'] = [dict(count=2, t=4.)]
    short = Forecast(.9, (40,), (40, 40), 'fire_board', value['boards'][1], .9)
    engine.retr_fire = SimpleNamespace(current=lambda side, now: short)
    monkeypatch.setenv(fix.LONG_ATTACK_ENV, '1')
    end = fix.updated_end(engine, value, 1, 2.)
    assert end >= 4.+8*retr.STEP_SEC
    assert end >= retr.duration(10)
    engine.chains[1]['best']['scores'] += [40]*2
    assert fix.updated_end(engine, value, 1, end) > end


def test_long_attack_no_prediction_observed_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    value['board_prediction_active'][1] = False
    engine.chains[1]['observed'] = [dict(count=3, t=8.)]
    monkeypatch.setenv(fix.LONG_ATTACK_ENV, '1')
    assert fix.updated_end(engine, value, 1, None) >= 8.+retr.STEP_SEC


def test_long_h_integration_reindexes_same_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    engine._resolving = lambda side: side == 1
    engine.chains[0] = None
    monkeypatch.setattr(retr, 'chain_score', lambda *args: (0, 0))
    monkeypatch.setattr(retr, 'chain_end', lambda *args: 2.)
    engine.chains[1]['best']['scores'] = [40]*10
    before, _ = retr.compute_retr(engine, value)
    monkeypatch.setenv(fix.LONG_ATTACK_ENV, '1')
    after, _ = retr.compute_retr(engine, value)
    assert after['retR_h_p1'] > before['retR_h_p1']
    assert after['retR_c_p1'] > before['retR_c_p1']
    assert engine.retr_diag[0]['long_attack_h']


def test_no_mutation_of_input_or_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    before = pickle.dumps((engine.chains, value, engine.state.sl.core))
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, '1')
    monkeypatch.setenv(fix.LONG_ATTACK_ENV, '1')
    retr.compute_retr(engine, value)
    assert pickle.dumps((engine.chains, value, engine.state.sl.core)) == before


def test_own_without_attack_remains_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, value = case(monkeypatch)
    monkeypatch.setattr(retr, 'chain_score', lambda *args: (0, 0))
    engine._resolving = lambda side: side == 0
    engine.chains[1] = None
    value['remaining'][1] = 0
    value['pending'] = [0, 10]
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, '1')
    result, why = retr.compute_retr(engine, value)
    assert why[0] == 'no_attack'
    assert math.isnan(result['retR_c_p1'])


@pytest.mark.parametrize('side', [0, 1])
def test_confirmed_landing_uses_actual_offset_order(monkeypatch: pytest.MonkeyPatch, side: int) -> None:
    engine, value = case(monkeypatch)
    core = engine.state.sl.core
    attack(core, 1-side, 15, 0.)
    core.batches[side][0].confirmed = True
    core.batches[side][0].confirmed_t = 0.
    # 先に到着した確定分15個と、後着の未確定分20個。返し10個は後者へ使う。
    attack(core, 1-side, 20, .5)
    value['remaining'] = [0, 0]
    assert fix.confirmed_after_own(engine, value, side, dict(fire_back=None), 10) == 15


@pytest.mark.parametrize('own,long', [(False, False), (True, False), (False, True), (True, True)])
def test_diagnostic_delivery_is_flagged(monkeypatch: pytest.MonkeyPatch, own: bool, long: bool) -> None:
    from src.phase_j.live_ex_compute import RETR_DIAG_KEYS, retr_details
    engine, value = case(monkeypatch)
    monkeypatch.setenv(fix.OWN_CHAIN_ENV, str(int(own)))
    monkeypatch.setenv(fix.LONG_ATTACK_ENV, str(int(long)))
    retr.compute_retr(engine, value)
    result = {}
    retr_details(engine, value, result)
    expected = tuple(dict.fromkeys(RETR_DIAG_KEYS + (fix.OWN_DIAG_KEYS if own else ()) +
                                   (fix.LONG_DIAG_KEYS if long else ())))
    assert tuple(result['retr_diag'][0]) == expected
    if own:
        assert result['retr_diag'][0]['own_chain'] == 10
