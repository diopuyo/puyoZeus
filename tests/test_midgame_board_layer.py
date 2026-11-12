"""中盤補正の規則、因果的持ち越し、排他、既定OFFを検査する。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.midgame_board_layer import (
    ENV_ENABLED, MODEL_PATH, board_shift, eligible, load_layer, settled_features, shift_probability,
)
from src.midgame_board_wrapper import MidgameBoardLayer, apply_midgame


def board(heights: tuple = (4, 4, 4, 4, 4, 4)) -> list[list[int]]:
    return [[1 + (r + c) % 3 if r >= 13 - heights[c] else 0 for c in range(6)]
            for r in range(13)]


def row(**changes: float) -> dict:
    result = {f'{key}_p{s}': 0 for key in
              ('chain_active', 'sl_pending', 'r_ledger_forecast', 'cells_ojama') for s in (1, 2)}
    result.update({f'{key}_p{s}': 24 for key in ('cells_total', 'cells_color') for s in (1, 2)})
    return dict(result, **changes)


def frame(t: float = 0, first: list | None = None, second: list | None = None) -> dict:
    return dict(t_sec=t, game_idx=1, sides=[
        dict(side='1P', confirmed_board=board() if first is None else first),
        dict(side='2P', confirmed_board=board((6, 4, 4, 4, 3, 3)) if second is None else second)])


def layer() -> MidgameBoardLayer:
    result = MidgameBoardLayer(load_layer(enabled=True))
    result.observe(frame())
    return result


def test_counts_and_hidden_row() -> None:
    assert settled_features(board()) == (24, 0)
    assert settled_features(board((6, 4, 4, 4, 3, 3))) == (24, 3)
    assert settled_features(board((13, 0, 0, 0, 0, 0))) == (13, 13)


@pytest.mark.parametrize('bad', [10, -1, 6, 11])
def test_unknown_or_invalid(bad: int) -> None:
    b = board()
    b[12][0] = bad
    assert settled_features(b) is None


def test_floating() -> None:
    b = board()
    b[11][0] = 0
    assert settled_features(b) is None


@pytest.mark.parametrize('cells', [[(12, c) for c in range(4)], [(r, 0) for r in range(9, 13)],
                                  [(12, 0), (12, 1), (11, 0), (11, 1)]])
def test_connected_four(cells: list) -> None:
    b = board()
    for r, c in cells:
        b[r][c] = 5
    assert settled_features(b) is None


def test_ojama_does_not_erase() -> None:
    b = board()
    b[12] = [9] * 6
    assert settled_features(b) == (18, 0)


@pytest.mark.parametrize('b', [None, [], [[0] * 6] * 12, [[0] * 5] * 13])
def test_bad_shape(b: list | None) -> None:
    assert settled_features(b) is None


@pytest.mark.parametrize('key', [f'{key}_p{s}' for key in
                                ('chain_active', 'sl_pending', 'r_ledger_forecast', 'cells_ojama')
                                for s in (1, 2)])
def test_gate_each_zero(key: str) -> None:
    assert not eligible(row(**{key: 1}))
    assert not eligible(row(**{key: math.nan}))


@pytest.mark.parametrize('total,expected', [(18, False), (19, True), (47, True), (48, False)])
def test_phase_edges(total: int, expected: bool) -> None:
    assert eligible(row(cells_total_p1=total, cells_total_p2=18)) is expected


@pytest.mark.parametrize('delta,expected', [(-5, False), (-4, True), (0, True), (4, True), (5, False)])
def test_color_edges(delta: int, expected: bool) -> None:
    assert eligible(row(cells_color_p1=24 + delta)) is expected


def test_missing_and_counter_gate() -> None:
    assert not eligible({})
    assert not eligible(row(), True)
    assert not eligible(row(cells_total_p1=math.nan))
    assert not eligible(row(cells_color_p1=math.inf))


def test_shift_symmetry_and_cap() -> None:
    model = load_layer(enabled=True)
    assert board_shift((24, 0), (24, 3), model) == -board_shift((24, 3), (24, 0), model)
    assert board_shift((24, 78), (24, 0), model) == -model['cap_logit']
    assert shift_probability(.3, .1) == pytest.approx(1 - shift_probability(.7, -.1))


def test_causal_carry_and_missing_side() -> None:
    obj = layer()
    obj.observe(frame(2, first=board((6, 4, 4, 4, 3, 3))))
    assert obj.target(1, row(), False) > 0
    assert obj.target(2, row(), False) == 0
    bad = board()
    bad[12][0] = 10
    obj.observe(frame(3, first=bad))
    assert obj.target(3, row(), False) == 0
    obj.reset()
    obj.observe(dict(t_sec=0, sides=[dict(side='1P', confirmed_board=board())]))
    assert obj.target(0, row(), False) is None


def test_ramp_decay_and_reentry() -> None:
    obj = layer()
    target = obj.target(0, row(), False)
    assert obj.apply(.5, 0, row()) == .5
    obj.apply(.5, .25, row())
    assert obj.shift == pytest.approx(target / 2)
    obj.apply(.5, .5, row())
    assert obj.shift == pytest.approx(target)
    obj.apply(.5, 1.5, row(chain_active_p1=1))
    assert obj.shift == pytest.approx(target)
    obj.apply(.5, 2.5, row(chain_active_p1=1))
    assert obj.shift == pytest.approx(target / math.e)
    previous = obj.shift
    obj.apply(.5, 2.6, row())
    assert obj.shift == 0
    obj.apply(.5, 3.1, row())
    assert obj.shift == pytest.approx(target)


def test_counter_clears_residual_immediately() -> None:
    obj = layer()
    obj.apply(.5, 0, row())
    obj.apply(.5, 1, row())
    obj.apply(.5, 1.1, row(chain_active_p1=1))
    assert obj.shift != 0
    assert obj.apply(.5, 1.2, row(), True) == .5
    assert obj.shift == 0
    assert obj.apply(.5, 3, row()) == .5


def test_same_timestamp_and_reset() -> None:
    obj = layer()
    obj.apply(.5, 0, row())
    p = obj.apply(.5, .25, row())
    assert obj.apply(.5, .25, row()) == p
    obj.reset()
    assert obj.apply(.5, 0, row()) == .5
    assert obj.target(0, row(), False) is None


@pytest.mark.parametrize('probability', [0.0, -0.0, .12345678901234567, 1.0])
def test_off_exact(probability: float) -> None:
    obj = MidgameBoardLayer(None)
    obj.observe({})
    actual = obj.apply(probability, 100, {})
    assert actual.hex() == probability.hex()
    assert shift_probability(probability, 0).hex() == probability.hex()
    assert obj.times == [[], []]


def test_hash_failure_and_default_on(tmp_path: Path, caplog: pytest.LogCaptureFixture,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.release_config import release_environment
    # 2026-11-12 user 採用で 0.6.5 は既定 ON。明示の '0' で OFF にできることも保つ。
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    assert load_layer() is not None
    assert release_environment({})[ENV_ENABLED] == '1'
    assert release_environment({ENV_ENABLED: '0'})[ENV_ENABLED] == '0'
    monkeypatch.setenv(ENV_ENABLED, '0')
    assert load_layer() is None
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    assert load_layer(tmp_path / 'absent', enabled=True) is None
    altered = tmp_path / 'bad.json'
    altered.write_bytes(MODEL_PATH.read_bytes() + b' ')
    assert load_layer(altered, enabled=True) is None
    assert 'SHA256' in caplog.text
    assert load_layer(altered, enabled=False) is None


def test_json_controls_cap_and_times() -> None:
    obj = layer()
    obj.model = dict(obj.model, cap_logit=.01, enter_sec=2, exit_sec=2)
    obj.apply(.5, 0, row())
    obj.apply(.5, 1, row())
    assert obj.shift == pytest.approx(.005)
    obj.apply(.5, 3, row(chain_active_p1=1))
    assert obj.shift == pytest.approx(.005)
    obj.apply(.5, 5, row(chain_active_p1=1))
    assert obj.shift == pytest.approx(.005 / math.e)


def test_real_counter_window_blocks_after_chain_end() -> None:
    from src.counter_fire_shift import CounterFireShift, load_shift
    owner = SimpleNamespace(midgame_board_layer=layer(), counter_fire_shift=CounterFireShift(load_shift(enabled=True)))
    for t, active in ((0., (0, 1)), (.1, (1, 1)), (.2, (0, 0))):
        r = row(chain_active_p1=active[0], chain_active_p2=active[1],
                mk_xc_p1=.2, max_height_p1=6)
        p = owner.counter_fire_shift.apply(.5, t, r)
        assert apply_midgame(owner, p, dict(t_sec=t, row=r), {}) == p
    assert owner.counter_fire_shift.context == (1, 0)
    assert owner.midgame_board_layer.shift == 0


@pytest.mark.parametrize('exit_row, values', [
    (row(chain_active_p2=1), {}),
    (row(sl_pending_p1=6), {}),
    (row(r_ledger_forecast_p2=1), {}),
    (row(cells_ojama_p1=3), {}),
    (row(), {'_light_scenarios': [{'weight': 1.0}]}),
])
def test_exchange_or_light_clears_residual_immediately(exit_row: dict, values: dict) -> None:
    owner = SimpleNamespace(midgame_board_layer=layer(), counter_fire_shift=None)
    for t in (0., 1.):
        apply_midgame(owner, .5, dict(t_sec=t, row=row()), {})
    assert owner.midgame_board_layer.shift != 0
    assert apply_midgame(owner, .6, dict(t_sec=1.1, row=exit_row), values) == .6
    assert owner.midgame_board_layer.shift == 0


def test_phase_exit_still_decays() -> None:
    owner = SimpleNamespace(midgame_board_layer=layer(), counter_fire_shift=None)
    for t in (0., 1.):
        apply_midgame(owner, .5, dict(t_sec=t, row=row()), {})
    target = owner.midgame_board_layer.shift
    apply_midgame(owner, .5, dict(t_sec=1.1, row=row(cells_color_p1=30)), {})
    assert owner.midgame_board_layer.shift == pytest.approx(target)


def test_live_reset_epoch_and_game(tmp_path: Path) -> None:
    from src.phase_j.live_24h import Live24h
    obj = Live24h(tmp_path, use_ama=False, counter_fire_shift_enabled=False,
                 midgame_board_layer_enabled=True)
    try:
        obj.update(frame(), 1, 0)
        assert obj.midgame_board_layer.target(0, row(), False) is not None
        obj.update(dict(t_sec=0, game_idx=2), 1, 0)
        assert obj.midgame_board_layer.target(0, row(), False) is None
        obj.update(frame(1), 2, 0)
        assert obj.midgame_board_layer.target(1, row(), False) is not None
        obj.update(dict(t_sec=0, game_idx=1), 3, 0)
        assert obj.midgame_board_layer.target(0, row(), False) is None
    finally:
        obj.close()


def test_research_vectors() -> None:
    """抜粋行なので付属状態を復元し、盤・条件・補正量・確率を比較する。"""
    from src.midgame_board_wrapper import MixingState, mix_shift
    path = Path(os.environ.get('PUYO_MIDGAME_TEST_VECTORS',
                               Path(__file__).parent / 'fixtures/midgame_board/vectors.json'))
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'd303c55d566278ee4a67a120fa1032fe084fef7a6af8086d483369031a6da221'
    data = json.loads(raw)
    assert len(data['vectors']) == data['n'] == 260
    model = load_layer(enabled=True)
    for v in data['vectors']:
        features = [settled_features(v['kept_settled_board'][s]['grid']) for s in ('1P', '2P')]
        assert all(features), v['row_key']
        first, second = features
        assert first[0] - second[0] == v['d_color'], v['row_key']
        assert first[1] - second[1] == v['d_bump'], v['row_key']
        for side in ('1P', '2P'):
            assert v['game_first_frame'] <= v['kept_settled_board'][side]['frame'] <= v['frame']
        inside = eligible(v['row'], v['counter_fire_window_research'])
        assert inside == v['in_layer'], v['row_key']
        target = board_shift(first, second, model)
        assert target == pytest.approx(max(-data['cap'], min(data['cap'], v['shift_v2_raw'])), abs=1e-12, rel=0)
        m = v['mixing_state']
        state = MixingState(m['last_applied'], m['t_enter'], m['t_exit'], inside)
        applied, _ = mix_shift(target if inside else None, v['t_sec'], state, model,
                               v['counter_fire_window_research'])
        assert applied == pytest.approx(v['shift_v2_applied'], abs=1e-12, rel=0), v['row_key']
        actual = shift_probability(v['p_temperature'], applied)
        assert actual == pytest.approx(v['p_final_v2'], abs=1e-12, rel=0), v['row_key']


@pytest.mark.parametrize('enabled', [False, True])
def test_live_after_temperature_and_off_bits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              enabled: bool) -> None:
    from src.phase_j.live_24h import Live24h
    from src.prob_temperature import apply_temperature, load_temperature
    temperature_path = Path(__file__).parent / 'fixtures/prob_temperature/prob_temperature_v1.json'
    live = Live24h(tmp_path, model=SimpleNamespace(predict_row=lambda r: .8), settings={},
                  worker_factory=lambda s: SimpleNamespace(close=lambda: None), use_ama=True,
                  temperature_path=temperature_path, temperature_enabled=True,
                  counter_fire_shift_enabled=False, midgame_board_layer_enabled=enabled)
    live.selection_audit = {}
    monkeypatch.setattr(live, 'selection', lambda r: ({}, 'same', 'test'))
    monkeypatch.setattr(live, 'flush_events', lambda: None)
    live.midgame_board_layer.observe(frame())
    result = dict(row=row(), frame=0, t_sec=0, epoch=1)
    calibrated = apply_temperature(.8, result['row'], load_temperature(temperature_path, True))
    assert live.predict(result).hex() == calibrated.hex()
    result.update(t_sec=1, frame=1)
    expected = (shift_probability(calibrated, live.midgame_board_layer.target(1, row(), False))
                if enabled else calibrated)
    assert live.predict(result).hex() == expected.hex()
    assert live.predict(result).hex() == expected.hex()
    live.close()
