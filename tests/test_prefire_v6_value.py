"""軽量比較の特徴・極性・選択後の評価回数を確認する。"""
from dataclasses import replace

import numpy as np
import pytest

from src import prefire_v5_search as base
from src import prefire_v6_value as v6
from src.prefire_v5c_search import TransitionTable


def position(fill: int = 0) -> base.Position:
    """比較用の同じ形状の盤面。"""
    return base.Position(bytes([fill])*v6.CAPACITY, (1, 2, 3, 4, 1, 2))


@pytest.mark.parametrize('fill', range(11))
def test_board_normalized(fill: int) -> None:
    assert 0 <= v6.board_value(position(fill).board) <= 1


@pytest.mark.parametrize('pending', [0, 1, 6, 30, 78, 1000])
def test_swap_sign(pending: int) -> None:
    states = (replace(position(), pending=pending), position(9))
    exchange = base.Exchange(states, (0, 0), (0, 0), (False, True))
    reverse = base.Exchange(states[::-1], (0, 0), (0, 0), (True, False))
    assert v6.light_value(exchange, 0) == -v6.light_value(reverse, 0)


def test_death_worse() -> None:
    exchange = base.Exchange((position(), position()), (0, 0), (0, 0), (False, False))
    assert v6.light_value(replace(exchange, dead=(True, False)), 0) < v6.light_value(exchange, 0)


def test_pending_worse() -> None:
    exchange = base.Exchange((position(), position()), (0, 0), (0, 0), (False, False))
    changed = replace(exchange, sides=(replace(position(), pending=30), position()))
    assert v6.light_value(changed, 0) < v6.light_value(exchange, 0)


@pytest.mark.parametrize('p', [0, 1, .1, .5, .9])
def test_logit_finite(p: float) -> None:
    assert np.isfinite(v6.logit(p))


@pytest.mark.parametrize('attacker', [0, 1])
def test_only_selected_and_wait_evaluated(monkeypatch: pytest.MonkeyPatch, attacker: int) -> None:
    attack = replace(position(), score=40, chains=1, consumed=1)
    wait = replace(position(), consumed=3)
    response = replace(position(), consumed=1)
    def selected(*args: object, **kwargs: object) -> tuple:
        return (0., wait if kwargs.get('waiting') else attack, response)
    monkeypatch.setattr(v6, 'select', selected)
    calls = []
    def evaluate(exchange: base.Exchange, elapsed: float) -> float:
        calls.append(exchange)
        return .7 if exchange.sides[attacker].chains else .3
    result, gap = v6.choose((position(), position()), attacker, 0, TransitionTable(), evaluate)
    assert len(calls) == 2
    assert result[0] == (.7 if attacker == 0 else .3)
    assert gap >= 0


def test_all_candidates_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = (position(), replace(position(), chains=1))
    monkeypatch.setattr(v6, 'side_options', lambda *args: candidates)
    def choose(states: tuple, attacker: int, elapsed: float, fn: object, **kwargs: object) -> None:
        assert kwargs['options_fn'](states[0], None, 0) == candidates
        assert kwargs['options_fn'](states[1], None, 1) == candidates
    monkeypatch.setattr(v6.exact, 'choose', choose)
    v6.select((position(), position()), 0, 0, TransitionTable())


def test_unknown_samples_keep_expectation(monkeypatch: pytest.MonkeyPatch) -> None:
    """最悪標本1つの値へ置き換えず、選択手の全標本を本番評価して平均する。"""
    attack = replace(position(), consumed=1)
    samples = tuple(replace(position(), queue=(1, 2, 3, 4, c, c)) for c in (1, 2))
    monkeypatch.setattr(v6.exact, 'response_states', lambda *args: samples)
    def choose(states: tuple, side: int, elapsed: float, fn: object, **kwargs: object) -> tuple:
        assert kwargs['options_fn'](states[0], None, 0) == (attack,)
        return (0., attack, states[1])
    monkeypatch.setattr(v6.exact, 'choose', choose)
    seen = []
    def evaluate(exchange: base.Exchange, elapsed: float) -> float:
        marker = exchange.sides[1].queue[-1]
        seen.append(marker)
        return .2 if marker == 1 else .8
    value = v6.selected_value((position(), position()), (0., attack, samples[0]),
                              0, 0., TransitionTable(), evaluate, None)
    assert value[0] == pytest.approx(.5)
    assert seen == [1, 2]
