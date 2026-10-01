"""反映係数の境界・入力検査とセット分離の回帰試験。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.prefire_best_play_v6 import FEATURE_NAMES, Strength, BestPlayV6Layer, assignment_confidence
from src.prefire_v5_search import Position, OK, MISSING
from src.prefire_v5b_queue import StableQueuesV5B


@pytest.mark.parametrize('coefficient', [-1000., -5., 0., 5., 1000.])
def test_strength_bounded(coefficient: float) -> None:
    assert 0 <= Strength((coefficient, 0., 0., 0., 0.))((1., 0., 0., 1.)) <= 1


@pytest.mark.parametrize('coefficients', [(), (0.,), (0.,)*4, (0.,)*6, (float('nan'),)*5])
def test_invalid_strength(coefficients: tuple) -> None:
    with pytest.raises(ValueError):
        Strength(coefficients)


def test_identity() -> None:
    assert Strength((0.,)*5, identity=True)((1., 0., 0., 1.)) == 1


@pytest.mark.parametrize('training_set', ['zenchi_set2_58', 'unknown'])
def test_reject_other_training_set(tmp_path: Path, training_set: str) -> None:
    path = tmp_path/'strength.json'
    path.write_text(json.dumps(dict(training_set=training_set, features=list(FEATURE_NAMES),
                                    coefficients=[0.]*5)))
    with pytest.raises(ValueError):
        Strength.load(path)


def test_load_set1(tmp_path: Path) -> None:
    path = tmp_path/'strength.json'
    path.write_text(json.dumps(dict(training_set='zenchi_set1_57', features=list(FEATURE_NAMES),
                                    coefficients=[0.]*5)))
    assert Strength.load(path)((1., 0., 0., 1.)) == .5


@pytest.mark.parametrize('now,status,used', [(0., OK, False), (1., OK, True), (1., MISSING, False)])
def test_latency_and_missing(now: float, status: int, used: bool) -> None:
    layer = BestPlayV6Layer(strength=Strength((0.,)*5))
    state = Position(bytes(78), (1, 2, 3, 4, 1, 2), consumed=1)
    key = ((state, state), 0.)
    layer._cache[key] = ((.8, state, state), (.8, state, state))
    layer._ready[key] = 1.
    tracker = SimpleNamespace(probability=.5, source='G_fe')
    layer._show_v5(tracker, key, (status, OK), now, 1)
    assert layer.trace[-1][-2] == used
    assert tracker.probability == pytest.approx(2/3 if used else .5)


def test_assignment_confidence_uses_observation_evidence() -> None:
    queue = StableQueuesV5B()
    assert assignment_confidence(queue) == 0
    for side in queue.sides:
        side.shifted = True
        for reading in side.pairs:
            reading.run = reading.accepted = (1, 2)
            reading.count = 3
    assert assignment_confidence(queue) == 1
    queue.sides[0].pairs[0].run = (2, 1)
    assert assignment_confidence(queue) == pytest.approx(2/3)


def test_v6_cli_default_off() -> None:
    import inspect
    from scripts.replay_exchange_event_20260926 import replay
    assert inspect.signature(replay).parameters['prefire_best_play_v6'].default is False
