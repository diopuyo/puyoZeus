"""非同期の既知不具合を再現し、V5の抑止を確認する。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.prefire_best_play_v5 import BestPlayV5Layer, TRACE_COLUMNS
from src.prefire_best_play_stable import StableBestPlayLayer
from src import prefire_v5_search as search
from tests.test_prefire_v5_search import position


def configured() -> tuple[BestPlayV5Layer, SimpleNamespace]:
    """計算結果がまだ届かない最小の台。"""
    layer = BestPlayV5Layer(latency_sec=0.3)
    state = position()
    layer._cache[()] = ((0.8, state, state), (0.8, state, state))
    layer._ready[()] = 1.3
    return layer, SimpleNamespace(probability=0.5, source='G_fe')


def test_old_bug_reproduced() -> None:
    layer = StableBestPlayLayer(latency_sec=0.3)
    layer._cache[()] = (None, None)
    layer._ready[()] = 1.3
    layer._schedule(None, (), (), 0.0, (), (), 1.1)
    assert layer._shown_key == ()  # 完了前でも旧版は表示対象にする。


@pytest.mark.parametrize('stamp', [1.0, 1.1, 1.299999, 1.3, 1.4])
def test_not_used_before_ready(stamp: float) -> None:
    layer, tracker = configured()
    layer._show_v5(tracker, (), (search.OK, search.OK), stamp, 1)
    assert tracker.probability == (0.8 if stamp >= 1.3 else 0.5)
    assert not layer.trace[-1][-1]


@pytest.mark.parametrize('reason', [search.MISSING, search.FIRING, search.INVALID])
@pytest.mark.parametrize('side', [0, 1])
def test_one_side_missing_suppresses_both(reason: int, side: int) -> None:
    layer, tracker = configured()
    statuses = [search.OK, search.OK]
    statuses[side] = reason
    layer._show_v5(tracker, (), tuple(statuses), 2.0, 1)
    assert tracker.probability == 0.5
    assert not dict(zip(TRACE_COLUMNS, layer.trace[-1]))['used']


@pytest.mark.parametrize('latency', [-1.0, np.nan, np.inf])
def test_invalid_latency(latency: float) -> None:
    with pytest.raises(ValueError):
        BestPlayV5Layer(latency)


def test_replay_flag_default_off() -> None:
    import inspect
    from scripts.replay_exchange_event_20260926 import replay
    assert inspect.signature(replay).parameters['prefire_best_play_v5'].default is False


def test_only_explicit_variant_enables_v5() -> None:
    from scripts.run_prefire_replay_20260930 import options
    assert not options('off').get('prefire_best_play_v5', False)
    assert options('v5')['prefire_best_play_v5'] is True


@pytest.mark.parametrize('count_sync, version', [(False, 'v3'), (True, 'v4')])
def test_reference_uses_replay_model(count_sync: bool, version: str) -> None:
    from scripts.run_prefire_replay_20260930 import model_directory
    assert model_directory(dict(count_sync=count_sync)).name == 'exchange_event_'+version
