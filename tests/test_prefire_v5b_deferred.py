"""固定遅れ再生の遅延実計算は表示・因果性を変えない。"""
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.prefire_v5b_deferred import DeferredComputations
from src.prefire_best_play_v5 import BestPlayV5Layer
from src.prefire_v5_search import OK, MISSING
from tests.test_prefire_v5_search import position


class FastLayer(BestPlayV5Layer):
    """入力凍結が見える決定的な安価な評価器。"""

    def _schedule_v5(self, overlay: Any, key: tuple, states: tuple, statuses: tuple,
                     elapsed: float, t_sec: float) -> None:
        if key in self._cache:
            return
        probability = overlay._snapshots[-1][1].value
        best = (probability, states[0], states[1])
        self._cache[key] = (best, best) if statuses == (OK, OK) else (None, None)
        self._ready[key] = t_sec + self.latency_sec


def timeline(latency: float, missing: bool) -> np.ndarray:
    layer = FastLayer(latency)
    snapshot = SimpleNamespace(value=.8)
    overlay = SimpleNamespace(_snapshots=[(0, snapshot)], _m0=None, _build_static=None,
                              tracker=SimpleNamespace(models=None))
    states = (position(), position())
    statuses = (MISSING, OK) if missing else (OK, OK)
    for stamp, key in ((0., ('a',)), (.1, ('b',)), (.25, ('b',)), (.5, ('b',))):
        tracker = SimpleNamespace(probability=.5, source='G_fe')
        if stamp in (0., .1):
            snapshot.value = .8 if stamp == 0 else .6
            layer._schedule_v5(overlay, key, states, statuses, stamp, stamp)
        else:
            snapshot.value = .9
        layer._show_v5(tracker, key, statuses, stamp, 1)
    return np.asarray(layer.trace, float)


@pytest.mark.parametrize('latency', [0., .3, 10.])
@pytest.mark.parametrize('missing', [False, True])
def test_deferred_is_identical_and_freezes_input(latency: float, missing: bool) -> None:
    expected = timeline(latency, missing)
    adapter = DeferredComputations(FastLayer)
    adapter.install()
    try:
        actual = timeline(latency, missing)
    finally:
        adapter.restore()
    np.testing.assert_array_equal(actual, expected)
    if latency == .3 and not missing:
        assert adapter.counts() == dict(submitted=2, cancelled=1, completed=1, pending=0)
        assert actual[-1, 3] == .6
