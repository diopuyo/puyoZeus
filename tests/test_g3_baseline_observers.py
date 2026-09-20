"""基準の採否/元resetを変えず、観測不備と元例外を区別する。"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
from scripts import g3_baseline_observers as O
from tests.test_g3_admission import A, T, B, collector, owned_replace, pipeline, admission_root  # noqa: F401  admission_root は autouse fixture。採録先の根を一時領域へ差し替える (W45)


class Recorder:
    def __init__(self, *, broken: bool = False) -> None:
        self.broken, self.time_sec = broken, 0.0
        self.rows: list[dict] = []

    @property
    def frame(self) -> int:
        if self.broken:
            raise LookupError('capture_failed')
        return 0

    def emit(self, value: dict) -> None:
        self.rows.append(value)


def test_observe_without_filter(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    c = collector
    pipe, _ = pipeline(c, monkeypatch)
    rec, state = Recorder(), dict(output=tmp_path)
    emit = c._should_emit
    acc, side, shared = c._LeanNpzAccumulator(), c._SideState(), c._SharedGameCounter()
    with ExitStack() as stack:
        stats = O.install(stack, c, rec, state, owned_replace(), source_id='artificial-baseline')
        pipe.update(B.O.FIRST, B.O.FIRST / 60, object())
        T.call(c, acc, side, shared, B.O.FIRST, '1P')
        assert c._should_emit is emit and side.last_emitted_grid is not None
        assert state[A.KEY].latest['status'] == A.UNKNOWN and state[A.KEY].decisions == 0
    assert stats['closed'] and stats['forbidden_repairs'] == 0


@pytest.mark.parametrize('broken,reset_error', [(False, False), (True, False), (True, True)])
def test_reset_observation(collector: Any, monkeypatch: Any, tmp_path: Path,
                           broken: bool, reset_error: bool) -> None:
    c = collector
    pipe, _ = pipeline(c, monkeypatch)
    calls: list[Any] = []
    def reset(self: Any, match_start_sec: Any = None) -> str:
        calls.append(match_start_sec)
        if reset_error:
            raise ValueError('original_reset_failed')
        return 'original_return'
    monkeypatch.setattr(c.RecognitionPipeline, 'reset', reset)
    rec, state = Recorder(broken=broken), dict(output=tmp_path)
    with ExitStack() as stack:
        stats = O.install(stack, c, rec, state, owned_replace(), source_id='artificial-baseline')
        if reset_error:
            with pytest.raises(ValueError, match='original_reset_failed'):
                pipe.reset(match_start_sec=0.0)
        elif broken:
            with pytest.raises(RuntimeError, match='reset_observation_failed'):
                pipe.reset(match_start_sec=0.0)
        else:
            assert pipe.reset(match_start_sec=0.0) == 'original_return'
        assert calls == [0.0] and stats['forbidden_repairs'] == 0
        assert rec.rows[0]['evidence']['episode_id'] is None
        assert rec.rows[0]['evidence']['observed_source_id'] == 'artificial-baseline'
    assert c.RecognitionPipeline.reset is reset and stats['closed']
