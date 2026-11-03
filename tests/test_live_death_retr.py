"""retR の反証、期限、通知の固定を確認する。"""
import math
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from src import exchange_event_landing as landing
from src.phase_j.live_death_retr import CURRENT, DeathRetrReader, can_return, capture, side_values, with_snapshot
from tests.test_e10b_exchange_landing import setup_projection
from tests.test_exchange_event_tracker import tracker


@pytest.mark.parametrize('nonfinite', [math.nan, math.inf, -math.inf])
def test_nonfinite_audit_save(tracker: object, monkeypatch: pytest.MonkeyPatch,
                             tmp_path: Path, nonfinite: float) -> None:
    """実際の判定記録を現試合・退避済み試合の両方から厳密 JSON で保存する。"""
    from src.phase_j.live_eval_supervisor import SupervisedOverlay
    projection, overlay, _, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    overlay.death_retr = dict(t_sec=2, game=1,
                             values=((nonfinite, 17), (58, nonfinite)))
    row = projection._evaluate(overlay, snapshot, tuple(h[-1] for h in overlay._history),
                               [0, 20], (1, 1), dict(p1=.8), 2)
    current = dict(records=[row], audits={}, diagnostics=[])
    supervisor = SimpleNamespace(pending=False, dirty=False, archive=[row],
                                 diagnostics=[], request=lambda _: current)
    path = tmp_path / 'events.jsonl'
    SupervisedOverlay.save(supervisor, path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 2
    for saved in rows:
        assert saved['death_retr'] == dict(t_sec=2, game=1, values=[[None, 17], [58, None]])
        json.dumps(saved, allow_nan=False)
    assert not math.isfinite(overlay.death_retr['values'][0][0])
    assert not math.isfinite(overlay.death_retr['values'][1][1])


@pytest.mark.parametrize('residual,cancel,death', [
    (-1, 0, False), (0, 0, False), (1, 1000, False),
    (1, 0, True), (math.nan, math.nan, True), (math.inf, math.inf, True)])
def test_death_refutation(tracker: object, monkeypatch: pytest.MonkeyPatch,
                          residual: float, cancel: float, death: bool) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    overlay.death_retr = dict(t_sec=2, values=((math.nan, math.nan), (residual, cancel)))
    projection.update(overlay, result, snapshot, 2)
    assert (projection.death is not None) is death
    assert tracker.source == ('unavoidable_death' if death else 'S3_landing')


def test_existing_death_released(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    assert projection.death is not None
    overlay.death_retr = dict(t_sec=2, values=((math.nan, math.nan), (-1, 1000)))
    projection.update(overlay, result, snapshot, 2)
    assert projection.death is None
    assert tracker.source == 'S3_landing'


@pytest.mark.parametrize('source,mode,usable', [
    (2, 'original', True), (3, 'original', False), (-1, 'original', False),
    (-.25, 'held', True), (-.75, 'held', False), (math.nan, 'held', False)])
def test_media_age(source: float, mode: str, usable: bool) -> None:
    value = dict(extra={'retR_r_p1': -1}, hold_audit={
        'ex_hold_source_input_t_p1': source, 'ex_hold_mode_p1': mode})
    residual, _ = side_values(value, 'p1', 2, 2)
    assert math.isfinite(residual) is usable


def test_snapshot_and_off(monkeypatch: pytest.MonkeyPatch) -> None:
    args = (None, None, None, 2, 1)
    snap = dict(t_sec=2, values=((-1, 0), (1, 0)))
    CURRENT.set(snap)
    monkeypatch.delenv('PUYO_LIVE_DEATH_CHECK_RETR', raising=False)
    assert with_snapshot(args) is args
    monkeypatch.setenv('PUYO_LIVE_DEATH_CHECK_RETR', '1')
    frozen = with_snapshot(args)
    CURRENT.set(None)
    assert frozen[-1] is snap and len(frozen) == 9
    assert with_snapshot(frozen) is frozen
    assert can_return(snap, 0, 10, 2)
    assert not can_return(snap, 0, 10, 3)


@pytest.mark.parametrize('generation,game,accepted', [(4, 2, True), (3, 2, False), (4, 1, False)])
def test_capture_boundary(monkeypatch: pytest.MonkeyPatch, generation: int,
                          game: int, accepted: bool) -> None:
    monkeypatch.setenv('PUYO_LIVE_DEATH_CHECK_RETR', '1')
    calls = []
    value = dict(extra={'retR_r_p2': -1}, hold_audit={
        'ex_hold_source_input_t_p2': 2, 'ex_hold_mode_p2': 'original'})
    def select(*args: object) -> dict:
        calls.append(args)
        return value
    unified = SimpleNamespace(epoch=generation, game=game, clock=lambda: 50,
                              worker=SimpleNamespace(fresh_sec=2),
                              death_retr_reader=SimpleNamespace(select=select))
    b3 = SimpleNamespace(live24h=unified, epoch=1, generation=3, adopted_game=1)
    notice = SimpleNamespace(captured_at=49, b3=(dict(t_sec=2, game_idx=2, adopted_game=2), [], [], 1))
    capture(b3, notice)
    assert (CURRENT.get() is not None) is accepted
    assert bool(calls) is accepted
    if accepted:
        assert calls[0][-1] == 49
        assert can_return(CURRENT.get(), 1, 10, 2)


def test_config_json_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import os
    from src.phase_j import live_speed
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    assert live_speed.validate({})['live_death_check_retr'] is False
    assert live_speed.validate({'live_death_check_retr': True})['live_death_check_retr'] is True
    parser = argparse.ArgumentParser()
    live_speed.add_arguments(parser)
    options = parser.parse_args(['--live-death-check-retr'])
    options.source = 'video'
    monkeypatch.setenv('PUYO_LIVE_DEATH_CHECK_RETR', '0')
    live_speed.configure(options, parser)
    assert os.environ['PUYO_LIVE_DEATH_CHECK_RETR'] == '1'


def test_reader_hold_expiry_and_reset() -> None:
    key = dict(key=('board', 1), fight=True, input_t=2, resets=[])
    source = dict(input_t=2, eligible_at=49, extra={'retR_r_p2': -1})
    worker = SimpleNamespace(epoch=1, key=key, available=lambda *a, **k: (source, 'available'))
    reader = DeathRetrReader(2)
    value = reader.select(worker, 50, 2, 49)
    assert side_values(value, 'p2', 2, 2)[0] == -1
    worker.available = lambda *a, **k: (None, 'not_ready_or_stale')
    value = reader.select(worker, 50.25, 2.25, 49.25)
    assert side_values(value, 'p2', 2.25, 2)[0] == -1
    value = reader.select(worker, 51, 3, 50)
    assert math.isnan(side_values(value, 'p2', 3, 2)[0])
    worker.epoch = 2
    value = reader.select(worker, 51, 3, 50)
    assert math.isnan(side_values(value, 'p2', 3, 2)[0])
