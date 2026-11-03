"""先読みしない経路の状態と行を、学習生成器へ照合する。"""
from pathlib import Path

import numpy as np
import pytest

from src.b3_realtime import QueueObservation, RealtimeRows
from src.b3_rows import generate_rows, _inside_match
from src.b3_source import iter_frames, load_events, load_lean, obs_game_ranges, winners_by_overlap
from src.b3_timing import TimingTable
from tests.b3_synth import build_record


def test_parity_can_disable_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """一致検査の計時OFFでは時計自体を読まない。"""
    def unexpected_clock() -> float:
        raise AssertionError('計時OFFで時計を読んだ')

    monkeypatch.setattr('src.b3_realtime.perf_counter', unexpected_clock)
    engine = RealtimeRows(TimingTable(), measure_timing=False)
    engine.update(next(iter_frames(build_record(tmp_path))))
    assert engine.feature_ms == 0.0


def test_stream_equals_training(tmp_path: Path) -> None:
    record = build_record(tmp_path)
    lean, events = load_lean(record), load_events(record)
    _, mapping = winners_by_overlap(lean, obs_game_ranges(record))
    expected, _ = generate_rows(record, 'synth', TimingTable())
    engine = RealtimeRows(TimingTable(), 'synth')
    actual = []
    for d in iter_frames(record):
        queues = tuple(QueueObservation(int(i), str(lean.side[i]), int(lean.frame[i]),
            float(lean.t_sec[i]), tuple(lean.queue[i])) for i in np.where(lean.frame == d['frame'])[0])
        found = tuple(e for e in events if e['frame'] == d['frame'])
        row = engine.update(d, found, queues, _inside_match(d['t_sec'], mapping.get(d['game_idx'], {})))
        if row is not None:
            actual.append(row)
    assert len(actual) == len(expected) > 0
    for a, b in zip(actual, expected):
        for key in a.keys() - {'won_p1', 'weight', 'held_sec'}:
            if isinstance(a[key], str):
                assert a[key] == b[key]
            else:
                np.testing.assert_equal(a[key], b[key], err_msg=key)


def test_future_input_rejected(tmp_path: Path) -> None:
    d = next(iter_frames(build_record(tmp_path)))
    d['available_at_sec'] += 1
    with pytest.raises(RuntimeError, match='未来'):
        RealtimeRows(TimingTable()).update(d)


def test_display_matches_stage2() -> None:
    from src.b3_live_display import B3Display
    from src.b3_stage2_trace import smooth_display
    times = np.arange(90) / 30
    raw = np.where(times < 1, .7, .2)
    sources = np.where(times < 1, 'static', 'exchange')
    physical = np.array([1.])
    expected = smooth_display(raw, sources, np.ones(90), times, physical, None)
    # 段2の従来EMAとの互換性は、既定ONの高速EMAを明示OFFにして比較する。
    display = B3Display(fast_ema=False)
    display.boundary(1)
    actual = []
    for t, p in zip(times, raw):
        display.prediction(float(p), int(t >= 1), float(t))
        actual.append(display.update(float(t), physical_count=int(t >= 1)))
    np.testing.assert_array_equal(np.asarray(actual).T, expected)


def test_death_failure_and_boundary() -> None:
    from src.b3_live_display import B3Display
    display = B3Display()
    display.boundary(1)
    display.prediction(.7, 0, 0.)
    display.failed = True
    display.update(.51)
    assert display.raw is None
    assert display.update(.6, 'confirmed_death', 1.) == (100., 1.)
    display.boundary(2)
    assert display.update(.7) == (0., .5)
